#!/usr/bin/env python3

import os
import json
import math
import argparse
import wandb
import wandb.sdk.lib.server
import re

_orig_query_with_timeout = wandb.sdk.lib.server.Server.query_with_timeout

def _patched_query_with_timeout(self):
    try:
        _orig_query_with_timeout(self)
    except TypeError:
        if hasattr(self, "_viewer") and self._viewer:
            flags = self._viewer.get("flags")
            self._flags = json.loads(flags) if isinstance(flags, str) else {}
        else:
            self._flags = {}

wandb.sdk.lib.server.Server.query_with_timeout = _patched_query_with_timeout


TEST_BENCHMARKS = {
    "agieval/acc",
    "arc_challenge_chat/exact_match",
    "arc_multilingual/acc",
    "gpqa_main_cot_zeroshot/exact_match,ordered-extract",
    "gsm8k_platinum_cot_zeroshot/exact_match",
    "mlogiqa_gen/exact_match",
}


def parse_metrics_file(path):
    """Parse metrics file with # Group headers. Returns list of (group_name, [(metric_key, display_name), ...])."""
    groups = []
    current_group = None
    current_metrics = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if current_group is not None:
                    groups.append((current_group, current_metrics))
                current_group = line.lstrip("# ").strip()
                current_metrics = []
            else:
                display = line.split("/")[0]
                current_metrics.append((line, display))
    if current_group is not None:
        groups.append((current_group, current_metrics))
    return groups


# --- Thinking / reasoning metrics ---
# The harness records these for every generate_until task; the uploader strips the ",none"
# filter suffix, so a W&B summary key is exactly "<task>/<metric>". Rates live in [0, 1] and
# render as percentages; counts are raw word/char/token counts and must never be rescaled.
THINKING_COUNT_SUFFIXES = frozenset({
    "response_length_words", "response_length_chars", "response_length_tokens",
    "thinking_length_words", "thinking_length_chars", "thinking_length_tokens",
})


def metric_kind(metric_key):
    """"count" for a raw length metric, "rate" for everything else."""
    return "count" if metric_key.split("/", 1)[-1] in THINKING_COUNT_SUFFIXES else "rate"


def thinking_specs(length_unit="tokens", format_detail=False):
    """(group label, metric suffix) pairs in render order. The unit lives in the label because
    updateBest() rewrites the text of every cell it touches."""
    specs = []
    if format_detail:
        specs.append(("Thinking Format: Has Open (%)", "thinking_format_has_open"))
        specs.append(("Thinking Format: Has Close (%)", "thinking_format_has_close"))
    specs.append(("Thinking Format Correct (%)", "thinking_format_correct"))
    specs.append((f"Thinking Length ({length_unit})", f"thinking_length_{length_unit}"))
    specs.append((f"Response Length ({length_unit})", f"response_length_{length_unit}"))
    return specs


def thinking_tasks_from_file(path):
    """Task names from a suite file: either a `tasks_*.txt` list or a `*_main_table.txt`.

    Not parse_metrics_file(): that returns nothing for a file with no `#` header, and several
    main tables have none. Inline comments are stripped, matching how the graceful launcher
    reads the same task lists (`line="${line%%#*}"`); some suites really do use them.
    """
    seen, tasks = set(), []
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            task = line.split("/")[0].strip()
            if task and task not in seen:
                seen.add(task)
                tasks.append(task)
    return tasks


def build_thinking_groups(tasks, specs, label_with_group=False):
    """One group per metric family, one row per task -- the transpose of a main table.

    `label_with_group` folds the family into the row label, for flat rendering where the
    group headers (which carry the unit) are not drawn.
    """
    return [
        (label, [(f"{task}/{suffix}", f"{task} - {label}" if label_with_group else task) for task in tasks])
        for label, suffix in specs
    ]


def compute_mgsm_avg(summary):
    vals = []
    for k, v in summary.items():
        if "mgsm" in k and k.endswith("exact_match") and "stderr" not in k:
            try:
                vals.append(float(v))
            except (ValueError, TypeError):
                pass
    return sum(vals) / len(vals) if vals else None


def compute_degeneration_avg(summary):
    """Average all metrics across benchmarks whose key contains '/degeneration'."""
    vals = []
    for k, v in summary.items():
        if "stderr" in k:
            continue
        if "/degeneration" in k.lower():
            nv = normalize_score(v)
            if nv is not None:
                vals.append(nv)
    return sum(vals) / len(vals) if vals else None


def compute_mmlu_pro_detail(summary):
    """Extract MMLU-Pro overall accuracy and per-subject breakdown from a run summary.

    Summary keys are flattened as '<task>/exact_match,<filter>'. The group task
    'mmlu_pro' is the overall accuracy; 'mmlu_pro_<subject>' are the per-subject scores
    (underscores in the subject become spaces, e.g. computer_science -> 'computer science').
    Returns None if no MMLU-Pro keys are present.
    """
    accuracy = None
    subjects = {}
    for k, v in summary.items():
        if "stderr" in k:
            continue
        kl = k.lower()
        if "mmlu_pro" not in kl or "exact_match" not in kl:
            continue
        task = k.split("/", 1)[0]
        try:
            val = float(v)
        except (ValueError, TypeError):
            continue
        if task == "mmlu_pro":
            accuracy = val
        elif task.startswith("mmlu_pro_"):
            subjects[task[len("mmlu_pro_"):].replace("_", " ")] = val
    if accuracy is None and not subjects:
        return None
    return {"accuracy": accuracy, "subjects": subjects}


def is_mgsm_aggregate(task, metric_name):
    """True when a request means "the MGSM average across languages".

    MGSM is many per-language subtasks, so a request for its accuracy is answered by
    averaging them. Keying this on the task name alone would answer *every* mgsm request
    that way -- including a thinking length -- so the requested metric has to match what
    compute_mgsm_avg() actually averages.
    """
    if metric_name != "exact_match":
        return False
    task = task.lower()
    return "mgsm" in task and "mgsm_en" not in task


def get_metric(summary, metric, fuzzy=True):
    """Resolve a `task/metric[,filter]` key against a W&B run summary.

    Main-table keys are hand-written and usually omit the harness's filter suffix, so by
    default we fall back to a substring scan. Pass fuzzy=False for keys we synthesized
    ourselves (they must match exactly, and guessing would silently cross-match siblings).
    """
    parts = metric.split("/", 1)
    task = parts[0]
    metric_name = parts[1] if len(parts) > 1 else None

    metric_filter = None
    if metric_name and "," in metric_name:
        metric_name, metric_filter = metric_name.split(",", 1)

    if is_mgsm_aggregate(task, metric_name):
        return compute_mgsm_avg(summary)

    if metric in summary:
        return summary[metric]

    if not fuzzy:
        return None

    candidates = []
    for k, v in summary.items():
        if "stderr" in k:
            continue
        k_lower = k.lower()
        if task.lower() not in k_lower:
            continue
        if metric_name and metric_name.lower() not in k_lower:
            continue
        if metric_filter and metric_filter.lower() not in k_lower:
            continue
        candidates.append((k, v))

    if not candidates:
        return None

    if "gsm8k" in task.lower():
        for k, v in candidates:
            if "strict-match" in k:
                return float(v)

    candidates.sort(key=lambda x: len(x[0]))
    return candidates[0][1]


def normalize_score(val):
    if val is None:
        return None
    val = float(val)
    if val > 1.0 or val < 0.0:
        return val / 100.0
    return val


def _prepare_group_data(groups, models, scores, metric_filter, thinking=False):
    """Build group_data and overall_avgs for a subset of metrics selected by metric_filter.

    Each group_data entry is (group_name, benchmarks, group_avgs, kind). Group averages are an
    unweighted macro-mean over the tasks that resolved; for a count group that is the mean of
    each task's mean length. Counts never enter overall_avgs -- averaging a length with a rate
    is meaningless.

    `kind` is only derived in thinking mode, where the groups are metric families we
    synthesized and every value in a group therefore shares a unit. A hand-written main table
    can mix units inside one group, and its values have already been through normalize_score,
    so classifying them by key suffix there would render the whole group wrong.
    """
    group_data = []
    all_model_scores = {m: [] for m in models}

    for group_name, metric_list in groups:
        filtered = [(k, d) for k, d in metric_list if metric_filter(k)]
        if not filtered and metric_list:
            continue
        kind = metric_kind(filtered[0][0]) if (thinking and filtered) else "rate"
        group_scores = {m: [] for m in models}
        benchmarks = []
        for metric_key, display_name in filtered:
            row = {model: scores.get(model, {}).get(metric_key) for model in models}
            # Tasks that emit no thinking metrics (loglikelihood / multiple-choice) resolve to
            # None for every model; drop them rather than render a row of dashes.
            if thinking and all(v is None for v in row.values()):
                continue
            for model, val in row.items():
                if val is not None:
                    group_scores[model].append(val)
                    if kind == "rate":
                        all_model_scores[model].append(val)
            benchmarks.append((display_name, row))
        if thinking and not benchmarks:
            continue
        group_avgs = {}
        for m in models:
            vals = group_scores[m]
            group_avgs[m] = sum(vals) / len(vals) if vals else None
        group_data.append((group_name, benchmarks, group_avgs, kind))

    overall_avgs = {}
    for m in models:
        vals = all_model_scores[m]
        overall_avgs[m] = sum(vals) / len(vals) if vals else None
    return group_data, overall_avgs


def _na_cell(col, indent):
    return f'{indent}<div class="cell na" data-col="{col}">-</div>\n'


def _value_cell(val, col, kind, indent):
    """One data cell. A count carries no data-val, which keeps it out of updateBest()'s
    `.score-row [data-val]` sweep: raw lengths are never rescaled and never win a "best"
    badge, because a shorter trace is not automatically a better one."""
    if val is None:
        return _na_cell(col, indent)
    if kind == "count":
        # Round half up: "{412.5:.0f}" is banker's rounding and yields 412.
        return f'{indent}<div class="cell" data-col="{col}">{math.floor(val + 0.5):.0f}</div>\n'
    return f'{indent}<div class="cell" data-col="{col}" data-val="{val * 100:.1f}">{val * 100:.1f}</div>\n'


def _row_classes(base, kind):
    return base if kind == "count" else f"{base} score-row"


def _render_table(html, table_id, models, group_data, overall_avgs, n_models, info_rows=None, flat=False):
    """Render one table-box with the given group_data."""
    # An overall average is only meaningful when every group shares a unit. A table carrying a
    # count group (raw token lengths) alongside a rate group has nothing to average.
    show_overall = all(kind == "rate" for _, _, _, kind in group_data)
    html.append(f'<div class="table-box" id="{table_id}">\n')
    html.append(f'  <div class="grid-row header-row">\n    <div class="cell">{"Benchmark" if flat else "Category / Benchmark"}</div>\n')
    for i, model in enumerate(models):
        html.append(f'    <div class="cell" data-col="{i+1}">{model}</div>\n')
    html.append("  </div>\n")

    if flat:
        for _, benchmarks, _, kind in group_data:
            for display_name, row_vals in benchmarks:
                html.append(f'  <div class="{_row_classes("bench-row", kind)}">\n    <div class="cell">{display_name}</div>\n')
                for i, model in enumerate(models):
                    html.append(_value_cell(row_vals[model], i + 1, kind, "    "))
                html.append("  </div>\n")
    else:
        for group_name, benchmarks, group_avgs, kind in group_data:
            if not benchmarks:
                html.append('  <div class="section"><div class="grid-row empty-group">\n')
                html.append(f'    <div class="cell">{group_name}</div>\n')
                for i, model in enumerate(models):
                    html.append(_na_cell(i + 1, "    "))
                html.append("  </div></div>\n")
                continue

            html.append(f'  <div class="section"><details>\n  <summary class="{_row_classes("grid-row", kind)}">\n')
            html.append(f'    <div class="cell"><span class="arrow">&#9654;</span>{group_name}</div>\n')
            for i, model in enumerate(models):
                html.append(_value_cell(group_avgs[model], i + 1, kind, "    "))
            html.append('  </summary>\n  <div class="bench-rows">\n')

            for display_name, row_vals in benchmarks:
                html.append(f'    <div class="{_row_classes("bench-row", kind)}">\n      <div class="cell">{display_name}</div>\n')
                for i, model in enumerate(models):
                    html.append(_value_cell(row_vals[model], i + 1, kind, "      "))
                html.append("    </div>\n")

            html.append("  </div>\n  </details></div>\n")

    if show_overall:
        html.append('  <div class="grid-row overall score-row">\n    <div class="cell">Overall Average</div>\n')
        for i, model in enumerate(models):
            html.append(_value_cell(overall_avgs[model], i + 1, "rate", "    "))
        html.append("  </div>\n")

    if info_rows:
        for label, row_vals in info_rows:
            html.append(f'  <div class="grid-row info-row">\n    <div class="cell">{label}</div>\n')
            for i, model in enumerate(models):
                val = row_vals.get(model)
                if val is None:
                    html.append(_na_cell(i + 1, "    "))
                else:
                    html.append(f'    <div class="cell" data-col="{i+1}">{val:.1f}</div>\n')
            html.append("  </div>\n")

    html.append("</div>\n")


def build_html(groups, models, scores, output_path, info_rows=None, olmo_default_hidden=False, apertus_default_hidden=False, apertus_baselines=None, no_split=False, flat=False, title="Evaluation Results", thinking=False):
    """Generate a self-contained HTML file with collapsible skill groups and train/test toggle."""

    def _count(group_data):
        # In thinking mode every task appears once per metric family, so summing the group
        # row-counts would report 3x the number of benchmarks actually evaluated. Count the
        # distinct tasks that resolved instead -- from the metric keys, since the display
        # label carries the family name under --flat.
        if thinking:
            return len({
                key.split("/")[0]
                for _, metric_list in groups
                for key, _ in metric_list
                if any(scores.get(m, {}).get(key) is not None for m in models)
            })
        return sum(len(b) for _, b, _, _ in group_data)

    if no_split:
        all_group_data, all_overall = _prepare_group_data(
            groups, models, scores, lambda k: True, thinking=thinking)
        total_benchmarks = _count(all_group_data)
    else:
        train_group_data, train_overall = _prepare_group_data(
            groups, models, scores, lambda k: k not in TEST_BENCHMARKS, thinking=thinking)
        test_group_data, test_overall = _prepare_group_data(
            groups, models, scores, lambda k: k in TEST_BENCHMARKS, thinking=thinking)

    if no_split:
        train_benchmarks = total_benchmarks
        test_benchmarks = 0
    else:
        train_benchmarks = _count(train_group_data)
        test_benchmarks = _count(test_group_data)

    n_models = len(models)

    html = []
    html.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {{
  --bg-color: #f3f4f6;
  --card-bg: #ffffff;
  --text-main: #111827;
  --text-muted: #6b7280;
  --border-color: #e5e7eb;
  --row-hover: #f9fafb;
  --primary: #4f46e5;
  --primary-hover: #4338ca;
  --best-bg: #dcfce7;
  --best-text: #166534;
  --header-bg: #f8fafc;
}}

* {{ margin: 0; padding: 0; box-sizing: border-box; }}

body {{
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: var(--bg-color);
  color: var(--text-main);
  padding: 40px 5%;
  max-width: 1800px;
  margin: 0 auto;
  line-height: 1.5;
}}

.header-container {{
  margin-bottom: 24px;
}}

.title-area h1 {{ font-size: 28px; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 6px; color: #111827; }}
.title-area .subtitle {{ color: var(--text-muted); font-size: 14px; font-weight: 500; }}

.controls {{ display: flex; gap: 12px; margin-top: 16px; align-items: center; }}
.controls button {{
  background: var(--card-bg);
  color: var(--text-main);
  border: 1px solid var(--border-color);
  padding: 8px 16px;
  border-radius: 6px;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
  box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}}
.controls button:hover {{ background: var(--row-hover); border-color: #d1d5db; }}
.controls button.primary {{ background: var(--primary); color: white; border-color: var(--primary); }}
.controls button.primary:hover {{ background: var(--primary-hover); }}

.toggle-group {{
  display: flex;
  border: 1px solid var(--border-color);
  border-radius: 6px;
  overflow: hidden;
  box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}}
.toggle-group button {{
  border: none;
  border-radius: 0;
  box-shadow: none;
  border-right: 1px solid var(--border-color);
}}
.toggle-group button:last-child {{ border-right: none; }}
.toggle-group button.active {{
  background: var(--primary);
  color: white;
}}

.table-box {{
  --col-template: 280px repeat({n_models}, minmax(0, 1fr));
  background: var(--card-bg);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
  overflow: hidden;
}}

.grid-row {{
  display: grid;
  grid-template-columns: var(--col-template);
  align-items: stretch;
}}

.header-row {{
  background: var(--header-bg);
  border-bottom: 2px solid var(--border-color);
  position: sticky;
  top: 0;
  z-index: 10;
}}
.header-row .cell {{
  padding: 16px 16px;
  font-size: 13px;
  font-weight: 600;
  color: var(--text-muted);
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  border-left: 1px solid var(--border-color);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}}
.header-row .cell:first-child {{ justify-content: flex-start; border-left: none; }}

.section {{ border-bottom: 1px solid var(--border-color); }}
.section:last-child {{ border-bottom: none; }}

.empty-group {{ background: #f5f7fa; }}
.empty-group .cell {{
  padding: 14px 16px;
  font-size: 14px;
  font-weight: 500;
  display: flex;
  align-items: center;
  justify-content: center;
  border-left: 1px solid var(--border-color);
  font-family: 'JetBrains Mono', monospace;
  color: var(--text-muted);
}}
.empty-group .cell:first-child {{
  justify-content: flex-start;
  border-left: none;
  font-family: 'Inter', sans-serif;
  color: var(--text-main);
}}

details {{ margin: 0; padding: 0; width: 100%; }}

summary {{
  padding: 0;
  margin: 0;
  cursor: pointer;
  list-style: none;
  background: #eef2f7;
  transition: background 0.15s ease;
  width: 100%;
}}
summary::-webkit-details-marker {{ display: none; }}
summary::marker {{ display: none; content: ""; }}
summary:hover {{ background: #e2e8f0; }}

details[open] summary {{ border-bottom: 1px solid var(--border-color); background: #e2e8f0; }}

summary .cell {{
  padding: 14px 16px;
  font-size: 14px;
  font-weight: 500;
  display: flex;
  align-items: center;
  justify-content: center;
  border-left: 1px solid var(--border-color);
  font-family: 'JetBrains Mono', monospace;
}}
summary .cell:first-child {{
  justify-content: flex-start;
  border-left: none;
  font-family: 'Inter', sans-serif;
}}

summary .arrow {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  border-radius: 4px;
  font-size: 10px;
  color: var(--text-muted);
  margin-right: 10px;
  transition: transform 0.2s ease, background 0.2s ease;
}}
summary:hover .arrow {{ background: #e2e8f0; }}
details[open] .arrow {{ transform: rotate(90deg); }}

.bench-rows {{ background: #ffffff; margin: 0; padding: 0; width: 100%; }}
.bench-row {{
  display: grid;
  grid-template-columns: var(--col-template);
  align-items: stretch;
  border-bottom: 1px solid var(--border-color);
  transition: background 0.15s;
}}
.bench-row:last-child {{ border-bottom: none; }}
.bench-row:hover {{ background: var(--card-bg); }}

.bench-row .cell {{
  padding: 10px 16px;
  font-size: 13px;
  font-weight: 400;
  display: flex;
  align-items: center;
  justify-content: center;
  border-left: 1px solid var(--border-color);
  color: var(--text-muted);
  font-family: 'JetBrains Mono', monospace;
}}
.bench-row .cell:first-child {{
  justify-content: flex-start;
  padding-left: 46px;
  color: var(--text-muted);
  border-left: none;
  font-family: 'Inter', sans-serif;
}}

.overall {{
  background: var(--header-bg);
  border-top: 3px solid var(--border-color);
}}
.overall .cell {{
  padding: 18px 16px;
  font-size: 15px;
  font-weight: 500;
  display: flex;
  align-items: center;
  justify-content: center;
  border-left: 1px solid var(--border-color);
  font-family: 'JetBrains Mono', monospace;
}}
.overall .cell:first-child {{ justify-content: flex-start; border-left: none; font-family: 'Inter', sans-serif; text-transform: uppercase; color: var(--text-main); font-weight: 600; }}

.info-row {{
  background: var(--card-bg);
  border-top: 1px solid var(--border-color);
}}
.info-row .cell {{
  padding: 12px 16px;
  font-size: 13px;
  font-weight: 400;
  display: flex;
  align-items: center;
  justify-content: center;
  border-left: 1px solid var(--border-color);
  color: var(--text-muted);
  font-family: 'JetBrains Mono', monospace;
}}
.info-row .cell:first-child {{ justify-content: flex-start; border-left: none; font-family: 'Inter', sans-serif; color: var(--text-main); font-weight: 500; }}

.na {{ color: #9ca3af; font-style: italic; }}
.best-badge {{
  background: var(--best-bg);
  color: var(--best-text);
  padding: 4px 10px;
  border-radius: 6px;
  font-weight: 400;
  display: inline-block;
  min-width: 45px;
  box-shadow: inset 0 0 0 1px rgba(22, 101, 52, 0.2);
}}

</style>
</head>
<body>

<div class="header-container">
  <div class="title-area">
    <h1>{title}</h1>
    <p class="subtitle">{len(models)} models &middot; {f'{total_benchmarks} benchmarks' if no_split else f'{train_benchmarks} train benchmarks &middot; {test_benchmarks} test benchmarks'}</p>
  </div>
  <div class="controls">
    {'<div class="toggle-group"><button id="btn-train" class="active">Training</button><button id="btn-test">Test</button></div>' if not no_split else ''}
    <div class="toggle-group">
      <button id="btn-all" class="active">All</button>
      <button id="btn-sft">SFT</button>
      <button id="btn-instruct">Instruct</button>
    </div>
    <button id="btn-apertus" class="{'active' if not apertus_default_hidden else ''}">Apertus</button>
    <button id="btn-olmo" class="{'active' if not olmo_default_hidden else ''}">OLMo</button>
    {'<div class="toggle-group"><button id="btn-collapse">Collapse</button><button id="btn-expand">Expand</button></div>' if not flat else ''}
  </div>
</div>

""")

    model_types_js = '[' + ','.join(
        f'"sft"' if 'sft' in m.lower() and 'instruct' not in m.lower() and 'dpo' not in m.lower()
        else f'"instruct"' for m in models
    ) + ']'

    is_olmo_js = '[' + ','.join(
        'true' if 'olmo' in m.lower() else 'false' for m in models
    ) + ']'

    _apertus_set = apertus_baselines or set()
    is_apertus_js = '[' + ','.join(
        'true' if m in _apertus_set else 'false' for m in models
    ) + ']'

    if no_split:
        html.append('<div id="wrap-all">\n')
        _render_table(html, "table-all", models, all_group_data, all_overall, n_models, info_rows=info_rows, flat=flat)
        html.append('</div>\n')
    else:
        html.append('<div id="wrap-train">\n')
        _render_table(html, "table-train", models, train_group_data, train_overall, n_models, info_rows=info_rows, flat=flat)
        html.append('</div>\n')

        html.append('<div id="wrap-test" style="display:none;">\n')
        _render_table(html, "table-test", models, test_group_data, test_overall, n_models, info_rows=info_rows, flat=flat)
        html.append('</div>\n')

    html.append(f"""
<script>
  var noSplit = {'true' if no_split else 'false'};
  var wrapTrain = document.getElementById('wrap-train');
  var wrapTest = document.getElementById('wrap-test');
  var wrapAll = document.getElementById('wrap-all');
  var btnTrain = document.getElementById('btn-train');
  var btnTest = document.getElementById('btn-test');
  var btnOlmo = document.getElementById('btn-olmo');
  var btnApertus = document.getElementById('btn-apertus');
  var modelTypes = {model_types_js};
  var isOlmo = {is_olmo_js};
  var isApertusBaseline = {is_apertus_js};
  var nModels = modelTypes.length;
  var currentTypeFilter = 'all';
  var showOlmo = {'false' if olmo_default_hidden else 'true'};
  var showApertus = {'false' if apertus_default_hidden else 'true'};

  if (!noSplit) {{
    btnTrain.addEventListener('click', function() {{
      wrapTrain.style.display = '';
      wrapTest.style.display = 'none';
      btnTrain.classList.add('active');
      btnTest.classList.remove('active');
    }});
    btnTest.addEventListener('click', function() {{
      wrapTrain.style.display = 'none';
      wrapTest.style.display = '';
      btnTest.classList.add('active');
      btnTrain.classList.remove('active');
    }});
  }}

  var btnExpand = document.getElementById('btn-expand');
  var btnCollapse = document.getElementById('btn-collapse');
  if (btnExpand) {{
    btnExpand.addEventListener('click', function() {{
      var target = noSplit ? wrapAll : (wrapTrain.style.display !== 'none' ? wrapTrain : wrapTest);
      target.querySelectorAll('details').forEach(function(d) {{ d.open = true; }});
    }});
  }}
  if (btnCollapse) {{
    btnCollapse.addEventListener('click', function() {{
      var target = noSplit ? wrapAll : (wrapTrain.style.display !== 'none' ? wrapTrain : wrapTest);
      target.querySelectorAll('details').forEach(function(d) {{ d.open = false; }});
    }});
  }}

  btnOlmo.addEventListener('click', function() {{
    showOlmo = !showOlmo;
    btnOlmo.classList.toggle('active', showOlmo);
    applyFilters();
  }});

  btnApertus.addEventListener('click', function() {{
    showApertus = !showApertus;
    btnApertus.classList.toggle('active', showApertus);
    applyFilters();
  }});

  function applyFilters() {{
    var visibleCols = [];
    for (var i = 0; i < nModels; i++) {{
      var typeMatch = (currentTypeFilter === 'all' || modelTypes[i] === currentTypeFilter);
      var olmoMatch = (showOlmo || !isOlmo[i]);
      var apertusMatch = (showApertus || !isApertusBaseline[i]);
      if (typeMatch && olmoMatch && apertusMatch) visibleCols.push(i + 1);
    }}

    var template = '280px repeat(' + visibleCols.length + ', minmax(0, 1fr))';
    document.querySelectorAll('.table-box').forEach(function(box) {{
      box.style.setProperty('--col-template', template);
    }});

    document.querySelectorAll('[data-col]').forEach(function(cell) {{
      var col = parseInt(cell.getAttribute('data-col'));
      cell.style.display = visibleCols.indexOf(col) !== -1 ? '' : 'none';
    }});
    updateBest();
  }}

  function filterModels(type) {{
    currentTypeFilter = type;
    document.getElementById('btn-all').classList.toggle('active', type === 'all');
    document.getElementById('btn-sft').classList.toggle('active', type === 'sft');
    document.getElementById('btn-instruct').classList.toggle('active', type === 'instruct');
    applyFilters();
  }}

  function updateBest() {{
    document.querySelectorAll('.score-row').forEach(function(row) {{
      var cells = row.querySelectorAll('[data-val]');
      var visible = [];
      cells.forEach(function(c) {{
        if (c.style.display !== 'none') visible.push(c);
      }});
      var maxVal = -Infinity;
      visible.forEach(function(c) {{
        var v = parseFloat(c.getAttribute('data-val'));
        if (v > maxVal) maxVal = v;
      }});
      visible.forEach(function(c) {{
        var v = parseFloat(c.getAttribute('data-val'));
        if (visible.length > 1 && Math.abs(v - maxVal) < 0.01) {{
          c.innerHTML = '<span class="best-badge">' + v.toFixed(1) + '</span>';
        }} else {{
          c.textContent = v.toFixed(1);
        }}
      }});
    }});
  }}

  applyFilters();
  document.getElementById('btn-all').addEventListener('click', function() {{ filterModels('all'); }});
  document.getElementById('btn-sft').addEventListener('click', function() {{ filterModels('sft'); }});
  document.getElementById('btn-instruct').addEventListener('click', function() {{ filterModels('instruct'); }});
</script>
</body>
</html>""")

    with open(output_path, "w") as f:
        f.write("".join(html))
    print(f"Saved HTML table to {output_path}")


def build_json(groups, models, scores, output_path, info_rows=None, summaries=None, thinking=False):
    """Write combined metrics to JSON: grouped by task, with per-group and overall averages.

    Score values are the normalized fractions (0-1) used internally by the table -- except for
    "count"-kind groups (thinking/response lengths), which are stored raw. The group/overall
    averages exclude missing (null) entries, matching the HTML rendering. In thinking mode the
    overall average is omitted, since it would mix rates with token counts.
    When summaries are provided, an 'mmlu_pro_detail' section with overall accuracy and
    per-subject scores is added for each model that has MMLU-Pro results.
    """
    out_groups = []
    all_vals = {m: [] for m in models}

    for group_name, metric_list in groups:
        # See _prepare_group_data: kind is only meaningful for the families we synthesized.
        kind = metric_kind(metric_list[0][0]) if (thinking and metric_list) else "rate"
        metrics_out = []
        group_vals = {m: [] for m in models}
        for metric_key, display_name in metric_list:
            row = {m: scores.get(m, {}).get(metric_key) for m in models}
            if thinking and all(v is None for v in row.values()):
                continue
            for m, v in row.items():
                if v is not None:
                    group_vals[m].append(v)
                    if kind == "rate":
                        all_vals[m].append(v)
            metrics_out.append({"key": metric_key, "display": display_name, "scores": row})
        if thinking and not metrics_out:
            continue
        group_avg = {m: (sum(group_vals[m]) / len(group_vals[m]) if group_vals[m] else None) for m in models}
        group_out = {"name": group_name, "average": group_avg, "metrics": metrics_out}
        if thinking:
            # Only thinking groups carry a unit worth naming; adding it elsewhere would change
            # the schema of every existing --json-output.
            group_out["kind"] = kind
        out_groups.append(group_out)

    result = {"models": models, "groups": out_groups}
    if not thinking:
        result["overall_average"] = {
            m: (sum(all_vals[m]) / len(all_vals[m]) if all_vals[m] else None) for m in models
        }
    if info_rows:
        result["info_rows"] = {label: {m: row_vals.get(m) for m in models} for label, row_vals in info_rows}

    if summaries is not None:
        mmlu_detail = {}
        for m in models:
            detail = compute_mmlu_pro_detail(summaries.get(m, {}))
            if detail is not None:
                mmlu_detail[m] = detail
        if mmlu_detail:
            result["mmlu_pro_detail"] = mmlu_detail

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved JSON metrics to {output_path}")


BASELINE_PROJECT = "apertus/apertus-1.5-post-training-v0.0"

PINNED_PREFIX = [
    ("baseline-apertus-1-sft", "Apertus 1.0 8B SFT"),
    ("baseline-apertus-1-instruct", "Apertus 1.0 8B Instruct"),
    ("ap-1p5-cooldown-sft-21-04-lr-8e-5", "Apertus 1.5 8B SFT"),
]

OPTIONAL_INSTRUCT_BASELINE = (
    "apertus-v1.5-sft-1.5--DPO-1.5--online-DPO-R16-beta0.1-bs256-lnF-lr5e-6-ml2048",
    "Apertus 1.5 8B Instruct",
)

PINNED_SUFFIX = [
    ("OLMO-3-7B-SFT", "OLMo 3 7B SFT"),
    ("OLMO-3-7B-DPO", "OLMo 3 7B DPO"),
    ("OLMO-3-7B-Instruct", "OLMo 3 7B Instruct"),
]

ALWAYS_PINNED = {run for run, _ in PINNED_PREFIX + PINNED_SUFFIX}


def fetch_model(api, project_path, run_name, display_name, groups, scores, debug=False, summaries=None, thinking=False):
    try:
        matched = list(api.runs(project_path, filters={"display_name": run_name}, order="-created_at", per_page=1))
    except Exception as e:
        print(f"Warning: failed to fetch run '{run_name}': {e}")
        return None

    if not matched:
        print(f"Warning: run '{run_name}' not found in {project_path}")
        return None

    summary = dict(matched[0].summary)

    if debug:
        print(f"\n=== Keys for {run_name} ===")
        for k in sorted(summary.keys()):
            print(f"  {k}: {summary[k]}")

    model_scores = {}
    for _, metric_list in groups:
        for metric_key, _ in metric_list:
            if thinking:
                # Keys are synthesized, so demand an exact hit. Counts stay raw:
                # normalize_score() would turn 612.4 tokens into 6.124.
                raw = get_metric(summary, metric_key, fuzzy=False)
                raw = None if raw is None else float(raw)
                val = raw if metric_kind(metric_key) == "count" else normalize_score(raw)
            else:
                val = normalize_score(get_metric(summary, metric_key))
            model_scores[metric_key] = val

    scores[display_name] = model_scores
    if summaries is not None:
        summaries[display_name] = summary
    return display_name


def main():
    parser = argparse.ArgumentParser(description="Generate an HTML eval table from W&B results")
    parser.add_argument("--metrics-file", required=True,
                        help="Metrics file with # group headers (e.g. tasks_posttrain_final_main_table.txt). "
                             "With --thinking this is read as a task source instead: any tasks_*.txt list or "
                             "*_main_table.txt works, only the task names are used.")
    parser.add_argument("--thinking", action="store_true",
                        help="Render a thinking-only table: one group per thinking metric family, one row per "
                             "task. Baselines and the train/test split are disabled; lengths render raw.")
    parser.add_argument("--length-unit", choices=["tokens", "words", "chars"], default="tokens",
                        help="Unit for the thinking/response length groups (default: tokens). "
                             "Prefer chars when comparing across backends.")
    parser.add_argument("--thinking-format-detail", action="store_true",
                        help="With --thinking, also show thinking_format_has_open / has_close groups.")
    parser.add_argument("--models", nargs="*", default=[],
                        help="Additional W&B run names to include between pinned columns")
    parser.add_argument("--models-file", default=None,
                        help="Path to a txt file with one W&B run name per line")
    parser.add_argument("--entity", default="apertus")
    parser.add_argument("--project", default="apertus-1.5-post-training-v0.0")
    parser.add_argument("--output", default="eval_table.html", help="Output HTML file path")
    parser.add_argument("--json-output", default=None,
                        help="Also write combined metrics (grouped by task, with group + overall averages) to this JSON path. Only includes models passed via --models/--models-file, no baselines.")
    parser.add_argument("--rename", nargs="*", default=[],
                        help="Rename models for display: 'long-run-name=Short Name'")
    parser.add_argument("--instruct-baseline", action="store_true",
                        help="Include Apertus-1.5-Instruct column after Apertus-1.5-SFT")
    parser.add_argument("--show-word-count", action="store_true",
                        help="Show alpaca_eval avg_word_count as an info row")
    parser.add_argument("--more-details", action="store_true",
                        help="Show averaged degeneration (across all benchmarks) as an info row")
    parser.add_argument("--overrefusal", action="store_true",
                        help="Show orbench overrefusal rate as an info row")
    parser.add_argument("--no-baselines", action="store_true",
                        help="Skip all baseline models; only use models from --models / --models-file")
    parser.add_argument("--no-split", action="store_true",
                        help="Disable train/test split; show all benchmarks in a single table")
    parser.add_argument("--flat", action="store_true",
                        help="Show all benchmarks as plain rows without group headers")
    parser.add_argument("--title", default="Evaluation Results",
                        help="Page title for the HTML report")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--sft-baseline", default=None,
                        help="Override the Apertus 1.5 8B SFT baseline run name")
    parser.add_argument("--no-sft-baseline", action="store_true",
                        help="Exclude the Apertus 1.5 8B SFT baseline column")
    args = parser.parse_args()

    all_models = list(args.models)
    if args.models_file:
        with open(args.models_file) as f:
            for line in f:
                name = line.strip()
                if name and not name.startswith("#"):
                    all_models.append(name)

    rename_map = {}
    for pair in args.rename:
        old, new = pair.split("=", 1)
        rename_map[old] = new

    if args.thinking:
        tasks = thinking_tasks_from_file(args.metrics_file)
        # Under --flat the group headers are not drawn, and they carry the unit, so the
        # family has to move into the row label or a rate and a token count look alike.
        groups = build_thinking_groups(
            tasks, thinking_specs(args.length_unit, args.thinking_format_detail),
            label_with_group=args.flat,
        )
        print(f"Thinking mode: {len(tasks)} task(s) x {len(groups)} metric group(s) from {args.metrics_file}")
    else:
        groups = parse_metrics_file(args.metrics_file)

    api = wandb.Api()
    project_path = f"{args.entity}/{args.project}"

    scores = {}
    summaries = {}
    display_names = []
    requested_display_names = []
    fetched_runs = set()
    apertus_baselines = set()

    if args.sft_baseline:
        PINNED_PREFIX[2] = (args.sft_baseline, "Apertus 1.5 8B SFT")

    prefix_baselines = PINNED_PREFIX
    if args.no_sft_baseline:
        prefix_baselines = [b for b in PINNED_PREFIX if b[1] != "Apertus 1.5 8B SFT"]

    # The pinned baselines were evaluated without thinking, so every one of their thinking
    # columns would be empty. The all-None column drop below would remove them anyway; not
    # fetching them just saves the W&B round-trips. Compare against a thinking run via --models.
    if not args.no_baselines and not args.thinking:
        for run_name, display in prefix_baselines:
            if fetch_model(api, BASELINE_PROJECT, run_name, display, groups, scores, args.debug, summaries):
                display_names.append(display)
                fetched_runs.add(run_name)
                apertus_baselines.add(display)

        if args.instruct_baseline:
            run_name, display = OPTIONAL_INSTRUCT_BASELINE
            if fetch_model(api, BASELINE_PROJECT, run_name, display, groups, scores, args.debug, summaries):
                display_names.append(display)
                fetched_runs.add(run_name)
                apertus_baselines.add(display)

    for model_name in all_models:
        if model_name in fetched_runs or (model_name in ALWAYS_PINNED and not args.thinking):
            continue
        display = rename_map.get(model_name, model_name)
        if fetch_model(api, project_path, model_name, display, groups, scores, args.debug, summaries, thinking=args.thinking):
            display_names.append(display)
            requested_display_names.append(display)

    if not args.thinking:
        for run_name, display in PINNED_SUFFIX:
            if fetch_model(api, BASELINE_PROJECT, run_name, display, groups, scores, args.debug, summaries):
                display_names.append(display)

    if args.thinking:
        # A model with no thinking metrics at all (thinking never enabled for its run) would
        # otherwise contribute a column of dashes.
        keep = [d for d in display_names if any(v is not None for v in scores.get(d, {}).values())]
        for dropped in [d for d in display_names if d not in keep]:
            print(f"Warning: '{dropped}' has no thinking metrics; dropping its column")
        display_names = keep
        requested_display_names = [d for d in requested_display_names if d in keep]

    if not display_names:
        print("No models found. Nothing to render.")
        return

    info_rows = []
    if args.show_word_count:
        wc_metrics = [
            "alpaca_eval/avg_word_count",
            "arena_hard_v2/avg_word_count",
            "arena_hard_v01/avg_word_count",
        ]
        row_vals = {}
        for display in display_names:
            summary = summaries.get(display, {})
            vals = []
            for mk in wc_metrics:
                v = get_metric(summary, mk)
                if v is not None:
                    vals.append(float(v))
            row_vals[display] = sum(vals) / len(vals) if vals else None
        info_rows.append(("Mean Avg Word Count", row_vals))
    if args.more_details:
        deg_vals = {}
        for display in display_names:
            summary = summaries.get(display, {})
            avg = compute_degeneration_avg(summary)
            deg_vals[display] = avg * 100 if avg is not None else None
        info_rows.append(("Degeneration", deg_vals))
    if args.overrefusal:
        ref_vals = {}
        for display in display_names:
            summary = summaries.get(display, {})
            val = get_metric(summary, "orbench/refusal")
            ref_vals[display] = normalize_score(val) * 100 if val is not None else None
        info_rows.append(("Overrefusal", ref_vals))
    if not info_rows:
        info_rows = None

    if args.json_output:
        build_json(groups, requested_display_names, scores, args.json_output, info_rows=info_rows,
                   summaries=summaries, thinking=args.thinking)

    build_html(groups, display_names, scores, args.output, info_rows=info_rows,
               olmo_default_hidden=args.no_baselines, apertus_baselines=apertus_baselines,
               no_split=args.no_split or args.thinking, flat=args.flat, title=args.title,
               thinking=args.thinking)


if __name__ == "__main__":
    main()