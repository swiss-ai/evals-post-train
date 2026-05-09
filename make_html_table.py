#!/usr/bin/env python3

import os
import argparse
import wandb
import re


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


def compute_mgsm_avg(summary):
    vals = []
    for k, v in summary.items():
        if "mgsm" in k and k.endswith("exact_match") and "stderr" not in k:
            try:
                vals.append(float(v))
            except (ValueError, TypeError):
                pass
    return sum(vals) / len(vals) if vals else None


def get_metric(summary, metric):
    if "mgsm" in metric.lower() and "mgsm_en" not in metric.lower():
        return compute_mgsm_avg(summary)

    if metric in summary:
        return summary[metric]

    parts = metric.split("/", 1)
    task = parts[0]
    metric_name = parts[1] if len(parts) > 1 else None

    metric_filter = None
    if metric_name and "," in metric_name:
        metric_name, metric_filter = metric_name.split(",", 1)

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


def _prepare_group_data(groups, models, scores, metric_filter):
    """Build group_data and overall_avgs for a subset of metrics selected by metric_filter."""
    group_data = []
    all_model_scores = {m: [] for m in models}

    for group_name, metric_list in groups:
        filtered = [(k, d) for k, d in metric_list if metric_filter(k)]
        if not filtered:
            continue
        group_scores = {m: [] for m in models}
        benchmarks = []
        for metric_key, display_name in filtered:
            row = {}
            for model in models:
                val = scores.get(model, {}).get(metric_key)
                row[model] = val
                if val is not None:
                    group_scores[model].append(val)
                    all_model_scores[model].append(val)
            benchmarks.append((display_name, row))
        group_avgs = {}
        for m in models:
            vals = group_scores[m]
            group_avgs[m] = sum(vals) / len(vals) if vals else None
        group_data.append((group_name, benchmarks, group_avgs))

    overall_avgs = {}
    for m in models:
        vals = all_model_scores[m]
        overall_avgs[m] = sum(vals) / len(vals) if vals else None
    return group_data, overall_avgs


def _render_table(html, table_id, models, group_data, overall_avgs, n_models, info_rows=None):
    """Render one table-box with the given group_data."""
    html.append(f'<div class="table-box" id="{table_id}">\n')
    html.append('  <div class="grid-row header-row">\n    <div class="cell">Category / Benchmark</div>\n')
    for model in models:
        html.append(f'    <div class="cell">{model}</div>\n')
    html.append("  </div>\n")

    for group_name, benchmarks, group_avgs in group_data:
        valid_gavgs = [v for v in group_avgs.values() if v is not None]
        best_gavg = max(valid_gavgs) if valid_gavgs else None

        html.append('  <div class="section"><details>\n  <summary class="grid-row">\n')
        html.append(f'    <div class="cell"><span class="arrow">&#9654;</span>{group_name}</div>\n')
        for model in models:
            avg = group_avgs[model]
            if avg is None:
                html.append('    <div class="cell na">-</div>\n')
            else:
                is_best = (best_gavg is not None and abs(avg - best_gavg) < 1e-6 and len(valid_gavgs) > 1)
                content = f'<span class="best-badge">{avg * 100:.1f}</span>' if is_best else f'{avg * 100:.1f}'
                html.append(f'    <div class="cell">{content}</div>\n')
        html.append('  </summary>\n  <div class="bench-rows">\n')

        for display_name, row_vals in benchmarks:
            valid_vals = [v for v in row_vals.values() if v is not None]
            best_val = max(valid_vals) if valid_vals else None

            html.append(f'    <div class="bench-row">\n      <div class="cell">{display_name}</div>\n')
            for model in models:
                val = row_vals[model]
                if val is None:
                    html.append('      <div class="cell na">-</div>\n')
                else:
                    is_best = (best_val is not None and abs(val - best_val) < 1e-6 and len(valid_vals) > 1)
                    content = f'<span class="best-badge">{val * 100:.1f}</span>' if is_best else f'{val * 100:.1f}'
                    html.append(f'      <div class="cell">{content}</div>\n')
            html.append("    </div>\n")

        html.append("  </div>\n  </details></div>\n")

    valid_overall = [v for v in overall_avgs.values() if v is not None]
    best_overall = max(valid_overall) if valid_overall else None

    html.append('  <div class="grid-row overall">\n    <div class="cell">Overall Average</div>\n')
    for model in models:
        avg = overall_avgs[model]
        if avg is None:
            html.append('    <div class="cell na">-</div>\n')
        else:
            is_best = (best_overall is not None and abs(avg - best_overall) < 1e-6 and len(valid_overall) > 1)
            content = f'<span class="best-badge">{avg * 100:.1f}</span>' if is_best else f'{avg * 100:.1f}'
            html.append(f'    <div class="cell">{content}</div>\n')
    html.append("  </div>\n")

    if info_rows:
        for label, row_vals in info_rows:
            html.append(f'  <div class="grid-row info-row">\n    <div class="cell">{label}</div>\n')
            for model in models:
                val = row_vals.get(model)
                if val is None:
                    html.append('    <div class="cell na">-</div>\n')
                else:
                    html.append(f'    <div class="cell">{val:.1f}</div>\n')
            html.append("  </div>\n")

    html.append("</div>\n")


def build_html(groups, models, scores, output_path, info_rows=None):
    """Generate a self-contained HTML file with collapsible skill groups and train/test toggle."""

    train_group_data, train_overall = _prepare_group_data(
        groups, models, scores, lambda k: k not in TEST_BENCHMARKS)
    test_group_data, test_overall = _prepare_group_data(
        groups, models, scores, lambda k: k in TEST_BENCHMARKS)

    train_benchmarks = sum(len(b) for _, b, _ in train_group_data)
    test_benchmarks = sum(len(b) for _, b, _ in test_group_data)

    n_models = len(models)

    html = []
    html.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Evaluation Dashboard</title>
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
    <h1>Evaluation Results</h1>
    <p class="subtitle">{len(models)} models &middot; {train_benchmarks} train benchmarks &middot; {test_benchmarks} test benchmarks</p>
  </div>
  <div class="controls">
    <div class="toggle-group">
      <button id="btn-train" class="active">Training</button>
      <button id="btn-test">Test</button>
    </div>
    <button id="btn-collapse">Collapse All</button>
    <button id="btn-expand" class="primary">Expand All</button>
  </div>
</div>

""")

    html.append('<div id="wrap-train">\n')
    _render_table(html, "table-train", models, train_group_data, train_overall, n_models, info_rows=info_rows)
    html.append('</div>\n')

    html.append('<div id="wrap-test" style="display:none;">\n')
    _render_table(html, "table-test", models, test_group_data, test_overall, n_models, info_rows=info_rows)
    html.append('</div>\n')

    html.append("""
<script>
  var wrapTrain = document.getElementById('wrap-train');
  var wrapTest = document.getElementById('wrap-test');
  var btnTrain = document.getElementById('btn-train');
  var btnTest = document.getElementById('btn-test');

  btnTrain.addEventListener('click', function() {
    wrapTrain.style.display = '';
    wrapTest.style.display = 'none';
    btnTrain.classList.add('active');
    btnTest.classList.remove('active');
  });
  btnTest.addEventListener('click', function() {
    wrapTrain.style.display = 'none';
    wrapTest.style.display = '';
    btnTest.classList.add('active');
    btnTrain.classList.remove('active');
  });

  document.getElementById('btn-expand').addEventListener('click', function() {
    var visible = wrapTrain.style.display !== 'none' ? wrapTrain : wrapTest;
    visible.querySelectorAll('details').forEach(function(d) { d.open = true; });
  });
  document.getElementById('btn-collapse').addEventListener('click', function() {
    var visible = wrapTrain.style.display !== 'none' ? wrapTrain : wrapTest;
    visible.querySelectorAll('details').forEach(function(d) { d.open = false; });
  });
</script>
</body>
</html>""")

    with open(output_path, "w") as f:
        f.write("".join(html))
    print(f"Saved HTML table to {output_path}")


PINNED_PREFIX = [
    ("baseline-apertus-1-sft", "Apertus 1.0 8B SFT"),
    ("baseline-apertus-1-instruct", "Apertus 1.0 8B Instruct"),
    ("ap-1p5-cooldown-sft-21-04-lr-8e-5", "Apertus 1.5 SFT"),
]

OPTIONAL_INSTRUCT_BASELINE = (
    "apertus-v1.5-sft-1.5--DPO-1.5--online-DPO-R16-beta0.1-bs256-lnF-lr5e-6-ml2048",
    "Apertus 1.5 Instruct",
)

OPTIONAL_OLMO_BASELINES = [
    ("OLMO-3-7B-SFT", "OLMo 3 7B SFT"),
    ("OLMO-3-7B-DPO", "OLMo 3 7B DPO"),
]

PINNED_SUFFIX = [
    ("OLMO-3-7B-Instruct", "OLMo 3 7B Instruct"),
]

ALWAYS_PINNED = {run for run, _ in PINNED_PREFIX + PINNED_SUFFIX}


def fetch_model(api, project_path, run_name, display_name, groups, scores, debug=False, summaries=None):
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
            val = normalize_score(get_metric(summary, metric_key))
            model_scores[metric_key] = val

    scores[display_name] = model_scores
    if summaries is not None:
        summaries[display_name] = summary
    return display_name


def main():
    parser = argparse.ArgumentParser(description="Generate an HTML eval table from W&B results")
    parser.add_argument("--metrics-file", required=True,
                        help="Metrics file with # group headers (e.g. tasks_posttrain_final_main_table.txt)")
    parser.add_argument("--models", nargs="*", default=[],
                        help="Additional W&B run names to include between pinned columns")
    parser.add_argument("--entity", default="apertus")
    parser.add_argument("--project", default="apertus-1.5-post-training-v0.0")
    parser.add_argument("--output", default="eval_table.html", help="Output HTML file path")
    parser.add_argument("--rename", nargs="*", default=[],
                        help="Rename models for display: 'long-run-name=Short Name'")
    parser.add_argument("--instruct-baseline", action="store_true",
                        help="Include Apertus-1.5-Instruct column after Apertus-1.5-SFT")
    parser.add_argument("--olmo-baselines", action="store_true",
                        help="Include Olmo-3-7B-SFT and Olmo-3-7B-DPO columns")
    parser.add_argument("--show-word-count", action="store_true",
                        help="Show alpaca_eval avg_word_count as an info row")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    rename_map = {}
    for pair in args.rename:
        old, new = pair.split("=", 1)
        rename_map[old] = new

    groups = parse_metrics_file(args.metrics_file)

    api = wandb.Api()
    project_path = f"{args.entity}/{args.project}"

    scores = {}
    summaries = {}
    display_names = []
    fetched_runs = set()

    for run_name, display in PINNED_PREFIX:
        if fetch_model(api, project_path, run_name, display, groups, scores, args.debug, summaries):
            display_names.append(display)
            fetched_runs.add(run_name)

    if args.instruct_baseline:
        run_name, display = OPTIONAL_INSTRUCT_BASELINE
        if fetch_model(api, project_path, run_name, display, groups, scores, args.debug, summaries):
            display_names.append(display)
            fetched_runs.add(run_name)

    for model_name in args.models:
        if model_name in fetched_runs or model_name in ALWAYS_PINNED:
            continue
        display = rename_map.get(model_name, model_name)
        if fetch_model(api, project_path, model_name, display, groups, scores, args.debug, summaries):
            display_names.append(display)

    if args.olmo_baselines:
        for run_name, display in OPTIONAL_OLMO_BASELINES:
            if fetch_model(api, project_path, run_name, display, groups, scores, args.debug, summaries):
                display_names.append(display)

    for run_name, display in PINNED_SUFFIX:
        if fetch_model(api, project_path, run_name, display, groups, scores, args.debug, summaries):
            display_names.append(display)

    if not display_names:
        print("No models found. Nothing to render.")
        return

    info_rows = None
    if args.show_word_count:
        row_vals = {}
        for display in display_names:
            summary = summaries.get(display, {})
            val = get_metric(summary, "alpaca_eval/avg_word_count")
            row_vals[display] = float(val) if val is not None else None
        info_rows = [("Alpaca Eval Avg Word Count", row_vals)]

    build_html(groups, display_names, scores, args.output, info_rows=info_rows)


if __name__ == "__main__":
    main()