#!/usr/bin/env python3
"""Build benchmark comparison tables from lm-evaluation-harness results.

The metrics file is expected to contain entries like:

    # Math
    gsm8k_cot/exact_match
    humaneval_instruct/pass@10

Comments become domain names. Results are read from lm-eval ``results_*.json``
files, with merged split outputs preferred when they exist.
"""

from __future__ import print_function

import argparse
import csv
import json
import math
import re
import sys
import textwrap
from collections import OrderedDict
from pathlib import Path


KNOWN_METRICS = [
    "length_controlled_winrate",
    "prompt_level_loose_acc",
    "pass_at_1",
    "exact_match",
    "acc_norm",
    "pass@10",
    "pass@1",
    "cometkiwi22",
    "score",
    "acc",
    "f1",
]

DOMAIN_LABELS = {
    "General ability & Knowledge": "General",
    "Instruction Following": "Instruction",
    "Cultural knowledge": "Cultural",
}

MODEL_ALIASES = {
    "apertus-1-base": "Apertus v1 Base",
    "apertus-1-it": "Apertus v1 Instruct",
    "v1-base-sft-21-04-lr-8e-5": "SFT on v1 Base",
    "1p5-cooldown-sft-stage3": "v1.5 stage 3 CPT (with SFT data x1)",
    "1p5-cooldownsft-stage3-sft-lr-8e-5": "SFT on v1.5 stage 3 (with SFT data x1)",
    "1p5-cooldownsftx5-stage3-sft-lr-8e-5": "SFT on v1.5 stage 3 (SFT x 5)",
    "1p5-cooldown-sft-21-04-lr-8e-5-ep2": "SFT on v1.5 stage 3 (without SFT)",
}

DEFAULT_EXCLUDED_BENCHMARKS = {"alpaca_eval"}

SCORE_LABELS = {
    "acc": "acc",
    "acc_norm": "acc norm",
    "cometkiwi22": "COMETKiwi",
    "exact_match": "EM",
    "f1": "F1",
    "length_controlled_winrate": "LC winrate",
    "pass@1": "pass@1",
    "pass@10": "pass@10",
    "pass_at_1": "pass@1",
    "prompt_level_loose_acc": "prompt loose acc",
    "score": "score",
}


def repo_root():
    return Path(__file__).resolve().parents[1]


def clean_comment(line):
    text = line.lstrip("#").strip()
    text = text.lstrip("#").strip()
    return text


def split_metric_line(line):
    """Return (benchmark, metric) from benchmark/metric, with typo recovery."""
    text = line.strip()
    if "/" in text:
        benchmark, metric = text.rsplit("/", 1)
        return benchmark.strip(), metric.strip()

    # Recover lines like "mmlu_flan_cot_zeroshotexact_match,ordered-extract".
    for metric in sorted(KNOWN_METRICS, key=len, reverse=True):
        idx = text.find(metric)
        if idx > 0:
            return text[:idx].strip("_- /"), text[idx:].strip()

    raise ValueError("Metric line has no '/' separator: {}".format(line))


def parse_metrics_file(path):
    specs = []
    current_domain = "Uncategorized"

    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("#"):
                comment = clean_comment(line)
                if comment:
                    current_domain = comment
                continue

            # Allow trailing comments after a metric entry.
            line = line.split("#", 1)[0].strip()
            if not line:
                continue

            benchmark, metric = split_metric_line(line)
            specs.append(
                {
                    "domain": current_domain,
                    "benchmark": benchmark,
                    "metric": metric,
                    "raw": line,
                    "line_no": line_no,
                }
            )

    return specs


def parse_output_paths_from_log(path):
    output_paths = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except TypeError:
        text = path.read_text(errors="ignore")

    for match in re.finditer(r"--output_path\s+(\S+)", text):
        output_paths.append(Path(match.group(1)))
    return output_paths


def discover_result_files(log_roots):
    result_files = OrderedDict()

    for root in log_roots:
        root = Path(root).expanduser()
        if not root.exists():
            continue

        for result_file in root.rglob("results_*.json"):
            result_files[str(result_file.resolve())] = result_file

        for log_file in root.rglob("*.out"):
            for output_path in parse_output_paths_from_log(log_file):
                if output_path.exists():
                    for result_file in output_path.rglob("results_*.json"):
                        result_files[str(result_file.resolve())] = result_file

    return list(result_files.values())


def result_file_info(path):
    parts = list(path.parts)
    run_name = path.parent.name
    eval_dir = path.parent.name

    if "harness" in parts:
        idx = parts.index("harness")
        if idx > 0:
            run_name = parts[idx - 1]
        if idx + 1 < len(parts):
            eval_dir = parts[idx + 1]

    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0

    return {
        "path": path,
        "run": run_name,
        "eval_dir": eval_dir,
        "is_merged": eval_dir.startswith("eval_merged"),
        "mtime": mtime,
    }


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def expand_model_args(model_args):
    if not model_args:
        return []

    models = []
    for arg in model_args:
        for model in arg.split(","):
            model = model.strip()
            if model:
                models.append(model)
    return models


def build_runs(result_files, run_filters=None, model_names=None, prefer_merged=True):
    grouped = OrderedDict()
    model_set = set(model_names or [])

    for path in result_files:
        info = result_file_info(path)
        run = info["run"]
        if model_set and run not in model_set:
            continue
        if run_filters and not any(re.search(pattern, run) for pattern in run_filters):
            continue
        grouped.setdefault(run, []).append(info)

    runs = OrderedDict()
    ordered_runs = model_names if model_names else sorted(grouped)
    for run in ordered_runs:
        if run not in grouped:
            continue
        infos = grouped[run]
        if prefer_merged:
            merged = [info for info in infos if info["is_merged"]]
            if merged:
                newest = sorted(merged, key=lambda x: (x["mtime"], str(x["path"])))[-1]
                infos = [newest]

        combined = OrderedDict()
        files_used = []
        for info in sorted(infos, key=lambda x: (x["mtime"], str(x["path"]))):
            try:
                data = load_json(info["path"])
            except Exception as exc:
                print(
                    "WARNING: failed to read {}: {}".format(info["path"], exc),
                    file=sys.stderr,
                )
                continue
            combined.update(data.get("results", {}))
            files_used.append(info["path"])

        if combined:
            runs[run] = {"results": combined, "files": files_used}

    return runs


def metric_candidates(metric_spec):
    if "," in metric_spec:
        return [metric_spec]
    return [metric_spec + ",none", metric_spec]


def get_metric_value(task_result, metric_spec):
    if not task_result:
        return None, None

    for key in metric_candidates(metric_spec):
        value = task_result.get(key)
        if is_number(value):
            return value, key

    metric_prefix = metric_spec.split(",", 1)[0]
    matches = []
    for key, value in task_result.items():
        if key == "alias" or "_stderr" in key:
            continue
        if key.split(",", 1)[0] == metric_prefix and is_number(value):
            matches.append((key, value))

    if not matches:
        return None, None

    matches.sort(key=lambda item: item[0])
    return matches[0][1], matches[0][0]


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def scaled_value(value, scale_mode):
    if value is None:
        return None
    if scale_mode == "raw":
        return value
    if scale_mode == "percent":
        return value * 100.0
    if abs(value) <= 1.0:
        return value * 100.0
    return value


def display_value(value, scale_mode, precision):
    shown = scaled_value(value, scale_mode)
    if shown is None:
        return ""
    if math.isnan(shown):
        return ""
    return ("{0:." + str(precision) + "f}").format(shown)


def average(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / float(len(values))


def build_tables(specs, runs, scale_mode="auto", precision=3):
    run_names = list(runs.keys())
    detail_rows = []
    values_by_domain = OrderedDict()
    all_values = {run: [] for run in run_names}
    missing = []

    for spec in specs:
        row = OrderedDict()
        row["domain"] = spec["domain"]
        row["benchmark"] = spec["benchmark"]
        row["metric"] = spec["metric"]
        row_values = {}

        values_by_domain.setdefault(spec["domain"], {run: [] for run in run_names})

        for run in run_names:
            task_result = runs[run]["results"].get(spec["benchmark"])
            value, matched_key = get_metric_value(task_result, spec["metric"])
            row_values[run] = value
            row[run] = display_value(value, scale_mode, precision)
            if value is None:
                missing.append((run, spec["benchmark"], spec["metric"]))
            else:
                shown_value = scaled_value(value, scale_mode)
                all_values[run].append(shown_value)
                values_by_domain[spec["domain"]][run].append(shown_value)

        detail_rows.append(row)

    summary_rows = []
    overall = OrderedDict()
    overall["summary"] = "Overall average"
    for run in run_names:
        overall[run] = display_value(average(all_values[run]), "raw", precision)
    summary_rows.append(overall)

    for domain, domain_values in values_by_domain.items():
        row = OrderedDict()
        row["summary"] = "{} average".format(domain)
        for run in run_names:
            row[run] = display_value(average(domain_values[run]), "raw", precision)
        summary_rows.append(row)

    return summary_rows, detail_rows, missing


def benchmark_column_name(detail_row):
    return "{}: {}/{}".format(
        detail_row.get("domain", ""),
        detail_row.get("benchmark", ""),
        detail_row.get("metric", ""),
    )


def model_label(run_name):
    return MODEL_ALIASES.get(run_name, run_name)


def transpose_tables(summary_rows, detail_rows, run_names):
    summary_fields = ["model"] + [row["summary"] for row in summary_rows]
    summary_model_rows = []

    for run in run_names:
        row = OrderedDict()
        row["model"] = model_label(run)
        for summary_row in summary_rows:
            row[summary_row["summary"]] = summary_row.get(run, "")
        summary_model_rows.append(row)

    benchmark_fields = [benchmark_column_name(row) for row in detail_rows]
    main_fields = summary_fields + benchmark_fields
    main_rows = []

    for run in run_names:
        row = OrderedDict()
        row["model"] = model_label(run)
        for summary_row in summary_rows:
            row[summary_row["summary"]] = summary_row.get(run, "")
        for detail_row, field in zip(detail_rows, benchmark_fields):
            row[field] = detail_row.get(run, "")
        main_rows.append(row)

    return summary_fields, summary_model_rows, main_fields, main_rows


def slugify(value):
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower())
    slug = slug.strip("_")
    return slug or "table"


def metric_column_name(detail_row):
    return "{}/{}".format(detail_row.get("benchmark", ""), detail_row.get("metric", ""))


def compact_domain_label(domain):
    return DOMAIN_LABELS.get(domain, domain)


def pretty_metric_text(value):
    value = str(value)
    value = value.replace("_", " ")
    value = value.replace(",", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def pretty_score_text(value):
    chunks = []
    for part in re.split(r"[,/]", str(value)):
        part = part.strip()
        if not part:
            continue
        chunks.append(SCORE_LABELS.get(part, pretty_metric_text(part)))
    return " ".join(chunks)


def split_metric_label(value):
    if "/" not in value:
        return "", value
    benchmark, metric = value.rsplit("/", 1)
    return benchmark, metric


def display_label(field, mode="markdown"):
    separator = "<br>" if mode == "markdown" else "\n"

    if field == "model":
        return "Model"

    if field == "Overall average":
        return "Overall"

    if field.endswith(" average"):
        domain = field[: -len(" average")]
        return compact_domain_label(domain)

    if ": " in field:
        domain, metric_spec = field.split(": ", 1)
        benchmark, metric = split_metric_label(metric_spec)
        parts = [
            compact_domain_label(domain),
            pretty_metric_text(benchmark),
            pretty_score_text(metric),
        ]
        return separator.join(part for part in parts if part)

    if "/" in field:
        benchmark, metric = split_metric_label(field)
        parts = [pretty_metric_text(benchmark), pretty_score_text(metric)]
        return separator.join(part for part in parts if part)

    return pretty_metric_text(field)


def display_labels(fields, mode="markdown"):
    return [display_label(field, mode=mode) for field in fields]


def wrap_text(value, width, break_long_words=False):
    value = str(value)
    if not value:
        return value
    pieces = []
    for line in value.split("\n"):
        wrapped = textwrap.wrap(
            line,
            width=width,
            break_long_words=break_long_words,
            break_on_hyphens=True,
        )
        pieces.extend(wrapped or [""])
    return "\n".join(pieces)


def build_domain_tables(detail_rows, run_names):
    domains = OrderedDict()
    for detail_row in detail_rows:
        domain = detail_row.get("domain", "Uncategorized")
        domains.setdefault(domain, []).append(detail_row)

    tables = OrderedDict()
    for domain, rows_for_domain in domains.items():
        fields = ["model"] + [metric_column_name(row) for row in rows_for_domain]
        table_rows = []
        for run in run_names:
            row = OrderedDict()
            row["model"] = model_label(run)
            for detail_row, field in zip(rows_for_domain, fields[1:]):
                row[field] = detail_row.get(run, "")
            table_rows.append(row)
        tables[domain] = (fields, table_rows)

    return tables


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_escape(value):
    return str(value).replace("|", "\\|")


def write_markdown(path, summary_rows, summary_fields, main_rows, main_fields):
    lines = []

    lines.extend(markdown_table(summary_rows, summary_fields, display_labels(summary_fields)))
    lines.append("")
    lines.extend(markdown_table(main_rows, main_fields, display_labels(main_fields)))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(rows, fields, labels=None):
    labels = labels or fields
    header = "| " + " | ".join(markdown_escape(label) for label in labels) + " |"
    sep = "| " + " | ".join("---" for _ in fields) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(markdown_escape(row.get(field, "")) for field in fields)
            + " |"
        )
    return lines


def write_single_markdown(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(markdown_table(rows, fields, display_labels(fields))) + "\n",
        encoding="utf-8",
    )


def write_png(path, rows_dicts, fields):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("WARNING: matplotlib unavailable; skipping PNG: {}".format(exc), file=sys.stderr)
        return False

    labels = display_labels(fields, mode="png")
    labels = [wrap_text(label, 14) for label in labels]
    rows = []
    for row in rows_dicts:
        rendered_row = []
        for field in fields:
            value = row.get(field, "")
            if field == "model":
                value = wrap_text(value, 36, break_long_words=False)
            rendered_row.append(value)
        rows.append(rendered_row)

    fig_width = max(8.5, 1.8 + 1.0 * len(fields))
    fig_height = max(4.6, 0.62 * (len(rows) + 1))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    if len(fields) > 1:
        model_width = 0.32
        other_width = (1.0 - model_width) / float(len(fields) - 1)
        col_widths = [model_width] + [other_width] * (len(fields) - 1)
    else:
        col_widths = [1.0]

    table = ax.table(
        cellText=rows,
        colLabels=labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.0)
    table.scale(1, 1.0)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#ccd3dd")
        cell.set_linewidth(0.6)
        if row_idx == 0:
            cell.set_height(0.20)
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e6edf7")
        else:
            cell.set_height(0.11)
            if row_idx % 2 == 0:
                cell.set_facecolor("#f7f9fc")
            if col_idx == 0:
                cell.set_text_props(ha="left")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.015, right=0.985, top=0.97, bottom=0.03)
    fig.savefig(str(path), dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return True


def main(argv=None):
    root = repo_root()
    default_metrics = root / "configs" / "apertus" / "tasks_default_main_table.txt"
    default_output = root / "benchmark_tables"
    default_log_roots = [root / "logs", root.parent / "eval-logs"]

    parser = argparse.ArgumentParser(
        description="Aggregate lm-eval benchmark results into comparison tables."
    )
    parser.add_argument("--metrics-file", type=Path, default=default_metrics)
    parser.add_argument(
        "--logs-root",
        type=Path,
        action="append",
        dest="log_roots",
        help="Root containing results_*.json or .out logs. Can be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument(
        "--run-filter",
        action="append",
        help="Regex filter for run names. Can be repeated.",
    )
    parser.add_argument(
        "--model",
        action="append",
        help=(
            "Exact model/run name to include. Can be repeated or comma-separated; "
            "output rows keep this order."
        ),
    )
    parser.add_argument(
        "--scale",
        choices=["auto", "percent", "raw"],
        default="auto",
        help=(
            "Value scaling. auto multiplies values in [-1, 1] by 100 and leaves "
            "larger metrics, such as some F1 scores, unchanged."
        ),
    )
    parser.add_argument("--precision", type=int, default=2)
    parser.add_argument(
        "--include-splits",
        action="store_true",
        help="Use split result files even when a merged result exists.",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Skip the matplotlib PNG table.",
    )
    parser.add_argument(
        "--include-translations",
        action="store_true",
        help="Keep the translations section from the metrics file.",
    )
    parser.add_argument(
        "--include-empty-benchmarks",
        action="store_true",
        help="Keep benchmark columns with no values for the selected models.",
    )
    parser.add_argument(
        "--include-alpaca-eval",
        action="store_true",
        help="Keep alpaca_eval in the instruction-following domain.",
    )
    args = parser.parse_args(argv)

    log_roots = args.log_roots if args.log_roots else default_log_roots
    model_names = expand_model_args(args.model)
    specs = parse_metrics_file(args.metrics_file)
    if not args.include_translations:
        specs = [
            spec
            for spec in specs
            if spec["domain"].strip().lower() != "translations"
        ]
    if not args.include_alpaca_eval:
        specs = [
            spec
            for spec in specs
            if spec["benchmark"] not in DEFAULT_EXCLUDED_BENCHMARKS
        ]
    result_files = discover_result_files(log_roots)
    runs = build_runs(
        result_files,
        run_filters=args.run_filter,
        model_names=model_names,
        prefer_merged=not args.include_splits,
    )

    if not specs:
        raise SystemExit("No metric specs found in {}".format(args.metrics_file))
    if not runs:
        roots = ", ".join(str(root) for root in log_roots)
        raise SystemExit("No result JSON files found under {}".format(roots))
    if model_names:
        missing_models = [model for model in model_names if model not in runs]
        if missing_models:
            print(
                "WARNING: requested models not found: {}".format(
                    ", ".join(missing_models)
                ),
                file=sys.stderr,
            )

    summary_rows, detail_rows, missing = build_tables(
        specs,
        runs,
        scale_mode=args.scale,
        precision=args.precision,
    )

    run_names = list(runs.keys())
    if not args.include_empty_benchmarks:
        kept_detail_rows = [
            row for row in detail_rows if any(row.get(run) for run in run_names)
        ]
        kept_keys = {
            (row.get("benchmark"), row.get("metric")) for row in kept_detail_rows
        }
        detail_rows = kept_detail_rows
        missing = [
            item for item in missing if (item[1], item[2]) in kept_keys
        ]

    summary_fields, summary_model_rows, main_fields, main_rows = transpose_tables(
        summary_rows,
        detail_rows,
        run_names,
    )
    summary_csv = args.output_dir / "benchmark_summary.csv"
    detail_csv = args.output_dir / "benchmark_table.csv"
    markdown_path = args.output_dir / "benchmark_table.md"
    png_path = args.output_dir / "benchmark_table.png"
    split_dir = args.output_dir / "by_domain"

    write_csv(summary_csv, summary_model_rows, summary_fields)
    write_csv(detail_csv, main_rows, main_fields)
    write_markdown(markdown_path, summary_model_rows, summary_fields, main_rows, main_fields)

    split_outputs = []
    averages_base = split_dir / "domain_averages"
    write_csv(averages_base.with_suffix(".csv"), summary_model_rows, summary_fields)
    write_single_markdown(averages_base.with_suffix(".md"), summary_model_rows, summary_fields)
    split_outputs.extend([averages_base.with_suffix(".csv"), averages_base.with_suffix(".md")])

    domain_tables = build_domain_tables(detail_rows, run_names)
    for domain, (domain_fields, domain_rows) in domain_tables.items():
        base = split_dir / slugify(domain)
        write_csv(base.with_suffix(".csv"), domain_rows, domain_fields)
        write_single_markdown(base.with_suffix(".md"), domain_rows, domain_fields)
        split_outputs.extend([base.with_suffix(".csv"), base.with_suffix(".md")])

    png_written = False
    if not args.no_png:
        png_written = write_png(png_path, main_rows, main_fields)
        if write_png(averages_base.with_suffix(".png"), summary_model_rows, summary_fields):
            split_outputs.append(averages_base.with_suffix(".png"))
        for domain, (domain_fields, domain_rows) in domain_tables.items():
            base = split_dir / slugify(domain)
            if write_png(base.with_suffix(".png"), domain_rows, domain_fields):
                split_outputs.append(base.with_suffix(".png"))

    print("Loaded {} metric specs and {} runs.".format(len(specs), len(runs)))
    print("Wrote {}".format(summary_csv))
    print("Wrote {}".format(detail_csv))
    print("Wrote {}".format(markdown_path))
    if png_written:
        print("Wrote {}".format(png_path))
    print("Wrote {} split table files in {}".format(len(split_outputs), split_dir))
    print("Missing values: {}".format(len(missing)))
    if missing:
        preview = missing[:20]
        for run, benchmark, metric in preview:
            print("  {}: {}/{}".format(run, benchmark, metric))
        if len(missing) > len(preview):
            print("  ... {} more".format(len(missing) - len(preview)))


if __name__ == "__main__":
    main()
