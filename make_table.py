#!/usr/bin/env python3

import os
import argparse
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

BASE_MODEL_NAME = "baseline-apertus-1-sft"
BASE_DPO_MODEL_NAME = "baseline-apertus-1-instruct"

def read_metrics_file(path):
    metrics = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                metrics.append(line)
    return metrics


def list_models(base_dir):
    return sorted(
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    )


def compute_mgsm_avg(summary):
    vals = []
    # Strictly target keys ending in 'exact_match' to filter out stderr and ordered-extract
    for k, v in summary.items():
        if "mgsm" in k and k.endswith("exact_match"):
            try:
                vals.append(float(v))
            except (ValueError, TypeError):
                pass

    if not vals:
        return None

    return sum(vals) / len(vals)


def is_mgsm_aggregate(task, metric_name):
    """True when a request means "the MGSM average across languages".

    MGSM is many per-language subtasks, so a request for its accuracy is answered by
    averaging them. Keying this on the task name alone would answer *every* mgsm request
    that way -- including a per-language score or a thinking length -- so the requested
    metric has to match what compute_mgsm_avg() actually averages.
    """
    if metric_name != "exact_match":
        return False
    task = task.lower()
    return "mgsm" in task and "mgsm_en" not in task


def get_metric(summary, metric):
    # Split metric into task and metric_name parts for fuzzy matching
    # e.g. "agieval/acc" -> task="agieval", metric_name="acc"
    # e.g. "mmlu_flan_cot_zeroshot/exact_match,ordered-extract" -> task="mmlu_flan_cot_zeroshot", metric_name="exact_match", filter="ordered-extract"
    parts = metric.split("/", 1)
    task = parts[0]
    metric_name = parts[1] if len(parts) > 1 else None

    # Handle comma-separated filter (e.g. "exact_match,ordered-extract")
    metric_filter = None
    if metric_name and "," in metric_name:
        metric_name, metric_filter = metric_name.split(",", 1)

    # Dynamically average MGSM
    if is_mgsm_aggregate(task, metric_name):
        return compute_mgsm_avg(summary)

    # Exact match first
    if metric in summary:
        return summary[metric]

    # Search for matching keys in summary
    candidates = []
    for k, v in summary.items():
        if "stderr" in k:
            continue
        k_lower = k.lower()
        task_lower = task.lower()
        # Key must contain the task name
        if task_lower not in k_lower:
            continue
        # If we have a metric_name, key must contain it
        if metric_name and metric_name.lower() not in k_lower:
            continue
        # If we have a filter, key must contain it
        if metric_filter and metric_filter.lower() not in k_lower:
            continue
        candidates.append((k, v))

    if not candidates:
        return None

    # Prefer strict-match for gsm8k
    if "gsm8k" in task.lower():
        for k, v in candidates:
            if "strict-match" in k:
                return float(v)

    # If multiple candidates, prefer the shortest key (most specific match)
    candidates.sort(key=lambda x: len(x[0]))
    return candidates[0][1]


def normalize_score(val):
    if val is None:
        return None
    val = float(val)
    if val > 1.0 or val < 0.0:
        return val / 100.0
    return val


def parse_model_info(name, method_filter=None):
    """
    Parses the model string to extract components for naming and strict sorting.
    Returns: (formatted_name, dataset_sort_order, norm_string, learning_rate_float)
    """
    if name == BASE_MODEL_NAME:
        return "Official Apertus-1.0-8B-SFT", 0, "", 0.0
    
    # 1. Extract beta
    beta_match = re.search(r'beta([0-9.]+)', name)
    beta_str = f"b{beta_match.group(1)}" if beta_match else "bUnk"

    # 2. Extract learning rate (including the exponent minus sign)
    lr_match = re.search(r'lr([0-9eE.\-]+)', name)
    if lr_match:
        lr_raw = lr_match.group(1).rstrip('-')
        lr_str = f"lr{lr_raw}"
        try:
            lr_val = float(lr_raw)
        except ValueError:
            lr_val = float('inf')
    else:
        lr_str = "unknown_lr"
        lr_val = float('inf')

    # 3. Extract ebs (batch size)
    ebs_match = re.search(r'ebs(\d+)', name)
    ebs_str = f"ebs{ebs_match.group(1)}" if ebs_match else "ebs128"

    # 4. Extract dataset and determine sorting order
    name_lower = name.lower()
    if "maxmin" in name_lower:
        dtset = "maxmin"
        ds_order = 1
    elif "deltaqwen" in name_lower or "0.6b" in name_lower or "qwen" in name_lower:
        dtset = "Qwen3-32B_vs_0.6B"
        ds_order = 2
    elif "qrpo" in name_lower:
        dtset = "maxmin"
        ds_order = 1
    else:
        dtset = name.split('_')[-1]
        ds_order = 4

    # 5. Extract length normalization
    ln_match = re.search(r'ln[_-]?(true|false)', name_lower)
    if ln_match:
        norm_str = "lenNorm" if ln_match.group(1) == "true" else ""
    elif "lengthnorm" in name_lower or "length_norm" in name_lower:
        norm_str = "lenNorm"
    else:
        norm_str = ""

    # 6. Extract training method (dpo/qrpo) — include in name when no filter is active
    method_str = ""
    if not method_filter:
        if "qrpo" in name_lower:
            method_str = "qrpo"
        elif "dpo" in name_lower:
            method_str = "dpo"

    parts = [dtset]
    if method_str:
        parts.append(method_str)
    parts.extend([beta_str, ebs_str, lr_str])
    if norm_str:
        parts.append(norm_str)
    final_name = "_".join(parts)

    return final_name, ds_order, norm_str, lr_val


def simplify_metric_name(metric):
    base = metric.split('/')[0]
    for term in ['_direct', '_avg', '_comprehensive']:
        base = base.replace(term, '')
    return base


def main():

    # get username from environment variable 
    username = os.getenv("USER")

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", required=True, nargs="+",
                        help="One or more directories containing model subdirectories")
    parser.add_argument("--filter", default=None,
                        help="Only include model directories whose name contains this substring")
    parser.add_argument("--entity", default="apertus")
    parser.add_argument("--project", default="apertus-1.5-post-training-v0.0")
    parser.add_argument("--metrics-file", default=["./configs/apertus/tasks_posttrain_main_table.txt"],
                        nargs="+", help="One or more metrics files (all are merged)")
    parser.add_argument("--output", default=f"/iopsstor/scratch/cscs/{username}/eval_tables",
                        help="Output directory (created if it doesn't exist)")
    parser.add_argument("--title", default="Eval Table",
                        help="Title text for the PNG table")
    parser.add_argument("--top-n", type=int, default=None,
                        help="Only show the top N configurations ranked by Avg_Imp (baselines always included)")
    parser.add_argument("--debug", action="store_true",
                        help="Print W&B summary keys for the base model and metric matching details")
    parser.add_argument("--exclude", nargs="+", default=[],
                        help="Benchmarks to exclude from the table and Avg_Imp (matched as substring, e.g. --exclude mbpp harmbench)")
    parser.add_argument("--include", nargs="+", default=[],
                        help="Only include these benchmarks (matched as substring, e.g. --include mmlu gsm8k). Overrides --exclude.")
    parser.add_argument("--label", default=None,
                        help="Optional label to use as the Model name for all runs from --base-model-path")
    parser.add_argument("--raw-names", action="store_true",
                        help="Use directory names directly as row names without parsing for lr/beta/ebs")
    parser.add_argument("--model-order", nargs="+", default=None,
                        help="Explicit order of model dir names (e.g. --model-order sft dpo)")
    parser.add_argument("--rename", nargs="+", default=[],
                        help="Rename models: old=new (e.g. --rename 'apertus1b-sft-stage2-dpo-new=apertus1b-sft-stage2-dpo')")
    parser.add_argument("--extra-baseline", default=None,
                        help="W&B run name to use as an additional baseline row (placed after the DPO baseline)")
    parser.add_argument("--extra-baseline-name", default=None,
                        help="Display name for the extra baseline (defaults to the run name)")
    parser.add_argument("--extra-run", default=None, nargs="+",
                        help="One or more W&B run names to append at the end of the table")
    parser.add_argument("--extra-run-name", default=None, nargs="+",
                        help="Display names for extra runs (defaults to the run names)")
    parser.add_argument("--complete-only", action="store_true",
                        help="Only keep rows with no N/A values")
    parser.add_argument("--baseline", default=None,
                        help="Override the base model W&B run name (default: baseline-apertus-1-sft)")
    parser.add_argument("--no-dpo-baseline", action="store_true",
                        help="Skip the DPO baseline row")
    parser.add_argument("--absolute", action="store_true",
                        help="Show absolute scores instead of delta improvements from the baseline")
    parser.add_argument("--wordcount", action="store_true",
                        help="Append an avg_word_count column (alpaca_eval/avg_word_count) with plain values, no deltas")

    args = parser.parse_args()

    rename_map = {}
    for pair in args.rename:
        old, new = pair.split("=", 1)
        rename_map[old] = new

    base_model_name = args.baseline if args.baseline else BASE_MODEL_NAME
    skip_dpo_baseline = args.no_dpo_baseline

    metrics = []
    for mf in args.metrics_file:
        metrics.extend(read_metrics_file(mf))
    # Deduplicate while preserving order
    seen = set()
    metrics = [m for m in metrics if not (m in seen or seen.add(m))]
    if args.include:
        before = len(metrics)
        metrics = [m for m in metrics if any(inc.lower() in m.lower() for inc in args.include)]
        print(f"Filtered to {len(metrics)} benchmarks (from {before}) matching --include {args.include}")
    elif args.exclude:
        before = len(metrics)
        metrics = [m for m in metrics if not any(ex.lower() in m.lower() for ex in args.exclude)]
        print(f"Excluded {before - len(metrics)} benchmarks, {len(metrics)} remaining")
    api = wandb.Api()
    models = []
    for bp in args.base_model_path:
        models.extend(list_models(bp))
    models = sorted(set(models))
    if args.filter:
        models = [m for m in models if args.filter in m]

    print("Models found:", models)

    project_path = f"{args.entity}/{args.project}"
    summaries = {}

    # Collect all run names we need to fetch
    needed_runs = set([base_model_name]) | set(models)
    if not skip_dpo_baseline:
        needed_runs.add(BASE_DPO_MODEL_NAME)
    if args.extra_baseline:
        needed_runs.add(args.extra_baseline)
    if args.extra_run:
        needed_runs.update(args.extra_run)

    # Fetch only the runs we need, one at a time
    runs = {}
    for name in needed_runs:
        try:
            matched = list(api.runs(project_path, filters={"display_name": name}, order="-created_at", per_page=1))
            if matched:
                runs[name] = matched[0]
        except Exception as e:
            print(f"Warning: failed to fetch run '{name}': {e}")

    # Always fetch baseline runs directly from W&B (they may not be dirs in base-model-path)
    baselines_to_fetch = [base_model_name]
    if not skip_dpo_baseline:
        baselines_to_fetch.append(BASE_DPO_MODEL_NAME)
    for baseline in baselines_to_fetch:
        if baseline in runs:
            summaries[baseline] = dict(runs[baseline].summary)
        else:
            print(f"Warning: baseline '{baseline}' not found in W&B project {project_path}")

    for model in models:
        if model not in runs:
            print("Warning: run not found:", model)
            continue
        summaries[model] = dict(runs[model].summary)


    if base_model_name not in summaries:
        raise RuntimeError(f"Base model run '{base_model_name}' not found in W&B project {args.entity}/{args.project}!")

    base_summary = summaries[base_model_name]

    if args.debug:
        print("\n=== W&B summary keys for base model ===")
        for k in sorted(base_summary.keys()):
            print(f"  {k}: {base_summary[k]}")
        print()
        for m in metrics:
            val = get_metric(base_summary, m)
            print(f"  Metric '{m}' -> {val}")
        print()

    base_values = {
        m: normalize_score(get_metric(base_summary, m))
        for m in metrics
    }

    # Optionally add DPO baseline if present
    dpo_row = None
    if not skip_dpo_baseline and BASE_DPO_MODEL_NAME in summaries:
        dpo_summary = summaries[BASE_DPO_MODEL_NAME]
        dpo_row = {"Model": "Official Apertus-1.0-8B-Instruct"}
        deltas = []
        for metric in metrics:
            val = normalize_score(get_metric(dpo_summary, metric))
            base_val = base_values[metric]
            short_metric = simplify_metric_name(metric)
            if val is None:
                dpo_row[short_metric] = "N/A"
            elif args.absolute:
                dpo_row[short_metric] = f"{val:.4f}"
                if base_val is not None:
                    deltas.append(val - base_val)
            elif base_val is not None:
                delta = val - base_val
                dpo_row[short_metric] = f"{delta:+.4f}"
                deltas.append(delta)
            else:
                dpo_row[short_metric] = f"{val:.4f}"
        if args.absolute:
            abs_vals = [normalize_score(get_metric(dpo_summary, m)) for m in metrics]
            abs_vals = [v for v in abs_vals if v is not None]
            dpo_row["Avg_Score"] = round(sum(abs_vals) / len(abs_vals), 4) if abs_vals else "N/A"
        else:
            dpo_row["Avg_Imp"] = round(sum(deltas) / len(deltas), 4) if deltas else "N/A"

    raw_rows = []

    # Add base model row first
    if args.raw_names or args.baseline:
        short_model_name = base_model_name
        ds_order, norm_str, lr_val = 0, "", 0.0
    else:
        short_model_name, ds_order, norm_str, lr_val = parse_model_info(base_model_name, method_filter=args.filter)
    if base_model_name in rename_map:
        short_model_name = rename_map[base_model_name]
    base_row = {
        "Model": short_model_name,
        "_ds_order": 0,
        "_norm_str": norm_str,
        "_lr_val": lr_val,
        "_orig_model": base_model_name,
    }
    for metric in metrics:
        val = normalize_score(get_metric(base_summary, metric))
        short_metric = simplify_metric_name(metric)
        base_row[short_metric] = f"{val:.4f}" if val is not None else "N/A"
    avg_col = "Avg_Score" if args.absolute else "Avg_Imp"
    if args.absolute:
        base_vals_list = [normalize_score(get_metric(base_summary, m)) for m in metrics]
        base_vals_list = [v for v in base_vals_list if v is not None]
        base_row[avg_col] = round(sum(base_vals_list) / len(base_vals_list), 4) if base_vals_list else "N/A"
    else:
        base_row[avg_col] = 0.0
    raw_rows.append(base_row)

    # Add DPO baseline row second if present
    if dpo_row is not None:
        # Use sorting keys to keep it second
        dpo_row["_ds_order"] = 0.5
        dpo_row["_norm_str"] = ""
        dpo_row["_lr_val"] = 0.0
        dpo_row["_orig_model"] = BASE_DPO_MODEL_NAME
        raw_rows.append(dpo_row)

    # Add extra baseline row if requested (placed after DPO baseline)
    if args.extra_baseline:
        if args.extra_baseline in runs:
            eb_summary = dict(runs[args.extra_baseline].summary)
            eb_display = args.extra_baseline_name or args.extra_baseline
            eb_row = {"Model": eb_display, "_ds_order": 0.75, "_norm_str": "", "_lr_val": 0.0, "_orig_model": args.extra_baseline}
            deltas = []
            for metric in metrics:
                val = normalize_score(get_metric(eb_summary, metric))
                base_val = base_values[metric]
                short_metric = simplify_metric_name(metric)
                if val is None:
                    eb_row[short_metric] = "N/A"
                elif args.absolute:
                    eb_row[short_metric] = f"{val:.4f}"
                    if base_val is not None:
                        deltas.append(val - base_val)
                elif base_val is not None:
                    delta = val - base_val
                    eb_row[short_metric] = f"{delta:+.4f}"
                    deltas.append(delta)
                else:
                    eb_row[short_metric] = f"{val:.4f}"
            if args.absolute:
                abs_vals = [normalize_score(get_metric(eb_summary, m)) for m in metrics]
                abs_vals = [v for v in abs_vals if v is not None]
                eb_row["Avg_Score"] = round(sum(abs_vals) / len(abs_vals), 4) if abs_vals else "N/A"
            else:
                eb_row["Avg_Imp"] = round(sum(deltas) / len(deltas), 4) if deltas else "N/A"
            raw_rows.append(eb_row)
        else:
            print(f"Warning: extra baseline '{args.extra_baseline}' not found in W&B project")

    # Add all other models
    for model in models:
        if model == base_model_name or model == BASE_DPO_MODEL_NAME:
            continue
        summary = summaries.get(model)
        # Unpack the parsing results including hidden sorting keys
        if args.raw_names:
            short_model_name = model
            ds_order, norm_str, lr_val = 1, "", 0.0
        else:
            short_model_name, ds_order, norm_str, lr_val = parse_model_info(model, method_filter=args.filter)
        if args.label:
            short_model_name = args.label
        if model in rename_map:
            short_model_name = rename_map[model]
        row = {
            "Model": short_model_name,
            "_ds_order": ds_order,
            "_norm_str": norm_str,
            "_lr_val": lr_val,
            "_orig_model": model,
        }
        deltas = []
        for metric in metrics:
            val = None
            short_metric = simplify_metric_name(metric)
            if summary is not None:
                val = normalize_score(get_metric(summary, metric))
            if val is None:
                row[short_metric] = "N/A"
                continue
            base_val = base_values[metric]
            if args.absolute:
                row[short_metric] = f"{val:.4f}"
                if base_val is not None:
                    deltas.append(val - base_val)
            elif base_val is not None:
                delta = val - base_val
                row[short_metric] = f"{delta:+.4f}"
                deltas.append(delta)
            else:
                row[short_metric] = f"{val:.4f}"
        if args.absolute:
            abs_vals = [normalize_score(get_metric(summary, m)) for m in metrics if summary is not None] if summary else []
            abs_vals = [v for v in abs_vals if v is not None]
            row[avg_col] = round(sum(abs_vals) / len(abs_vals), 4) if abs_vals else "N/A"
        elif deltas:
            row[avg_col] = round(sum(deltas) / len(deltas), 4)
        else:
            row[avg_col] = "N/A"
        raw_rows.append(row)

    # Filter to top N by Avg_Imp if requested (baselines always kept)
    if args.top_n is not None:
        baseline_rows = [r for r in raw_rows if r["_ds_order"] < 1]
        non_baseline_rows = [r for r in raw_rows if r["_ds_order"] >= 1]
        # Sort non-baselines by Avg_Imp descending (treat "N/A" as -inf)
        non_baseline_rows.sort(
            key=lambda x: x[avg_col] if isinstance(x[avg_col], (int, float)) else float('-inf'),
            reverse=True
        )
        non_baseline_rows = non_baseline_rows[:args.top_n]
        raw_rows = baseline_rows + non_baseline_rows

    # Sort: baselines first (by _ds_order), then non-baselines by explicit order or Avg_Imp
    if args.model_order:
        order_map = {name: i for i, name in enumerate(args.model_order)}
        def sort_key(x):
            is_baseline = x["_ds_order"] < 1
            order_idx = order_map.get(x["_orig_model"], len(order_map))
            return (0 if is_baseline else 1, x["_ds_order"] if is_baseline else order_idx)
    else:
        def sort_key(x):
            is_baseline = x["_ds_order"] < 1
            avg = x[avg_col] if isinstance(x[avg_col], (int, float)) else float('-inf')
            return (0 if is_baseline else 1, x["_ds_order"] if is_baseline else -avg)
    raw_rows.sort(key=sort_key)

    # Append extra runs at the end if requested
    if args.extra_run:
        extra_names = args.extra_run_name or []
        for i, er in enumerate(args.extra_run):
            if er in runs:
                extra_summary = dict(runs[er].summary)
                summaries[er] = extra_summary
                if args.debug:
                    print(f"\n=== W&B summary keys for extra run '{er}' ===")
                    for k in sorted(extra_summary.keys()):
                        print(f"  {k}: {extra_summary[k]}")
                    for m in metrics:
                        val = get_metric(extra_summary, m)
                        print(f"  Metric '{m}' -> {val}")
                    print()
                display_name = extra_names[i] if i < len(extra_names) else rename_map.get(er, er)
                extra_row = {"Model": display_name, "_orig_model": er}
                deltas = []
                for metric in metrics:
                    val = normalize_score(get_metric(extra_summary, metric))
                    base_val = base_values[metric]
                    short_metric = simplify_metric_name(metric)
                    if val is None:
                        extra_row[short_metric] = "N/A"
                    elif args.absolute:
                        extra_row[short_metric] = f"{val:.4f}"
                        if base_val is not None:
                            deltas.append(val - base_val)
                    elif base_val is not None:
                        delta = val - base_val
                        extra_row[short_metric] = f"{delta:+.4f}"
                        deltas.append(delta)
                    else:
                        extra_row[short_metric] = f"{val:.4f}"
                if args.absolute:
                    abs_vals = [normalize_score(get_metric(extra_summary, m)) for m in metrics]
                    abs_vals = [v for v in abs_vals if v is not None]
                    extra_row[avg_col] = round(sum(abs_vals) / len(abs_vals), 4) if abs_vals else "N/A"
                else:
                    extra_row[avg_col] = round(sum(deltas) / len(deltas), 4) if deltas else "N/A"
                raw_rows.append(extra_row)
            else:
                print(f"Warning: extra run '{er}' not found in W&B project")

    # Append plain wordcount column if requested
    if args.wordcount:
        wc_metric = "alpaca_eval/avg_word_count"
        for r in raw_rows:
            orig = r.get("_orig_model")
            if orig and orig in summaries:
                wc_val = get_metric(summaries[orig], wc_metric)
                if wc_val is not None:
                    r["avg_word_count"] = f"{float(wc_val):.1f}"
                else:
                    r["avg_word_count"] = "N/A"
            elif orig and orig in runs:
                s = dict(runs[orig].summary)
                wc_val = get_metric(s, wc_metric)
                if wc_val is not None:
                    r["avg_word_count"] = f"{float(wc_val):.1f}"
                else:
                    r["avg_word_count"] = "N/A"
            else:
                r["avg_word_count"] = "N/A"

    # Clean up the hidden sorting keys before building the DataFrame
    for r in raw_rows:
        r.pop("_ds_order", None)
        r.pop("_norm_str", None)
        r.pop("_lr_val", None)
        r.pop("_orig_model", None)

    if args.complete_only:
        raw_rows = [r for r in raw_rows if all(
            v != "N/A" and "nan" not in str(v).lower() for v in r.values()
        )]

    df = pd.DataFrame(raw_rows).set_index("Model")

    os.makedirs(args.output, exist_ok=True)


    n_rows = len(df)
    n_cols = len(df.columns)
    fig_width = max(30, 2 + n_cols * 2.5)
    fig_height = max(4, 1.5 + n_rows * 0.6)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    fig.suptitle(args.title, fontsize=16, fontweight='bold', y=0.98)

    table_data = df.reset_index()
    num_cols = len(table_data.columns)
    model_col_width = 0.18
    metric_col_width = (1.0 - model_col_width) / (num_cols - 1)
    col_widths = [model_col_width] + [metric_col_width] * (num_cols - 1)

    tbl = ax.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        loc='center',
        cellLoc='center',
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.2)

    # Color header row
    for col_idx in range(num_cols):
        tbl[0, col_idx].set_facecolor('#4472C4')
        tbl[0, col_idx].set_text_props(color='white', fontweight='bold')

    # Alternating row colors for readability
    row_colors = ['#FFFFFF', '#DCE6F1']
    for row_idx in range(1, n_rows + 1):
        bg = row_colors[(row_idx - 1) % 2]
        for col_idx in range(num_cols):
            tbl[row_idx, col_idx].set_facecolor(bg)

    # Color-code Avg column values (overrides row stripe)
    avg_col_idx = list(table_data.columns).index(avg_col)
    for row_idx in range(1, n_rows + 1):
        cell = tbl[row_idx, avg_col_idx]
        try:
            val = float(table_data.iloc[row_idx - 1][avg_col])
            if not args.absolute:
                if val > 0:
                    cell.set_facecolor('#C6EFCE')  # green
                elif val < 0:
                    cell.set_facecolor('#FFC7CE')  # red
        except (ValueError, TypeError):
            pass

    for (row, col), cell in tbl.get_celld().items():
        cell.set_height(0.07)

    plt.tight_layout()
    png_path = os.path.join(args.output, "eval_table.png")
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Saved PNG table to", png_path)

    csv_path = os.path.join(args.output, "eval_table.csv")
    df.to_csv(csv_path)
    print("Saved CSV table to", csv_path)


if __name__ == "__main__":
    main()