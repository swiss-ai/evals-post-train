#!/usr/bin/env python3

import os
import argparse
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

BASE_MODEL_NAME = "baseline-Olmo-3-7B-SFT"
BASE_DPO_MODEL_NAME = "baseline-Olmo-3-7B-DPO"

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


def get_metric(summary, metric):
    # Dynamically average MGSM
    if "mgsm" in metric.lower():
        return compute_mgsm_avg(summary)

    # Exact match first
    if metric in summary:
        return summary[metric]
        
    # Robust fallback for GSM8K to avoid stderr and pick strict-match
    if "gsm8k" in metric.lower():
        candidates = []
        for k, v in summary.items():
            if "gsm8k" in k and "exact_match" in k and "stderr" not in k:
                candidates.append((k, v))
        
        if candidates:
            # Prefer strict-match if it exists
            for k, v in candidates:
                if "strict-match" in k:
                    return float(v)
            # Otherwise return the first valid exact_match found
            return float(candidates[0][1])

    return None


def normalize_score(val):
    if val is None:
        return None
    val = float(val)
    if val > 1.0 or val < 0.0:
        return val / 100.0
    return val


def parse_model_info(name):
    """
    Parses the model string to extract components for naming and strict sorting.
    Returns: (formatted_name, dataset_sort_order, norm_string, learning_rate_float)
    """
    if name == BASE_MODEL_NAME:
        return "base_model", 0, "", 0.0
    
    # 1. Extract learning rate (including the exponent minus sign)
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

    # 2. Extract ebs (batch size)
    ebs_match = re.search(r'ebs(\d+)', name)
    ebs_str = f"ebs{ebs_match.group(1)}" if ebs_match else "ebsUnknown"

    # 3. Extract length norm
    norm_match = re.search(r'norm(true|false)', name, re.IGNORECASE)
    norm_str = f"norm{norm_match.group(1).capitalize()}" if norm_match else "normUnknown"

    # 4. Extract dataset and determine sorting order
    name_lower = name.lower()
    if "maxmin" in name_lower:
        dtset = "maxmin"
        ds_order = 1
    elif "deltaqwen" in name_lower or "0.6b" in name_lower or "qwen" in name_lower:
        dtset = "Qwen3-32B_vs_0.6B"
        ds_order = 2
    else:
        dtset = name.split('_')[-1]
        ds_order = 3

    final_name = f"{dtset}_{norm_str}_{ebs_str}_{lr_str}"

    return final_name, ds_order, norm_str, lr_val


def simplify_metric_name(metric):
    base = metric.split('/')[0]
    for term in ['_direct', '_avg', '_comprehensive']:
        base = base.replace(term, '')
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--entity", required="apertus")
    parser.add_argument("--project", required="apertus-1.5-post-training-v0.0")
    parser.add_argument("--metrics-file", required=True)
    parser.add_argument("--output", default="/iopsstor/scratch/cscs/dmelikidze/evals-post-train/eval_table.tex")

    args = parser.parse_args()

    metrics = read_metrics_file(args.metrics_file)
    api = wandb.Api()
    models = list_models(args.base_model_path)

    print("Models found:", models)

    runs = {r.name: r for r in api.runs(f"{args.entity}/{args.project}")}
    summaries = {}

    for model in models:
        if model not in runs:
            print("Warning: run not found:", model)
            continue
        summaries[model] = dict(runs[model].summary)


    if BASE_MODEL_NAME not in summaries:
        raise RuntimeError("Base model run not found in W&B!")

    base_summary = summaries[BASE_MODEL_NAME]
    base_values = {
        m: normalize_score(get_metric(base_summary, m))
        for m in metrics
    }

    # Optionally add DPO baseline if present
    dpo_row = None
    if BASE_DPO_MODEL_NAME in summaries:
        dpo_summary = summaries[BASE_DPO_MODEL_NAME]
        dpo_row = {"Model": "dpo_baseline"}
        deltas = []
        for metric in metrics:
            val = normalize_score(get_metric(dpo_summary, metric))
            base_val = base_values[metric]
            short_metric = simplify_metric_name(metric)
            if val is None:
                dpo_row[short_metric] = "N/A"
            elif base_val is not None:
                delta = val - base_val
                dpo_row[short_metric] = f"{delta:+.4f}"
                deltas.append(delta)
            else:
                dpo_row[short_metric] = f"{val:.4f}"
        dpo_row["Avg_Imp"] = round(sum(deltas) / len(deltas), 4) if deltas else "N/A"

    raw_rows = []

    # Add base model row first
    short_model_name, ds_order, norm_str, lr_val = parse_model_info(BASE_MODEL_NAME)
    base_row = {
        "Model": short_model_name,
        "_ds_order": 0,
        "_norm_str": norm_str,
        "_lr_val": lr_val
    }
    for metric in metrics:
        val = normalize_score(get_metric(base_summary, metric))
        short_metric = simplify_metric_name(metric)
        base_row[short_metric] = f"{val:.4f}" if val is not None else "N/A"
    base_row["Avg_Imp"] = 0.0
    raw_rows.append(base_row)

    # Add DPO baseline row second if present
    if dpo_row is not None:
        # Use sorting keys to keep it second
        dpo_row["_ds_order"] = 0.5
        dpo_row["_norm_str"] = ""
        dpo_row["_lr_val"] = 0.0
        raw_rows.append(dpo_row)

    # Add all other models
    for model in models:
        if model == BASE_MODEL_NAME or model == BASE_DPO_MODEL_NAME:
            continue
        summary = summaries.get(model)
        # Unpack the parsing results including hidden sorting keys
        short_model_name, ds_order, norm_str, lr_val = parse_model_info(model)
        row = {
            "Model": short_model_name,
            "_ds_order": ds_order,
            "_norm_str": norm_str,
            "_lr_val": lr_val
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
            if base_val is not None:
                delta = val - base_val
                row[short_metric] = f"{delta:+.4f}"
                deltas.append(delta)
            else:
                row[short_metric] = f"{val:.4f}"
        if deltas:
            row["Avg_Imp"] = round(sum(deltas) / len(deltas), 4)
        else:
            row["Avg_Imp"] = "N/A"
        raw_rows.append(row)

    # Automatically sort the rows using the extracted numeric and string keys
    raw_rows.sort(key=lambda x: (x["_ds_order"], x["_norm_str"], x["_lr_val"]))

    # Clean up the hidden sorting keys before building the DataFrame
    for r in raw_rows:
        del r["_ds_order"]
        del r["_norm_str"]
        del r["_lr_val"]

    df = pd.DataFrame(raw_rows).set_index("Model")

    # 1. Output LaTeX
    latex = df.to_latex(escape=False)
    with open(args.output, "w") as f:
        f.write(latex)
    print("Saved LaTeX table to", args.output)

    # 2. Output CSV
    csv_output = args.output.replace('.tex', '.csv')
    df.to_csv(csv_output)
    print("Saved CSV table to", csv_output)

    # 3. Output PNG Layout (Significantly wider for long metric names)
    fig = plt.figure(figsize=(46, 10)) 
    
    ax_table = fig.add_subplot(111) 
    ax_table.axis('off')
    
    table_data = df.reset_index()
    
    # Calculate completely uniform proportional width across metrics to fit the massive width
    num_cols = len(table_data.columns)
    model_col_width = 0.14
    metric_col_width = (1.0 - model_col_width) / (num_cols - 1)
    col_widths = [model_col_width] + [metric_col_width] * (num_cols - 1)

    table = ax_table.table(
        cellText=table_data.values, 
        colLabels=table_data.columns, 
        loc='center', 
        cellLoc='center',
        colWidths=col_widths
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)

    # Add more vertical spacing between rows
    row_height = 0.08  # Increase this value for more spacing
    for (row, col), cell in table.get_celld().items():
        cell.set_height(row_height)

    plt.tight_layout()
    png_output = args.output.replace('.tex', '.png')
    plt.savefig(png_output, dpi=300, bbox_inches='tight')
    print("Saved PNG table to", png_output)


if __name__ == "__main__":
    main()