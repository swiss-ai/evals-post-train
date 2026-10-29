#!/usr/bin/env python3
"""
Script to scan all evaluation logs and update W&B with results from all models.
This script goes through all existing runs in the log directory and updates a W&B table.
"""

from pathlib import Path
from argparse import ArgumentParser
from typing import List

from .wandb_alignment_utils import (
    find_all_eval_dirs, 
    upload_multi_model_results,
    create_model_evaluation_from_results
)
from .data_structures import ModelEvaluation, Task

def load_main_metrics() -> List[str]:
    """Load main metrics from config file."""
    config_path = Path(__file__).parents[3] / "configs" / "alignment" / "tasks_english_main_table.txt"
    with open(config_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def scan_all_models(logs_root: Path) -> List[ModelEvaluation]:
    """Scan all models and create ModelEvaluation objects."""
    model_evaluations = []
    print(f"Scanning logs in: {logs_root}")

    # Process all model directories found in logs_root
    processed_count = 0
    for model_dir in sorted(logs_root.iterdir()):
        
        model_name = model_dir.name
        print(f"Processing {model_name}")
        eval_dirs = find_all_eval_dirs(logs_root, model_name)
        
        # Use all evaluation directories and merge their results (old to new)
        print(f"  Found {len(eval_dirs)} evaluation directories, merging results...")
        
        # Merge all evaluation directories into a single ModelEvaluation
        all_tasks_dict = {}
        
        for eval_dir in eval_dirs:
            print(f"    + Processing {eval_dir.name}")
            
            # Create evaluation for this directory
            temp_eval = create_model_evaluation_from_results(model_name, eval_dir)
            
            # Merge tasks into combined dictionary
            for task in temp_eval.tasks:
                if task.task_name not in all_tasks_dict:
                    all_tasks_dict[task.task_name] = {"metrics": [], "samples": []}
                
                # Add metrics (newer ones will be added last, effectively overwriting in final task)
                all_tasks_dict[task.task_name]["metrics"].extend(task.metrics)
                # Add samples
                all_tasks_dict[task.task_name]["samples"].extend(task.samples)
        
        # Create final merged tasks (remove duplicate metrics, keep latest)
        merged_tasks = []
        for task_name, data in all_tasks_dict.items():
            # Remove duplicate metrics (keep latest by name)
            unique_metrics = {}
            for metric in data["metrics"]:
                unique_metrics[metric.name] = metric  # Later metrics overwrite earlier ones
            
            merged_tasks.append(Task(
                task_name=task_name,
                metrics=list(unique_metrics.values()),
                samples=data["samples"]  # Keep all samples
            ))
        
        # Create final ModelEvaluation
        model_eval = ModelEvaluation(model_name=model_name, tasks=merged_tasks)
        model_evaluations.append(model_eval)
        processed_count += 1
        
        # Print summary
        print(f"  ✓ Total: {model_eval.total_metrics_count} metrics, {model_eval.total_samples_count} samples across {len(model_eval.tasks)} tasks")
    
    print(f"Successfully processed {processed_count} models")
    return model_evaluations


def main():
    parser = ArgumentParser(description="Scan all evaluation logs and update W&B with model results")
    parser.add_argument("--entity", type=str, required=True, help="W&B entity name")
    parser.add_argument("--project", type=str, required=True, help="W&B project name")
    parser.add_argument("--logs_root", type=Path, default="/iopsstor/scratch/cscs/ihakimi/eval-logs", 
                       help="Root directory containing all evaluation logs")
    parser.add_argument("--main_metrics", nargs='+', type=str, default=None,
                       help="List of main metrics for the summary table (defaults to config file)")
    parser.add_argument("--dry_run", action="store_true", help="Just scan and print results without uploading to W&B")
    
    args = parser.parse_args()
    
    # Load main metrics from config if not provided
    if args.main_metrics is None:
        args.main_metrics = load_main_metrics()
    
    # Scan all models
    model_evaluations = scan_all_models(args.logs_root)
    
    # Print summary
    print("\n=== SUMMARY ===")
    print(f"Found results for {len(model_evaluations)} models:")
    for model_eval in model_evaluations:
        available_main_metrics = sum(1 for metric in args.main_metrics if metric in model_eval.get_flattened_metrics())
        print(f"  {model_eval.model_name}: {model_eval.total_metrics_count} total metrics, {available_main_metrics}/{len(args.main_metrics)} main metrics, {model_eval.total_samples_count} samples")
    
    if args.dry_run:
        print("\nDry run completed. No data uploaded to W&B.")
        return
    
    # Upload to W&B using structured approach
    upload_multi_model_results(args.entity, args.project, model_evaluations, args.main_metrics)


if __name__ == "__main__":
    main()
