#!/bin/bash

# bash launcher_graceful.sh --base_dir $SCRATCH/aper_mods6 --config_file ./configs/apertus/tasks_posttrain.txt --table_metrics ./configs/apertus/tasks_posttrain_main_table.txt

# Exit on error, undefined variable, or pipeline failure
set -euo pipefail

# --- Defaults ---
LOGS_ROOT="${SCRATCH:-/tmp}/eval_logs_start/apertus/apertus-1.5-post-training-v0.0/"
WANDB_ENTITY="apertus"
WANDB_PROJECT="apertus-1.5-post-training-v0.0"
DEBUG=0
GROUP_SIZE=1
BASE_DIR=""
CONFIG_FILE=""
TABLE_METRICS=""

# --- Argument Parsing ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --base_dir) BASE_DIR="$2"; shift 2 ;;
        --base_dir=*) BASE_DIR="${1#*=}"; shift 1 ;;
        --config_file) CONFIG_FILE="$2"; shift 2 ;;
        --config_file=*) CONFIG_FILE="${1#*=}"; shift 1 ;;
        --table_metrics) TABLE_METRICS="$2"; shift 2 ;;
        --table_metrics=*) TABLE_METRICS="${1#*=}"; shift 1 ;;
        --logs_root) LOGS_ROOT="$2"; shift 2 ;;
        --logs_root=*) LOGS_ROOT="${1#*=}"; shift 1 ;;
        --wandb_entity) WANDB_ENTITY="$2"; shift 2 ;;
        --wandb_entity=*) WANDB_ENTITY="${1#*=}"; shift 1 ;;
        --wandb_project) WANDB_PROJECT="$2"; shift 2 ;;
        --wandb_project=*) WANDB_PROJECT="${1#*=}"; shift 1 ;;
        --group_size) GROUP_SIZE="$2"; shift 2 ;;
        --group_size=*) GROUP_SIZE="${1#*=}"; shift 1 ;;
        --debug) DEBUG=1; shift 1 ;;
        *)
            echo "Error: Unknown argument '$1'"
            exit 1
            ;;
    esac
done

if [[ -z "$BASE_DIR" || -z "$CONFIG_FILE" ]]; then
    echo "Error: --base_dir and --config_file are required."
    exit 1
fi

mapfile -t MODEL_DIRS < <(find -L "$BASE_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
if [[ ${#MODEL_DIRS[@]} -eq 0 ]]; then exit 0; fi

if [[ $DEBUG -eq 1 ]]; then
    echo -e "\n[DEBUG MODE] Restricting execution to 1 model only."
    MODEL_DIRS=("${MODEL_DIRS[0]}")
fi

for model_path in "${MODEL_DIRS[@]}"; do
    model_name=$(basename "$model_path")
    echo -e "\n============================================================"
    echo "Processing Model: $model_name"
    echo "============================================================"

    CMD=(
        bash "scripts/launch_evaluations_gracefuly.sh"
        "--task_file" "$CONFIG_FILE"
        "--model" "$model_path"
        "--eval_prefix" "$LOGS_ROOT"
        "--wandb_entity" "$WANDB_ENTITY"
        "--wandb_project" "$WANDB_PROJECT"
        "--group_size" "$GROUP_SIZE"
    )
    
    if [[ -n "$TABLE_METRICS" ]]; then
        CMD+=("--table_metrics" "$TABLE_METRICS")
    fi
    
    if [[ $DEBUG -eq 1 ]]; then CMD+=("--debug"); fi
    
    if ! "${CMD[@]}"; then
        echo "Error: Failed to process $model_name."
        exit 1 
    fi
    #if [[ $model_name == "/iopsstor/scratch/cscs/dmelikidze/aper_mods2/baseline-Olmo-3-7B-SFT2" ]]; then
    echo "Reached debug model, exiting after first iteration."
    # exit 0
    #fi
done