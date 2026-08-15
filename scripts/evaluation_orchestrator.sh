#!/bin/bash

# Slurm orchestration for resumable task chunks. The file is sourced by
# runners/hf_base_runner.sh and executed by evaluation_controller.sbatch.

_eval_repo_root() {
    cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

_eval_container_for_backend() {
    case "${LM_EVAL_BACKEND:-vllm}" in
        vllm|megatron_lm) echo "./containers/env_vllm.toml" ;;
        sglang) echo "./containers/env_sglang.toml" ;;
        hf|openai) echo "./containers/env.toml" ;;
        *) echo "Unsupported LM_EVAL_BACKEND: ${LM_EVAL_BACKEND:-}" >&2; return 1 ;;
    esac
}

_eval_submit_aggregator() {
    local model="$1" name="$2" eval_dirs_file="$3" incomplete_file="${4:-}"
    local -a command=(sbatch --parsable --export="ALL,EVAL_DIRS_FILE=$eval_dirs_file")
    if [[ -n "$incomplete_file" ]]; then
        command[2]="--export=ALL,EVAL_DIRS_FILE=$eval_dirs_file,EVAL_INCOMPLETE_TASKS_FILE=$incomplete_file"
    fi
    command+=(scripts/aggregate_chunks.sbatch "$model" "$name")

    if [[ "${EVAL_DRY_RUN:-false}" == "true" ]]; then
        echo "[DRY RUN] ${command[*]}" >&2
        EVAL_SUBMITTED_JOB_ID="dry-aggregate"
    else
        EVAL_SUBMITTED_JOB_ID=$("${command[@]}")
        echo "Aggregation job submitted: $EVAL_SUBMITTED_JOB_ID"
    fi
}

_eval_schedule_judge_cleanup() {
    local dependency_job="${1:-}"
    [[ -n "${JUDGE_JOB_IDS:-}" && "${KEEP_JUDGE:-false}" != "true" ]] || return 0

    local dependency=""
    [[ -n "$dependency_job" ]] && dependency="--dependency=afterany:$dependency_job"
    local -a command=(sbatch --parsable --account="${SBATCH_ACCOUNT:-infra01}" --partition=normal
        --job-name judge-cleanup --time=00:05:00)
    [[ -n "$dependency" ]] && command+=("$dependency")
    command+=(--wrap="scancel $JUDGE_JOB_IDS")
    if [[ "${EVAL_DRY_RUN:-false}" == "true" ]]; then
        echo "[DRY RUN] ${command[*]}" >&2
    else
        local cleanup_job
        cleanup_job=$("${command[@]}")
        echo "Judge cleanup job $cleanup_job will cancel judge(s) [$JUDGE_JOB_IDS]"
    fi
}

_eval_scan() {
    local state_dir="$1" suffix="$2"
    local -a harness_args=(--harness-dir "$EVAL_HARNESS_DIR")
    local pattern
    shift 2
    if [[ -n "${EVAL_LEGACY_HARNESS_DIR:-}" && "$EVAL_LEGACY_HARNESS_DIR" != "$EVAL_HARNESS_DIR" ]]; then
        harness_args+=(--harness-dir "$EVAL_LEGACY_HARNESS_DIR")
    fi
    if [[ -s "$state_dir/force_patterns.txt" ]]; then
        while IFS= read -r pattern; do
            [[ -n "$pattern" ]] && harness_args+=(--force-pattern "$pattern")
        done < "$state_dir/force_patterns.txt"
        harness_args+=(--force-after "$(< "$state_dir/force_after.txt")")
    fi
    python3 -m scripts.eval_state scan \
        --tasks-file "$state_dir/expected_tasks.txt" \
        "${harness_args[@]}" \
        --missing-output "$state_dir/missing_${suffix}.txt" \
        --eval-dirs-output "$state_dir/eval_dirs_${suffix}.txt" \
        --completed-output "$state_dir/completed_${suffix}.tsv" "$@"
}

_eval_submit_wave() {
    local model="$1" name="$2" state_dir="$3" attempt="$4" chunk_size="$5"
    local missing_file="$state_dir/missing_${attempt}.txt"
    local chunks_file="$state_dir/chunks_${attempt}.txt"
    local chunk_count array_spec array_job controller_job safe_name controller_name

    python3 -m scripts.eval_state chunk --tasks-file "$missing_file" \
        --chunk-size "$chunk_size" --output "$chunks_file"
    chunk_count=$(wc -l < "$chunks_file" | tr -d ' ')
    (( chunk_count > 0 )) || { echo "No missing tasks to submit"; return 1; }

    array_spec="0-$((chunk_count - 1))"
    if [[ -n "${EVAL_MAX_PARALLEL:-}" ]] && (( EVAL_MAX_PARALLEL < chunk_count )); then
        array_spec="${array_spec}%${EVAL_MAX_PARALLEL}"
    fi
    safe_name=${name//[^a-zA-Z0-9_.-]/-}
    # Keep controller jobs and their %x-based log filenames attributable to the
    # evaluation run. The attempt suffix distinguishes retry-wave controllers.
    controller_name="eval-ctrl-${safe_name:0:80}-a${attempt}"

    local -a eval_command=(sbatch --parsable --array="$array_spec"
        --job-name="eval-${safe_name}"
        --export="ALL,EVAL_CHUNKS_FILE=$chunks_file,EVAL_ENV_MANIFEST=$EVAL_ENV_MANIFEST,EVAL_CHUNKED=true,WANDB_MODE=disabled"
        "$SBATCH_SCRIPT" "$model" "$name")
    if (( attempt == 0 )) && [[ -n "${EVAL_PREP_JOB_ID:-}" ]]; then
        eval_command=(sbatch --parsable --array="$array_spec"
            --job-name="eval-${safe_name}"
            --dependency="afterok:${EVAL_PREP_JOB_ID}"
            --export="ALL,EVAL_CHUNKS_FILE=$chunks_file,EVAL_ENV_MANIFEST=$EVAL_ENV_MANIFEST,EVAL_CHUNKED=true,WANDB_MODE=disabled"
            "$SBATCH_SCRIPT" "$model" "$name")
    fi

    if [[ "${EVAL_DRY_RUN:-false}" == "true" ]]; then
        echo "[DRY RUN] ${eval_command[*]}" >&2
        array_job="dry-array-$attempt"
        controller_job="dry-controller-$attempt"
        echo "[DRY RUN] sbatch --parsable --job-name=$controller_name --dependency=afterany:$array_job scripts/evaluation_controller.sbatch" >&2
    else
        array_job=$("${eval_command[@]}")
        controller_job=$(sbatch --parsable --job-name="$controller_name" \
            --dependency="afterany:$array_job" \
            --export=ALL scripts/evaluation_controller.sbatch \
            "$model" "$name" "$state_dir" "$attempt" "$chunk_size")
    fi

    echo "Submitted $chunk_count chunks (chunk_size=$chunk_size, max_parallel=${EVAL_MAX_PARALLEL:-$chunk_count}) as array $array_job"
    echo "Controller job: $controller_job (afterany:$array_job)"
    EVAL_SUBMITTED_JOB_ID="$controller_job"
}

_eval_prepare_environment() {
    local state_dir="$1" repo="$2"
    EVAL_ENV_MANIFEST="$state_dir/environment.sh"
    export EVAL_ENV_MANIFEST EVAL_HARNESS_REPO="$repo"
    export EVAL_HARNESS_REF="${LM_EVAL_HARNESS_BRANCH:-HEAD}"
    EVAL_CONTAINER_CONFIG=$(_eval_container_for_backend)
    EVAL_CONTAINER_CONFIG=$(realpath "$EVAL_CONTAINER_CONFIG")
    export EVAL_CONTAINER_CONFIG EVAL_REPO_ROOT="$PWD"
    export EVAL_RUNTIME_REQUIREMENTS="$PWD/requirements/eval-runtime.txt"

    if [[ "${EVAL_DRY_RUN:-false}" == "true" ]]; then
        echo "[DRY RUN] sbatch --parsable scripts/prepare_eval_env.sbatch" >&2
        EVAL_PREP_JOB_ID="dry-environment"
    else
        EVAL_PREP_JOB_ID=$(sbatch --parsable --export=ALL scripts/prepare_eval_env.sbatch)
    fi
    export EVAL_PREP_JOB_ID
    echo "Environment preparation job: $EVAL_PREP_JOB_ID ($repo@${EVAL_HARNESS_REF})"
}

submit_evaluation() {
    local model="$1" name="$2"
    local repo_root logs_root run_root state_dir missing_count repo legacy_project
    local -a force_patterns=()
    repo_root=$(_eval_repo_root)
    cd "$repo_root"

    logs_root=${LOGS_ROOT:-${SCRATCH:-/tmp}/eval_logs_start}
    run_root="$logs_root/${WANDB_ENTITY:-apertus}/${WANDB_PROJECT:-swissai-evals-test}/$name"
    EVAL_HARNESS_DIR="$run_root/harness"
    legacy_project="${WANDB_PROJECT:-swissai-evals-test}"
    [[ "$legacy_project" == *-single ]] || legacy_project="${legacy_project}-single"
    EVAL_LEGACY_HARNESS_DIR="$logs_root/${WANDB_ENTITY:-apertus}/$legacy_project/$name/harness"
    export EVAL_HARNESS_DIR EVAL_LEGACY_HARNESS_DIR
    state_dir="$EVAL_HARNESS_DIR/controller/run_$(date +%Y%m%d_%H%M%S)_$$"
    mkdir -p "$state_dir"
    python3 -m scripts.eval_state normalize --tasks "$TASKS" \
        --output "$state_dir/expected_tasks.txt"

    if [[ -n "${EVAL_FORCE_TASKS:-}" ]]; then
        IFS=',' read -ra force_patterns <<< "$EVAL_FORCE_TASKS"
        for pattern in "${force_patterns[@]}"; do
            printf '%s\n' "$pattern" >> "$state_dir/force_patterns.txt"
        done
        printf 'eval_%s\n' "$(date +%Y%m%d_%H%M%S)" > "$state_dir/force_after.txt"
    fi
    _eval_scan "$state_dir" 0
    missing_count=$(wc -l < "$state_dir/missing_0.txt" | tr -d ' ')

    echo "Evaluation state: $state_dir"
    echo "Completed: $(wc -l < "$state_dir/completed_0.tsv" | tr -d ' ')"
    echo "Missing:   $missing_count"

    if [[ "${EVAL_MERGE_ONLY:-false}" == "true" || $missing_count -eq 0 ]]; then
        if [[ ! -s "$state_dir/eval_dirs_0.txt" ]]; then
            echo "No completed evaluation directories are available to aggregate" >&2
            return 1
        fi
        _eval_submit_aggregator "$model" "$name" "$state_dir/eval_dirs_0.txt"
        _eval_schedule_judge_cleanup "$EVAL_SUBMITTED_JOB_ID"
        ORCHESTRATION_FINAL_JOB_ID="$EVAL_SUBMITTED_JOB_ID"
        return
    fi

    if grep -Eq '^(bfcl_v3|swiss_ai_charter_alignment)([/:_-]|$)' \
        "$state_dir/expected_tasks.txt"; then
        repo="ymetz/lm-evaluation-harness"
    else
        repo="swiss-ai/lm-evaluation-harness"
    fi
    _eval_prepare_environment "$state_dir" "$repo"

    if [[ "${EVAL_FAILURE_POLICY:-resume}" == "fail-fast" ]]; then
        local safe_name job_id
        safe_name=${name//[^a-zA-Z0-9_.-]/-}
        if [[ "${EVAL_DRY_RUN:-false}" == "true" ]]; then
            echo "[DRY RUN] sbatch evaluate (fail-fast)" >&2
            job_id="dry-evaluation"
        else
            job_id=$(sbatch --parsable --job-name="eval-${safe_name}" \
                --dependency="afterok:${EVAL_PREP_JOB_ID}" \
                --export="ALL,EVAL_ENV_MANIFEST=$EVAL_ENV_MANIFEST,EVAL_CHUNKED=false" \
                "$SBATCH_SCRIPT" "$model" "$name")
        fi
        echo "Fail-fast evaluation job: $job_id"
        ORCHESTRATION_FINAL_JOB_ID="$job_id"
        return
    fi

    _eval_submit_wave "$model" "$name" "$state_dir" 0 "$EVAL_CHUNK_SIZE"
    ORCHESTRATION_FINAL_JOB_ID="$EVAL_SUBMITTED_JOB_ID"
}

evaluation_controller() {
    local model="$1" name="$2" state_dir="$3" attempt="$4" chunk_size="$5"
    local missing_count completed_count next_attempt next_chunk aggregate_job legacy_project
    cd "$(_eval_repo_root)"
    EVAL_HARNESS_DIR="${LOGS_ROOT:-${SCRATCH:-/tmp}/eval_logs_start}/${WANDB_ENTITY:-apertus}/${WANDB_PROJECT:-swissai-evals-test}/$name/harness"
    legacy_project="${WANDB_PROJECT:-swissai-evals-test}"
    [[ "$legacy_project" == *-single ]] || legacy_project="${legacy_project}-single"
    EVAL_LEGACY_HARNESS_DIR="${LOGS_ROOT:-${SCRATCH:-/tmp}/eval_logs_start}/${WANDB_ENTITY:-apertus}/$legacy_project/$name/harness"
    export EVAL_HARNESS_DIR EVAL_LEGACY_HARNESS_DIR EVAL_ENV_MANIFEST="$state_dir/environment.sh"

    _eval_scan "$state_dir" "$((attempt + 1))"
    missing_count=$(wc -l < "$state_dir/missing_$((attempt + 1)).txt" | tr -d ' ')
    completed_count=$(wc -l < "$state_dir/completed_$((attempt + 1)).tsv" | tr -d ' ')
    echo "Controller attempt $attempt: $completed_count completed, $missing_count missing"

    if (( missing_count == 0 )); then
        _eval_submit_aggregator "$model" "$name" "$state_dir/eval_dirs_$((attempt + 1)).txt"
        aggregate_job="$EVAL_SUBMITTED_JOB_ID"
        _eval_schedule_judge_cleanup "$aggregate_job"
        echo "Evaluation complete"
        return
    fi

    if [[ ! -f "$EVAL_ENV_MANIFEST" ]]; then
        cp "$state_dir/missing_$((attempt + 1)).txt" "$state_dir/final_failed_tasks.txt"
        echo "Environment preparation failed; remaining tasks are in $state_dir/final_failed_tasks.txt" >&2
        _eval_schedule_judge_cleanup ""
        return 1
    fi

    if (( attempt < EVAL_MAX_RETRIES )); then
        next_attempt=$((attempt + 1))
        next_chunk=$((chunk_size / 2))
        (( next_chunk >= 1 )) || next_chunk=1
        echo "Retrying missing tasks with chunk_size=$next_chunk"
        _eval_submit_wave "$model" "$name" "$state_dir" "$next_attempt" "$next_chunk"
        return
    fi

    cp "$state_dir/missing_$((attempt + 1)).txt" "$state_dir/final_failed_tasks.txt"
    echo "Retry budget exhausted; failed tasks:" >&2
    sed 's/^/  - /' "$state_dir/final_failed_tasks.txt" >&2
    if [[ -s "$state_dir/eval_dirs_$((attempt + 1)).txt" ]]; then
        _eval_submit_aggregator "$model" "$name" \
            "$state_dir/eval_dirs_$((attempt + 1)).txt" "$state_dir/final_failed_tasks.txt"
        aggregate_job="$EVAL_SUBMITTED_JOB_ID"
        _eval_schedule_judge_cleanup "$aggregate_job"
    else
        _eval_schedule_judge_cleanup ""
    fi
    return 1
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    set -euo pipefail
    command=${1:-}
    shift || true
    case "$command" in
        controller) evaluation_controller "$@" ;;
        *) echo "Usage: $0 controller <model> <name> <state-dir> <attempt> <chunk-size>" >&2; exit 2 ;;
    esac
fi
