# wandb.sh - Whether, and where, a run_*_eval.sh wrapper uploads its results to W&B. Sourced
# from the repo root.
#
# Uploads are on by default, as for lm-evaluation-harness (launch_evaluations.sh): to
# WANDB_ENTITY/WANDB_PROJECT (--wandb-entity/--wandb-project), by default
# apertus/apertus-1.5-post-training-v0.1, whenever a W&B key is available (WANDB_API_KEY, else
# scripts/wandb_api_key.txt). Off with --no-wandb or WANDB_MODE=disabled, and for --gold runs
# (a pipeline check, not a model score).

WANDB_DEFAULT_ENTITY=apertus
WANDB_DEFAULT_PROJECT=apertus-1.5-post-training-v0.1

# wandb_setup [<reason to skip>]: sets WANDB_UPLOAD (1 or 0) and the WANDB_ENTITY/WANDB_PROJECT to
# upload to, and exports WANDB_API_KEY. Fails when an entity or project was given but there is
# no key: that upload was asked for.
wandb_setup() {
    local skip=${1:-} asked=0
    [[ -n "${WANDB_ENTITY:-}${WANDB_PROJECT:-}" ]] && asked=1
    [[ -z "$skip" && "${WANDB_MODE:-}" == disabled ]] && skip="WANDB_MODE=disabled"
    WANDB_UPLOAD=0
    if [[ -n "$skip" ]]; then
        echo "W&B upload: off ($skip)"
        return 0
    fi
    WANDB_ENTITY=${WANDB_ENTITY:-$WANDB_DEFAULT_ENTITY}
    WANDB_PROJECT=${WANDB_PROJECT:-$WANDB_DEFAULT_PROJECT}
    WANDB_API_KEY=${WANDB_API_KEY:-}
    if [[ -z "$WANDB_API_KEY" && -f ./scripts/wandb_api_key.txt ]]; then
        WANDB_API_KEY="$(tr -d '\r\n' < ./scripts/wandb_api_key.txt)"
    fi
    if [[ -z "$WANDB_API_KEY" ]]; then
        if (( asked )); then
            echo "W&B upload to $WANDB_ENTITY/$WANDB_PROJECT requested, but WANDB_API_KEY is not set" \
                "and scripts/wandb_api_key.txt is missing (--no-wandb to skip the upload)." >&2
            return 1
        fi
        echo "W&B upload: off (no WANDB_API_KEY or scripts/wandb_api_key.txt)"
        return 0
    fi
    export WANDB_API_KEY
    WANDB_UPLOAD=1
    echo "W&B upload: $WANDB_ENTITY/$WANDB_PROJECT (--no-wandb to skip)"
}
