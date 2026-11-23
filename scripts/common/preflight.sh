# shellcheck shell=bash
# preflight.sh - Lets a runner stop after checking its arguments and keys. Sourced by every runner that a
# direct-serving job (scripts/direct_serving/job.py) can run as an --eval.
#
# The job runs each --eval command once with EVAL_PREFLIGHT=1 before it loads the model, so a missing key or a
# bad argument fails the job in seconds, not after the server's half hour. A runner calls preflight_done right
# after its own checks, and before it installs, downloads or submits anything.

# preflight_done: with EVAL_PREFLIGHT=1, ends the runner here, successfully.
preflight_done() {
    if [[ "${EVAL_PREFLIGHT:-0}" == 1 ]]; then
        echo "preflight ok: $(basename "$0")"
        exit 0
    fi
}
