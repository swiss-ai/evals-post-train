#!/bin/bash
#
# run_tau2_bench.sh - Run tau2-bench (https://github.com/sierra-research/tau2-bench) straight from
# upstream: any ref of the repository, with the dependencies upstream's own uv.lock pins, and any
# domain that ref has (mock, airline, retail, telecom, telecom-workflow, banking_knowledge, intake
# at v2.0.0). Self-contained: no framework of this repo in between, and no lock of ours -- the ref
# is the version. The model under test is reached the way run_inspect_eval.sh reaches its model;
# the defaults are plain tau2-bench's (nothing of any leaderboard's protocol).
#
# In the prebuilt image (containers/Dockerfile.eval, EVAL_PREBUILT_ENV=1) the tau2-bench checkout and its
# dependencies are baked in at the ref of requirements/tau2-upstream.commit: nothing is installed or
# downloaded, and --git-ref must name that ref. Elsewhere the ref is checked out into a cache keyed by its
# commit and `uv sync --frozen --extra knowledge` installs upstream's locked dependencies into the
# checkout's own .venv (Python 3.12); that needs git, uv and network access the first time a commit is used.
# Either way tau2 reads its data through a writable view in --workdir and writes its results there.
#
# Usage:
#   scripts/custom/run_tau2_bench.sh --domain <domain> --model <model> [--api-base-url <url>] [options]
#
# Required:
#   --domain <name>            A domain of the ref (see above; `tau2 --help` lists the CLI's).
#   --model <model>            Model under test. With --api-base-url, the served model name (wrapped
#                               through litellm's openai/ provider against that endpoint); without
#                               it, a model string litellm understands (with its own credentials).
#
# Options:
#   --api-base-url <url>       OpenAI-compatible endpoint serving --model (bare host or /v1 root).
#                               Its key: TARGET_API_KEY (default: CSCS_SERVING_API; both also from
#                               scripts/secrets.env, as in run_inspect_eval.sh).
#   --agent-llm-args <json>    Replace the agent's litellm kwargs (default: api_base, and api_key as
#                               "os.environ/TARGET_API_KEY": the key stays out of results.json and the logs).
#   --user-llm <model>         The simulated user (default: --model on the
#                               same endpoint, so no second credential). A bare name is reached by
#                               litellm with its own credentials (e.g. OPENAI_API_KEY).
#                               With --api-base-url the NL-assertion judge (retail) is the model under test too, at
#                               the same low effort and cap: upstream hardcodes an OpenAI model (tau2_judge/).
#   --user-llm-args <json>     Extra litellm kwargs for the user, merged over its default key by key. The default
#                               user (the model under test) runs at reasoning_effort low with max_tokens 4096: at its
#                               default effort a turn could run 15k tokens and end with no reply, and tau2 redoes the task.
#   --judge-llm <model>        The retail NL-assertion judge (default with --api-base-url: --model, which then judges
#                               its own agent runs). Named like --user-llm: openai/<served name> is reached on the same
#                               endpoint and key at the pin below; a bare name by litellm with its own credentials.
#                               A judge started with scripts/lm_eval_harness/launch_judge.py is hosted as
#                               <user>/<served name>: give the served name, the launcher finds the hosted id in
#                               the endpoint's /models and stops when the model is not served.
#                               Only used with --api-base-url, else the judge is upstream's own.
#   --judge-llm-args <json>    Extra litellm kwargs for the judge, merged over its default key by key (the default
#                               judge runs at reasoning_effort low with max_tokens 4096, like the default user).
#   --retrieval-config <name>  banking_knowledge: the retrieval pipeline (default: bm25, offline keyword search).
#                               tau2's own default embeds the documents with OpenAI and needs an OPENAI_API_KEY;
#                               those configs (openai_embeddings*, ...) are available by naming them.
#   --num-trials <n>           Trials per task (default: 1). The summary prints pass^k for k = 1..n.
#   --max-steps <n>            Max steps per trial (default: 100, tau2's).
#   --num-tasks <n>            Only the first n tasks (a smoke test).
#   --seed <n>                 Random seed (default: 300, tau2's).
#   --max-concurrency <n>      Parallel trials (default: 4).
#   --temperature V, --top-p V, --top-k V, --min-p V, --repetition-penalty V
#                              Sampling of the agent model, over its litellm args. tau2 has no default
#                              here: it sends no sampling setting unless you pass one (docs/sampling.md).
#                              temperature and top-p are litellm kwargs; the rest go in extra_body.
#   --sampling-profile <name>  A profile of the sampling config (e.g. thinking, greedy);
#   --sampling-config <file>   another file instead of configs/sampling_defaults.json.
#   --enable-thinking, --no-enable-thinking, --reasoning-effort <level>
#                              Thinking and reasoning effort of the agent model (scripts/common/thinking.py).
#                              The user simulator never gets them (--user-llm-args does).
#   --git-ref <ref>            tau2-bench tag, branch or full commit (default: v2.0.0). In the prebuilt
#                               image only the ref (or commit) baked in is accepted.
#   --repo <url>               Repository the ref is looked up in (default: upstream; not in the image).
#   --save-to <name>           tau2's run name, the results subdirectory (default: <domain>-<date and time>, a new
#                               run each time). Giving the name of an earlier run resumes it (--auto-resume):
#                               only with the same domain, model and options, else tau2 refuses.
#   --workdir <path>           Where the run's data view and results go (default: a temp dir under /tmp, which
#                               a job's container loses: give a path on storage you keep).
#   --cache-dir <path>         Outside the image: where checkouts live (default:
#                               ${EVAL_ENV_CACHE_ROOT:-$SCRATCH/eval-envs}/tau2-upstream,
#                               ~/.cache/evals-post-train/eval-envs/tau2-upstream without $SCRATCH).
#   -- <extra args>             Forwarded verbatim to `tau2 run` (e.g. a domain's own options).
#
# Results: <--workdir>/tau2-data/simulations/<--save-to>/results.json (every simulation, reward and
# messages); this script prints avg_reward and pass^k computed from it over the simulations that got a reward,
# lists the ones that did not (tau2 gives up on a task after 4 attempts: an infrastructure error, no reward) and
# the ones missing altogether, and exits 1 if there are any, so a lost task never passes as a score.
#
# Example (retail smoke test against a CSCS-served model):
#   scripts/custom/run_tau2_bench.sh --domain retail --model CSCS-Inference/swiss-ai/Apertus-v1.5-8B \
#     --api-base-url https://api.swissai.svc.cscs.ch/v1 --num-tasks 2

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
JUDGE_SHIM_DIR="$PWD/scripts/custom/tau2_judge"

die() { echo "$*" >&2; exit 1; }

usage() {
    sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'
}

DOMAIN=""
MODEL=""
API_BASE_URL=${API_BASE_URL:-""}
AGENT_LLM_ARGS_OVERRIDE=""
USER_LLM=""
USER_LLM_ARGS_OVERRIDE=""
JUDGE_LLM=""
JUDGE_LLM_ARGS_OVERRIDE=""
RETRIEVAL_CONFIG=""
NUM_TRIALS=1
MAX_STEPS=100
NUM_TASKS=""
SEED=300
MAX_CONCURRENCY=4
TAU_GIT_REF="v2.0.0"
TAU_REPO="https://github.com/sierra-research/tau2-bench.git"
SAVE_TO=""
CACHE_DIR=""
WORKDIR=""
SAMPLING_ARGS=(--framework tau2 --format litellm --warn-greedy)
THINKING_ARGS=(--format json)
EXTRA_ARGS=()

while (( $# > 0 )); do
    case "$1" in
        --domain) DOMAIN=$2; shift 2 ;;
        --model) MODEL=$2; shift 2 ;;
        --api-base-url) API_BASE_URL=$2; shift 2 ;;
        --agent-llm-args) AGENT_LLM_ARGS_OVERRIDE=$2; shift 2 ;;
        --user-llm) USER_LLM=$2; shift 2 ;;
        --user-llm-args) USER_LLM_ARGS_OVERRIDE=$2; shift 2 ;;
        --judge-llm) JUDGE_LLM=$2; shift 2 ;;
        --judge-llm-args) JUDGE_LLM_ARGS_OVERRIDE=$2; shift 2 ;;
        --retrieval-config) RETRIEVAL_CONFIG=$2; shift 2 ;;
        --num-trials) NUM_TRIALS=$2; shift 2 ;;
        --max-steps) MAX_STEPS=$2; shift 2 ;;
        --num-tasks) NUM_TASKS=$2; shift 2 ;;
        --seed) SEED=$2; shift 2 ;;
        --max-concurrency) MAX_CONCURRENCY=$2; shift 2 ;;
        --temperature|--top-p|--top-k|--min-p|--repetition-penalty)
            SAMPLING_ARGS+=("$1" "$2"); shift 2 ;;
        --sampling-profile) SAMPLING_ARGS+=(--profile "$2"); shift 2 ;;
        --sampling-config) SAMPLING_ARGS+=(--config "$2"); shift 2 ;;
        --enable-thinking|--no-enable-thinking)
            SAMPLING_ARGS+=("$1"); THINKING_ARGS+=("$1"); shift ;;
        --reasoning-effort)
            SAMPLING_ARGS+=("$1" "$2"); THINKING_ARGS+=("$1" "$2"); shift 2 ;;
        --git-ref) TAU_GIT_REF=$2; shift 2 ;;
        --repo) TAU_REPO=$2; shift 2 ;;
        --save-to) SAVE_TO=$2; shift 2 ;;
        --cache-dir) CACHE_DIR=$2; shift 2 ;;
        --workdir) WORKDIR=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        *) die "Unknown argument: $1 (see --help)" ;;
    esac
done

[[ -n "$DOMAIN" ]] || die "Missing --domain. See --help."
[[ -n "$MODEL" ]] || die "Missing --model. See --help."
# The domain goes into a path and a command line: a name, nothing else.
[[ "$DOMAIN" =~ ^[A-Za-z0-9_-]+$ ]] || die "--domain must be a domain name, got: $DOMAIN"
SAVE_TO=${SAVE_TO:-"$DOMAIN-$(date +%Y%m%d-%H%M%S)"}
[[ "$SAVE_TO" =~ ^[A-Za-z0-9_.-]+$ ]] || die "--save-to must be a plain name, got: $SAVE_TO"

# The agent: through the endpoint like run_inspect_eval.sh does, or a bare litellm model string.
AGENT_LLM_ARGS='{}'
if [[ -n "$API_BASE_URL" ]]; then
    [[ "$API_BASE_URL" == */v1 || "$API_BASE_URL" == */v1/ ]] || API_BASE_URL="${API_BASE_URL%/}/v1"
    # shellcheck source=scripts/common/secrets.sh
    source scripts/common/secrets.sh
    export TARGET_API_KEY="${TARGET_API_KEY:-${CSCS_SERVING_API:-}}"
    # The key never goes into the llm args: tau2 writes them to results.json and to its LLM call logs, and
    # prints them in a resume diff. litellm resolves "os.environ/<NAME>" from the environment at call time.
    AGENT_LLM_ARGS=$(python3 -c "import json,sys; print(json.dumps({'api_base': sys.argv[1], 'api_key': 'os.environ/TARGET_API_KEY'}))" \
        "$API_BASE_URL")
    AGENT_LLM="openai/$MODEL"
else
    AGENT_LLM="$MODEL"
fi
[[ -z "$AGENT_LLM_ARGS_OVERRIDE" ]] || AGENT_LLM_ARGS=$AGENT_LLM_ARGS_OVERRIDE

# Sampling of the agent (scripts/common/sampling.py), over its litellm args key by key (extra_body too).
SAMPLING_LITELLM=$(python3 -m scripts.common.sampling "${SAMPLING_ARGS[@]}") || exit 1
AGENT_LLM_ARGS=$(python3 -c '
import json, sys
args, sampling = json.loads(sys.argv[1]), json.loads(sys.argv[2])
body = {**args.get("extra_body", {}), **sampling.pop("extra_body", {})}
args.update(sampling)
if body:
    args["extra_body"] = body
print(json.dumps(args))' "$AGENT_LLM_ARGS" "$SAMPLING_LITELLM")

# The user simulator: by default the model under test on the same endpoint, with the agent's litellm
# args as they are now (sampling included, but not the thinking flags, added below). A bare name is
# litellm's own business (its credentials, no api_base).
if [[ -z "$USER_LLM" ]]; then
    [[ -n "$API_BASE_URL" ]] || die "Without --api-base-url the user simulator needs its own --user-llm."
    USER_LLM=$AGENT_LLM
fi
# The litellm kwargs of a thinking setting (scripts/common/thinking.py): $1 the args so far, $2 its values (json).
with_thinking() {
    python3 -c '
import json, sys
from scripts.common import thinking
args, values = json.loads(sys.argv[1]), json.loads(sys.argv[2])
args.update({k: v for k, v in thinking.litellm_kwargs(values).items() if k != "extra_body"})
body = thinking.merge_extra_body(args.get("extra_body"), values)
if body:
    args["extra_body"] = body
print(json.dumps(args))' "$1" "$2"
}
# The model under test as its own user and judge: low reasoning effort and a capped reply. At the model's default
# effort a user turn sometimes ran 15k tokens (7 min) and ended with no reply: tau2 rejects an empty message and
# redoes the whole task, up to 4 times. --user-llm-args overrides both for the user.
if [[ -n "$API_BASE_URL" ]]; then
    PINNED_LLM_ARGS=$(with_thinking "$AGENT_LLM_ARGS" "$(python3 -m scripts.common.thinking --format json --reasoning-effort low)") || exit 1
    PINNED_LLM_ARGS=$(python3 -c '
import json, sys
args = json.loads(sys.argv[1]); args["max_tokens"] = 4096; print(json.dumps(args))' "$PINNED_LLM_ARGS")
fi
if [[ "$USER_LLM" == openai/* && -n "$API_BASE_URL" ]]; then
    USER_LLM_ARGS=$PINNED_LLM_ARGS
else
    USER_LLM_ARGS='{}'
fi
if [[ -n "$USER_LLM_ARGS_OVERRIDE" ]]; then
    USER_LLM_ARGS=$(python3 -c '
import json, sys
merged = json.loads(sys.argv[1])
merged.update(json.loads(sys.argv[2]))
print(json.dumps(merged))' "$USER_LLM_ARGS" "$USER_LLM_ARGS_OVERRIDE")
fi

# Thinking and reasoning effort of the agent (scripts/common/thinking.py), after the user copied its args.
THINKING_VALUES=$(python3 -m scripts.common.thinking "${THINKING_ARGS[@]}") || exit 1
AGENT_LLM_ARGS=$(with_thinking "$AGENT_LLM_ARGS" "$THINKING_VALUES")

# The commit a ref names in $TAU_REPO (a full commit is taken as is).
resolve_ref() {
    local ref=$1 out sha
    if [[ "$ref" =~ ^[0-9a-f]{40}$ ]]; then echo "$ref"; return; fi
    out=$(git ls-remote "$TAU_REPO" "refs/tags/$ref^{}" "refs/tags/$ref" "refs/heads/$ref") || return 1
    sha=$(grep -m1 '\^{}$' <<<"$out" | cut -f1 || true)
    [[ -n "$sha" ]] || sha=$(head -1 <<<"$out" | cut -f1)
    [[ -n "$sha" ]] && echo "$sha"
}

DEFAULT_REPO="https://github.com/sierra-research/tau2-bench.git"
if [[ "${EVAL_PREBUILT_ENV:-0}" == "1" && -n "${TAU2_UPSTREAM_DIR:-}" ]]; then
    # The prebuilt image: upstream's checkout and its .venv are baked in at requirements/tau2-upstream.commit.
    [[ -x "$TAU2_UPSTREAM_DIR/.venv/bin/tau2" && -f "${TAU2_UPSTREAM_REF_FILE:-}" ]] ||
        die "EVAL_PREBUILT_ENV=1 but this image has no upstream tau2-bench (TAU2_UPSTREAM_DIR=$TAU2_UPSTREAM_DIR)." \
            "Use the image built from a commit that has requirements/tau2-upstream.commit."
    read -r BAKED_REF BAKED_COMMIT < "$TAU2_UPSTREAM_REF_FILE"
    [[ "$TAU_REPO" == "$DEFAULT_REPO" ]] || die "--repo is not available in the prebuilt image (it has $DEFAULT_REPO@$BAKED_REF)."
    [[ "$TAU_GIT_REF" == "$BAKED_REF" || "$TAU_GIT_REF" == "$BAKED_COMMIT" ]] ||
        die "--git-ref $TAU_GIT_REF: this image has tau2-bench $BAKED_REF ($BAKED_COMMIT). Use that ref," \
            "run outside the image (EVAL_PREBUILT_ENV unset) to install another, or bump requirements/tau2-upstream.commit."
    COMMIT=$BAKED_COMMIT
    CHECKOUT=$TAU2_UPSTREAM_DIR
    echo "EVAL_PREBUILT_ENV=1: tau2-bench@$COMMIT ($BAKED_REF) from $CHECKOUT, no install"
else
    command -v git >/dev/null 2>&1 || die "git is required."
    command -v uv >/dev/null 2>&1 || die "uv is required (https://docs.astral.sh/uv/): it installs upstream's uv.lock."
    COMMIT=$(resolve_ref "$TAU_GIT_REF") || die "--git-ref $TAU_GIT_REF: not found in $TAU_REPO"
    [[ -n "$COMMIT" ]] || die "--git-ref $TAU_GIT_REF: not found in $TAU_REPO"
    ROOT=${CACHE_DIR:-${EVAL_ENV_CACHE_ROOT:-${SCRATCH:-${XDG_CACHE_HOME:-$HOME/.cache}/evals-post-train}/eval-envs}/tau2-upstream}
    # A venv only works with the machine and Python that built it, so the key has the architecture too.
    CHECKOUT="$ROOT/$COMMIT-$(uname -m)"
    mkdir -p "$ROOT"

    # Simultaneous runs wait for the first builder (flock is missing on macOS, where runs rarely race).
    exec 9>"$ROOT/$COMMIT.lock"
    if command -v flock >/dev/null 2>&1; then flock 9; fi
    if [[ ! -f "$CHECKOUT/.complete" ]]; then
        echo "Installing tau2-bench@$COMMIT ($TAU_GIT_REF) into $CHECKOUT"
        # Built in place (a venv embeds its path); an interrupted build is kept for diagnosis.
        if [[ -e "$CHECKOUT" ]]; then mv "$CHECKOUT" "$CHECKOUT.incomplete.$(date +%s).$$"; fi
        git init --quiet "$CHECKOUT"
        git -C "$CHECKOUT" fetch --quiet --depth 1 "$TAU_REPO" "$COMMIT"
        git -C "$CHECKOUT" checkout --quiet FETCH_HEAD
        # Upstream's own pins (uv.lock); the knowledge extra is banking_knowledge's retrieval.
        (cd "$CHECKOUT" && uv sync --frozen --quiet --extra knowledge --python 3.12) ||
            die "uv sync --frozen of tau2-bench@$COMMIT failed (does that ref have a uv.lock and a knowledge extra?)"
        [[ -x "$CHECKOUT/.venv/bin/tau2" ]] || die "tau2-bench@$COMMIT installed no tau2 command."
        touch "$CHECKOUT/.complete"
    fi
    exec 9>&-
    touch "$CHECKOUT/.complete"  # also refreshes it for age-based scratch retention
    echo "tau2-bench@$COMMIT ($TAU_GIT_REF) from $CHECKOUT"
fi

# tau2 reads its data from TAU2_DATA_DIR and writes its results next to it: a writable view of the checkout's
# data in the workdir (the image is read-only), the same way for both sources.
WORKDIR=${WORKDIR:-"$(mktemp -d /tmp/tau2-custom.XXXXXX)"}
VIEW="$WORKDIR/tau2-data"
mkdir -p "$VIEW"
VIEW=$(cd "$VIEW" && pwd)  # absolute: tau2 runs from the checkout, so a relative --workdir would point elsewhere
for entry in "$CHECKOUT"/data/* "$CHECKOUT"/data/.[!.]*; do
    [[ -e "$entry" && "$(basename "$entry")" != simulations ]] && ln -sfn "$entry" "$VIEW/"
done
export TAU2_DATA_DIR="$VIEW"

ARGS=(run --domain "$DOMAIN"
      --agent-llm "$AGENT_LLM" --agent-llm-args "$AGENT_LLM_ARGS"
      --user-llm "$USER_LLM" --user-llm-args "$USER_LLM_ARGS"
      --num-trials "$NUM_TRIALS" --max-steps "$MAX_STEPS" --seed "$SEED"
      --max-concurrency "$MAX_CONCURRENCY"
      --save-to "$SAVE_TO" --auto-resume --log-level INFO)
# banking_knowledge retrieves from a knowledge base: bm25 runs offline, tau2's own default needs OpenAI embeddings.
[[ -n "$RETRIEVAL_CONFIG" || "$DOMAIN" != banking_knowledge ]] || RETRIEVAL_CONFIG=bm25
[[ -z "$RETRIEVAL_CONFIG" ]] || ARGS+=(--retrieval-config "$RETRIEVAL_CONFIG")
[[ -z "$NUM_TASKS" ]] || ARGS+=(--num-tasks "$NUM_TASKS")
(( ${#EXTRA_ARGS[@]} == 0 )) || ARGS+=("${EXTRA_ARGS[@]}")

echo "+ tau2 run --domain $DOMAIN --agent-llm $AGENT_LLM --user-llm $USER_LLM" \
     "--num-trials $NUM_TRIALS --max-steps $MAX_STEPS --seed $SEED" \
     "--max-concurrency $MAX_CONCURRENCY (llm args name the key by its environment variable, never the key)"
# The NL-assertion judge (retail) is hardcoded upstream to an OpenAI model, which we have no key for: with an
# endpoint, it is --judge-llm, else the model under test (tau2_judge/sitecustomize.py repoints it).
if [[ -n "$API_BASE_URL" ]]; then
    JUDGE_LLM=${JUDGE_LLM:-$AGENT_LLM}
    if [[ "$JUDGE_LLM" == openai/* && "$JUDGE_LLM" != "$AGENT_LLM" ]]; then
        # A launched judge is hosted under its launcher's scope (<user>/<served name>); a shared one as named.
        JUDGE_HOSTED=$(python3 -c '
import getpass, json, os, sys, urllib.request
base, name = sys.argv[1:]
request = urllib.request.Request(base.rstrip("/") + "/models", headers={"Authorization": "Bearer " + os.environ.get("TARGET_API_KEY", "")})
try:
    with urllib.request.urlopen(request, timeout=30) as response:
        served = [m["id"] for m in json.load(response)["data"]]
except Exception as error:
    sys.exit(f"cannot list the models of {base}: {error}")
for candidate in (f"{getpass.getuser()}/{name}", name):
    if candidate in served:
        print(candidate)
        break
else:
    sys.exit(f"judge {name!r} is not served on {base} (looked for {getpass.getuser()}/{name} and {name}); start it with scripts/lm_eval_harness/launch_judge.py")
' "$API_BASE_URL" "${JUDGE_LLM#openai/}") || exit 1
        JUDGE_LLM="openai/$JUDGE_HOSTED"
    fi
    if [[ "$JUDGE_LLM" == openai/* ]]; then JUDGE_LLM_ARGS=$PINNED_LLM_ARGS; else JUDGE_LLM_ARGS='{}'; fi
    if [[ -n "$JUDGE_LLM_ARGS_OVERRIDE" ]]; then
        JUDGE_LLM_ARGS=$(python3 -c '
import json, sys
merged = json.loads(sys.argv[1])
merged.update(json.loads(sys.argv[2]))
print(json.dumps(merged))' "$JUDGE_LLM_ARGS" "$JUDGE_LLM_ARGS_OVERRIDE")
    fi
    export TAU2_JUDGE_LLM=$JUDGE_LLM TAU2_JUDGE_LLM_ARGS=$JUDGE_LLM_ARGS
    if [[ "$JUDGE_LLM" == "$AGENT_LLM" ]]; then echo "NL-assertion judge: $TAU2_JUDGE_LLM (the model under test, through $API_BASE_URL)"
    else echo "NL-assertion judge: $TAU2_JUDGE_LLM"; fi
else
    echo "NL-assertion judge: upstream's own (OpenAI; needs OPENAI_API_KEY for retail)"
fi
(cd "$CHECKOUT" && PYTHONPATH="$JUDGE_SHIM_DIR${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/tau2 "${ARGS[@]}")

RESULTS_FILE="$VIEW/simulations/$SAVE_TO/results.json"
[[ -f "$RESULTS_FILE" ]] || RESULTS_FILE=$(find "$VIEW/" -path "*/$SAVE_TO/results.json" 2>/dev/null | head -1)
[[ -n "$RESULTS_FILE" && -f "$RESULTS_FILE" ]] || die "tau2 produced no results.json"
echo "results: $RESULTS_FILE"

EXPECTED=""
[[ -z "$NUM_TASKS" ]] || EXPECTED=$((NUM_TASKS * NUM_TRIALS))
python3 - "$RESULTS_FILE" "$NUM_TRIALS" "$EXPECTED" <<'PY'
import json
import math
import sys
from collections import defaultdict

with open(sys.argv[1]) as f:
    data = json.load(f)
num_trials = int(sys.argv[2])
expected = int(sys.argv[3]) if sys.argv[3] else None

all_sims = data.get("simulations") or []
sims = [s for s in all_sims if (s.get("reward_info") or {}).get("reward") is not None]
unscored = [s for s in all_sims if (s.get("reward_info") or {}).get("reward") is None]
per_task = defaultdict(list)
for s in sims:
    per_task[s["task_id"]].append(s)

if not per_task:
    print("no scored simulations")
    raise SystemExit(1)

rewards = [s["reward_info"]["reward"] for s in sims]
metrics = {"avg_reward": sum(rewards) / len(rewards)}


def is_success(reward):
    return (1 - 1e-6) <= reward <= (1 + 1e-6)


for k in range(1, num_trials + 1):
    vals = []
    for runs in per_task.values():
        n = len(runs)
        c = sum(is_success(s["reward_info"]["reward"]) for s in runs)
        if n >= k:
            vals.append(math.comb(c, k) / math.comb(n, k))
    if vals:
        metrics[f"pass^{k}"] = sum(vals) / len(vals)
metrics["n_tasks"] = len(per_task)
metrics["n_simulations"] = len(sims)
metrics["n_unscored"] = len(unscored)
if expected is not None:
    metrics["n_missing"] = max(expected - len(all_sims), 0)
print(json.dumps(metrics, indent=2))
problems = []
for s in unscored:
    problems.append("task %s: no reward (%s)" % (s.get("task_id"), s.get("termination_reason")))
if expected is not None and len(all_sims) < expected:
    problems.append("%d of %d simulations are not in results.json" % (expected - len(all_sims), expected))
if problems:
    print("NOT A COMPLETE RESULT: " + "; ".join(problems), file=sys.stderr)
    raise SystemExit(1)
PY
