#!/bin/bash
#
# run_gdpval.sh - Run GDPval-AA v2 (https://artificialanalysis.ai/evaluations/gdpval-aa) against
# a model, standalone -- the same real protocol evals-svc's own gdpval.py suite runs (Stirrup,
# AA's own open-source agent harness, github.com/ArtificialAnalysis/Stirrup), extracted here so
# it's runnable directly from evals-post-train without going through evals-svc's API. NOT the
# same as this repo's inspect_evals/gdpval (documented separately above): that's a narrower,
# independent port with a different (bash/python-in-Docker, no web) tool surface; this script
# drives Stirrup, matching AA's stated protocol tools (Web Fetch, Web Search, View Image, Code
# Exec, Finish, Abandon Task) and evals-svc's own implementation exactly.
#
# Protocol (AA's methodology page): the full 220-task openai/gdpval gold set, 1 generation per
# task, up to 250 turns, a code-exec sandbox (E2B by default here, matching AA's real protocol;
# evals-svc itself defaults to a free local subprocess sandbox instead -- see --sandbox-backend
# below for the cost/isolation tradeoff). GRADING here is a proxy, not AA's real cross-model Elo
# (that needs their whole comparison pool, not reproducible standalone): a win/tie/loss rate
# from pairwise comparisons between the model's own deliverable and each task's human-expert
# reference deliverable, judged by whichever of a 3-provider judge panel (OpenAI/Google/
# Anthropic) has credentials configured.
#
# The model under test is reached the same way run_inspect_eval.sh reaches its model:
# --api-base-url wraps --model as an OpenAI-compatible endpoint; without it, --model is passed
# straight through Stirrup's OpenAI-SDK client (needs its own OPENAI_API_KEY).
#
# Usage:
#   aaii/run_gdpval.sh --model <model> [--api-base-url <url>] [options]
#
# Required:
#   --model <model>             Model under test.
#
# Options:
#   --api-base-url <url>        OpenAI-compatible endpoint serving --model. Reads the key from
#                               TARGET_API_KEY (default: scripts/cscs_serving_api_key.txt, same
#                               fallback run_inspect_eval.sh uses).
#   --num-tasks <n>              Restrict to the first n tasks (default: 220, AA's full set;
#                               smoke-test with a small number). Ignored when --task-ids is given.
#   --task-ids <id1 id2 ...>    Exactly these space-separated openai/gdpval task_ids, instead of
#                               --num-tasks.
#   --max-turns <n>              Max agent turns per task (default: 250, AA's protocol).
#   --max-concurrency <n>        Parallel tasks (default: 4).
#   --sandbox-backend <name>    "e2b" (default here -- AA's real protocol; isolated, real
#                               per-second cost, needs E2B_API_KEY) or "local" (free plain
#                               subprocess, NOT isolated from the agent's own generated code --
#                               evals-svc's own default, for cost reasons).
#   --e2b-template <name>       E2B sandbox template (default: code-interpreter-v1).
#   --workdir <path>            Scratch dir for generated deliverables (default: a temp dir
#                               under /tmp).
#   --no-grade                  Skip pairwise grading even if judge credentials are set --
#                               deliverables are still generated.
#
# Environment variables:
#   E2B_API_KEY                 Required when --sandbox-backend e2b (the default).
#   BRAVE_API_KEY                Optional: web-search tool. Web-fetch works without it; missing
#                               this is a tracked protocol deviation, not a refusal.
#   GDPVAL_JUDGE_OPENAI_MODEL/_API_KEY, GDPVAL_JUDGE_GOOGLE_MODEL/_API_KEY,
#   GDPVAL_JUDGE_ANTHROPIC_MODEL/_API_KEY   Judge panel (need at least one MODEL+KEY pair to
#                               grade at all; ungraded deliverables are still written).
#
# Results: one directory per task under <workdir>/gdpval_runs/<task_id>/ (every deliverable
# file the agent produced), a printed win/tie/loss summary once grading completes (or a note
# that no judge credentials were configured), and a structured <workdir>/gdpval_result.json
# (status/results/samples/error, same shape evals-svc's own callback payload uses).
#
# Example (smoke test, free local sandbox, no grading credentials):
#   aaii/run_gdpval.sh --model CSCS-Inference/swiss-ai/Apertus-v1.5-8B \
#     --api-base-url https://api.swissai.svc.cscs.ch/v1 --sandbox-backend local --num-tasks 1

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

die() { echo "$*" >&2; exit 1; }

usage() {
    sed -n '2,/^set -uo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'
}

MODEL=""
API_BASE_URL=${API_BASE_URL:-""}
NUM_TASKS=220
TASK_IDS=""
MAX_TURNS=250
MAX_CONCURRENCY=4
SANDBOX_BACKEND="e2b"
E2B_TEMPLATE="code-interpreter-v1"
WORKDIR=""
NO_GRADE=0

while (( $# > 0 )); do
    case "$1" in
        --model) MODEL=$2; shift 2 ;;
        --api-base-url) API_BASE_URL=$2; shift 2 ;;
        --num-tasks) NUM_TASKS=$2; shift 2 ;;
        --task-ids) TASK_IDS=$2; shift 2 ;;
        --max-turns) MAX_TURNS=$2; shift 2 ;;
        --max-concurrency) MAX_CONCURRENCY=$2; shift 2 ;;
        --sandbox-backend) SANDBOX_BACKEND=$2; shift 2 ;;
        --e2b-template) E2B_TEMPLATE=$2; shift 2 ;;
        --workdir) WORKDIR=$2; shift 2 ;;
        --no-grade) NO_GRADE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown argument: $1 (see --help)" ;;
    esac
done

[[ -n "$MODEL" ]] || die "Missing --model. See --help."
if [[ "$SANDBOX_BACKEND" == "e2b" && -z "${E2B_API_KEY:-}" ]]; then
    die "--sandbox-backend e2b needs E2B_API_KEY (or pass --sandbox-backend local for the free," \
        "non-isolated subprocess backend)."
fi
WORKDIR=${WORKDIR:-"$(mktemp -d /tmp/gdpval.XXXXXX)"}
mkdir -p "$WORKDIR"

if [[ -n "$API_BASE_URL" ]]; then
    [[ "$API_BASE_URL" == */v1 || "$API_BASE_URL" == */v1/ ]] || API_BASE_URL="${API_BASE_URL%/}/v1"
    TARGET_API_KEY="${TARGET_API_KEY:-}"
    if [[ -z "$TARGET_API_KEY" && -f ./scripts/cscs_serving_api_key.txt ]]; then
        TARGET_API_KEY="$(tr -d '\r\n' < ./scripts/cscs_serving_api_key.txt)"
    fi
    export CSCS_SERVING_API="$TARGET_API_KEY"
else
    API_BASE_URL="https://api.openai.com/v1"
fi

if [[ "$NO_GRADE" = 1 ]]; then
    unset GDPVAL_JUDGE_OPENAI_API_KEY GDPVAL_JUDGE_GOOGLE_API_KEY GDPVAL_JUDGE_ANTHROPIC_API_KEY
fi

python3 -c "import stirrup.tools.code_backends.e2b" >/dev/null 2>&1 || \
    pip install --no-cache-dir -q "stirrup[e2b]" datasets huggingface_hub openai anthropic "google-genai" 1>&2

EVAL_MODEL="$MODEL" EVAL_API_BASE_URL="$API_BASE_URL" EVAL_WORKDIR="$WORKDIR" \
GDPVAL_NUM_TASKS="$NUM_TASKS" GDPVAL_TASK_IDS="$TASK_IDS" GDPVAL_MAX_TURNS="$MAX_TURNS" \
GDPVAL_MAX_CONCURRENCY="$MAX_CONCURRENCY" GDPVAL_SANDBOX_BACKEND="$SANDBOX_BACKEND" \
GDPVAL_E2B_TEMPLATE="$E2B_TEMPLATE" \
python3 - <<'PY'
import asyncio
import json
import os
import random
import shutil
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from stirrup import Agent
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.tools import ViewImageToolProvider, WebToolProvider
from stirrup.tools.code_backends.e2b import E2BCodeExecToolProvider
from stirrup.tools.code_backends.local import LocalCodeExecToolProvider

WORKDIR = Path(os.environ.get("EVAL_WORKDIR", "/tmp/gdpval"))
OUT_DIR = WORKDIR / "gdpval_runs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".py", ".yaml", ".yml"}
MAX_JUDGE_TEXT_CHARS = 20_000


def load_tasks():
    dataset = load_dataset("openai/gdpval", split="train")
    task_ids = (os.environ.get("GDPVAL_TASK_IDS") or "").split()
    if task_ids:
        wanted = set(task_ids)
        return [row for row in dataset if row["task_id"] in wanted]
    num_tasks = int(os.environ.get("GDPVAL_NUM_TASKS", "220"))
    return list(dataset)[:num_tasks]


def download_reference_deliverable(row):
    return [
        Path(hf_hub_download("openai/gdpval", name, repo_type="dataset", local_dir=str(OUT_DIR / "reference_cache")))
        for name in row.get("deliverable_files") or []
    ]


def download_reference_files(row):
    return [
        Path(hf_hub_download("openai/gdpval", name, repo_type="dataset", local_dir=str(OUT_DIR / "input_cache")))
        for name in row.get("reference_files") or []
    ]


def build_model_client():
    return ChatCompletionsClient(
        model=os.environ["EVAL_MODEL"],
        base_url=os.environ["EVAL_API_BASE_URL"],
        api_key=os.environ.get("CSCS_SERVING_API") or os.environ.get("OPENAI_API_KEY") or "unused",
        max_tokens=8_192,
        context_window_tokens=128_000,
    )


def build_code_exec_provider(sandbox_backend, e2b_template):
    if sandbox_backend == "e2b":
        return E2BCodeExecToolProvider(template=e2b_template)
    return LocalCodeExecToolProvider()


async def run_one_task(row, semaphore, max_turns, sandbox_backend, e2b_template):
    task_id = row["task_id"]
    task_dir = OUT_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    async with semaphore:
        client = build_model_client()
        code_exec = build_code_exec_provider(sandbox_backend, e2b_template)
        agent = Agent(
            client=client,
            name=f"gdpval-{task_id}",
            tools=[code_exec, WebToolProvider(), ViewImageToolProvider()],
            max_turns=max_turns,
        )
        reference_paths = download_reference_files(row)
        prompt = row["prompt"]
        if reference_paths:
            prompt += "\n\nReference materials (also available in your sandbox):\n"
            prompt += "\n".join(f"- {p.name}" for p in reference_paths)
        error = None
        try:
            async with agent.session(output_dir=str(task_dir)) as session:
                for path in reference_paths:
                    try:
                        shutil.copy(path, task_dir / path.name)
                    except OSError:
                        pass
                await session.run(prompt)
        except Exception as exc:  # noqa: BLE001 -- one task's failure must not sink the run
            error = f"{type(exc).__name__}: {exc}"
        deliverable_files = [
            p for p in task_dir.rglob("*")
            if p.is_file() and p.name not in {p2.name for p2 in reference_paths}
        ]
        return {"row": row, "task_dir": task_dir, "deliverable_files": deliverable_files, "error": error}


def _readable_text(paths):
    parts = []
    for p in paths:
        if p.suffix.lower() in TEXT_EXTENSIONS:
            try:
                text = p.read_text(errors="replace")[:MAX_JUDGE_TEXT_CHARS]
                parts.append(f"--- {p.name} ---\n{text}")
            except OSError:
                pass
        else:
            size = p.stat().st_size if p.exists() else 0
            parts.append(f"--- {p.name} (binary, {size} bytes, not extracted) ---")
    return "\n\n".join(parts) if parts else "(no deliverable files produced)"


_JUDGE_PROMPT = """You are comparing two deliverables produced for the same real-world \
work task. Judge which one better satisfies the task, considering correctness, \
completeness, and professional quality. Ties are allowed when neither is clearly better.

[TASK]
{prompt}

[DELIVERABLE A]
{a}

[DELIVERABLE B]
{b}

Reply with exactly one line: "WINNER: A", "WINNER: B", or "WINNER: TIE"."""


def _judge_pool():
    pool = []
    for provider, default_model in (
        ("openai", "gpt-5.5"), ("google", "gemini-3.1-pro-preview"), ("anthropic", "claude-opus-4-8")
    ):
        model = os.environ.get(f"GDPVAL_JUDGE_{provider.upper()}_MODEL", default_model)
        key = os.environ.get(f"GDPVAL_JUDGE_{provider.upper()}_API_KEY")
        if key:
            pool.append((provider, model, key))
    return pool


async def _call_judge(provider, model, key, prompt):
    if provider == "openai":
        from openai import AsyncOpenAI

        resp = await AsyncOpenAI(api_key=key).chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content or ""
    if provider == "google":
        from google import genai

        resp = await genai.Client(api_key=key).aio.models.generate_content(model=model, contents=prompt)
        return resp.text or ""
    if provider == "anthropic":
        from anthropic import AsyncAnthropic

        resp = await AsyncAnthropic(api_key=key).messages.create(
            model=model, max_tokens=1024, messages=[{"role": "user", "content": prompt}]
        )
        return "".join(b.text for b in resp.content if hasattr(b, "text"))
    raise ValueError(provider)


async def grade_one_task(result, pool):
    if result["error"] or not pool:
        return None
    row = result["row"]
    reference_paths = download_reference_deliverable(row)
    candidate_text = _readable_text(result["deliverable_files"])
    reference_text = _readable_text(reference_paths)
    candidate_is_a = random.random() < 0.5
    a = candidate_text if candidate_is_a else reference_text
    b = reference_text if candidate_is_a else candidate_text
    provider, model, key = random.choice(pool)
    prompt = _JUDGE_PROMPT.format(prompt=row["prompt"][:4000], a=a, b=b)
    try:
        verdict = await _call_judge(provider, model, key, prompt)
    except Exception as exc:  # noqa: BLE001 -- a judge failure must not sink the run
        return {"task_id": row["task_id"], "judge": f"{provider}/{model}", "error": str(exc)}
    verdict_upper = verdict.upper()
    if "WINNER: A" in verdict_upper:
        outcome = "win" if candidate_is_a else "loss"
    elif "WINNER: B" in verdict_upper:
        outcome = "loss" if candidate_is_a else "win"
    elif "WINNER: TIE" in verdict_upper:
        outcome = "tie"
    else:
        outcome = "unparsed"
    return {"task_id": row["task_id"], "judge": f"{provider}/{model}", "outcome": outcome}


async def main():
    rows = load_tasks()
    max_turns = int(os.environ.get("GDPVAL_MAX_TURNS", "250"))
    sandbox_backend = os.environ.get("GDPVAL_SANDBOX_BACKEND", "e2b")
    e2b_template = os.environ.get("GDPVAL_E2B_TEMPLATE", "code-interpreter-v1")
    concurrency = int(os.environ.get("GDPVAL_MAX_CONCURRENCY", "4"))
    semaphore = asyncio.Semaphore(concurrency)

    results = await asyncio.gather(
        *[run_one_task(row, semaphore, max_turns, sandbox_backend, e2b_template) for row in rows]
    )
    pool = _judge_pool()
    grades = await asyncio.gather(*[grade_one_task(r, pool) for r in results])

    generation_errors = [r["row"]["task_id"] for r in results if r["error"]]
    graded = [g for g in grades if g and "outcome" in g]
    wins = sum(1 for g in graded if g["outcome"] == "win")
    ties = sum(1 for g in graded if g["outcome"] == "tie")
    losses = sum(1 for g in graded if g["outcome"] == "loss")
    unparsed = sum(1 for g in graded if g["outcome"] == "unparsed")
    n_graded = len(graded)

    metrics = {"n_tasks": len(rows), "n_generation_errors": len(generation_errors), "n_graded": n_graded}
    if n_graded:
        metrics["win_rate"] = wins / n_graded
        metrics["win_or_tie_rate"] = (wins + ties) / n_graded
        metrics["ties"] = ties
        metrics["losses"] = losses
        metrics["unparsed"] = unparsed
    results_payload = {"gdpval-aa-v2": metrics} if pool else None

    grade_by_task = {g["task_id"]: g for g in grades if g}
    samples = [
        {
            "task": "gdpval-aa-v2",
            "doc_id": i,
            "sample": {
                "task_id": r["row"]["task_id"],
                "occupation": r["row"].get("occupation"),
                "sector": r["row"].get("sector"),
                "generation_error": r["error"],
                "n_deliverable_files": len(r["deliverable_files"]),
                "grade": grade_by_task.get(r["row"]["task_id"]),
            },
        }
        for i, r in enumerate(results)
    ]

    # Same "succeeded whenever there is anything to show" rule as evals-svc's own results
    # parsing: real graded results, or at least one deliverable generated even if ungraded.
    any_deliverable = len(generation_errors) < len(rows)
    status = "succeeded" if results_payload is not None or any_deliverable else "failed"
    payload = {"status": status, "results": results_payload, "samples": samples}
    if status == "failed":
        payload["error"] = f"every one of {len(rows)} tasks failed to generate: " + ", ".join(
            generation_errors[:20]
        )
    elif not pool:
        payload["error"] = (
            "no judge credentials configured -- deliverables were generated "
            f"({len(rows) - len(generation_errors)}/{len(rows)} succeeded) but never graded"
        )
    elif generation_errors:
        payload["error"] = f"{len(generation_errors)}/{len(rows)} tasks failed to generate (results kept): " + ", ".join(
            generation_errors[:20]
        )

    result_path = WORKDIR / "gdpval_result.json"
    result_path.write_text(json.dumps(payload, indent=2))
    print(f"result: {result_path}")
    print(f"generated: {len(rows) - len(generation_errors)}/{len(rows)} deliverables")
    if pool and n_graded:
        print(json.dumps(metrics, indent=2))
    elif not pool:
        print("no judge credentials configured (GDPVAL_JUDGE_*_MODEL/_API_KEY) -- deliverables written, not graded")


asyncio.run(main())
PY
echo "deliverables: $WORKDIR/gdpval_runs/"
