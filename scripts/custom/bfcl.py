"""Run BFCL (the Berkeley Function Call Leaderboard) with the official bfcl-eval package: its own generation
(the model's requests, the multi-turn loop with its simulated tool environments), its own checkers and its own
score aggregation. Nothing of EvalScope in between. Run by scripts/custom/run_bfcl.sh in a venv of
requirements/bfcl-runtime.lock; see docs/custom-bfcl.md.

The model is served by any OpenAI-compatible endpoint. It is registered at run time under "<name>-FC" (bfcl-eval
only knows the models of its own registry) with the package's OpenAI chat-completions handler, in function-calling
mode. The two patches below keep that handler from dropping the sampling and thinking settings and from replaying
null fields to vLLM; everything else is the package's.

    python -m scripts.custom.bfcl --version v3 --model NAME --api-base-url URL --workdir DIR

BFCL_PROJECT_ROOT is the workdir: results go to <workdir>/result, scores to <workdir>/score.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# What EvalScope's BFCL v3 ran: the non-live, live and multi-turn categories. BFCL v4 adds the memory
# categories; its web search ones need a SerpAPI key and are not run (--test-category adds them).
V3_CATEGORIES = [
    "simple_python", "simple_java", "simple_javascript", "multiple", "parallel", "parallel_multiple", "irrelevance",
    "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple", "live_irrelevance", "live_relevance",
    "multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context",
]
CATEGORIES = {"v3": V3_CATEGORIES, "v4": V3_CATEGORIES + ["memory_kv", "memory_vector", "memory_rec_sum"]}

# bfcl-eval's own default for API models.
DEFAULT_TEMPERATURE = 0.001


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", required=True, choices=sorted(CATEGORIES))
    p.add_argument("--model", required=True, help="The served model name.")
    p.add_argument("--api-base-url", help="OpenAI-compatible endpoint (its /v1 root); else OPENAI_BASE_URL.")
    p.add_argument("--workdir", required=True, type=Path)
    p.add_argument("--llm-args", default="{}", help="JSON of sampling and thinking settings (litellm style).")
    p.add_argument("--request-timeout", type=float, default=3600, help="Seconds per request (the SDK's own is 600).")
    p.add_argument("--num-threads", type=int, default=32)
    p.add_argument("--num-tasks", type=int, help="Only the first n entries of each category (a smoke test).")
    p.add_argument("--test-category", action="append", help="Replace the version's categories (repeatable).")
    return p.parse_args(argv)


def split_llm_args(llm_args: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """(temperature for the handler, kwargs for every chat.completions.create call)."""
    kwargs = dict(llm_args)
    return kwargs.pop("temperature", DEFAULT_TEMPERATURE), kwargs


def registry_name(model: str) -> str:
    """The registry key: bfcl-eval builds file paths from it, so no slash."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", model) + "-FC"


def clean_message(message: Any) -> dict[str, Any]:
    """An assistant turn (an SDK message object or a dict) as a dict without null fields.

    bfcl-eval stores each reply as it came back and sends it with the next request; a vLLM reply carries explicit
    nulls and its request validation rejects a text-only turn with ``tool_calls: null`` (HTTP 400), which would end
    that multi-turn conversation and lose the sample.
    """
    data = message.model_dump() if hasattr(message, "model_dump") else dict(message)
    cleaned = {key: value for key, value in data.items() if value is not None}
    if not cleaned.get("tool_calls"):
        cleaned.pop("tool_calls", None)
        cleaned.setdefault("content", "")
    return cleaned


def install_patches(request_kwargs: dict[str, Any]) -> None:
    """Add request_kwargs to every request of bfcl-eval's OpenAI handler and keep nulls out of its history."""
    from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler as Handler

    generate = Handler.generate_with_backoff

    def generate_with_settings(self, **kwargs):
        for key, value in request_kwargs.items():
            kwargs.setdefault(key, value)
        return generate(self, **kwargs)

    Handler.generate_with_backoff = generate_with_settings

    def wrap(add_assistant_message):
        def add_clean(self, inference_data, model_response_data):
            key = "model_responses_message_for_chat_history"
            model_response_data = {**model_response_data, key: clean_message(model_response_data[key])}
            return add_assistant_message(self, inference_data, model_response_data)

        return add_clean

    for name in ("_add_assistant_message_FC", "_add_assistant_message_prompting"):
        setattr(Handler, name, wrap(getattr(Handler, name)))


def register(model: str) -> str:
    """Add the served model to bfcl-eval's registry; return its key."""
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
    from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler

    name = registry_name(model)
    MODEL_CONFIG_MAPPING[name] = ModelConfig(
        model_name=model,
        display_name=f"{model} (FC)",
        url="",
        org="",
        license="",
        model_handler=OpenAICompletionsHandler,
        is_fc_model=True,
        underscore_to_dot=True,  # as the package's own OpenAI function-calling entries
    )
    return name


def first_entries(categories: list[str], n: int, workdir: Path) -> None:
    """Write bfcl-eval's test_case_ids_to_generate.json: the first n entries of each category."""
    from bfcl_eval.utils import load_dataset_entry

    ids = {c: [e["id"] for e in load_dataset_entry(c, include_prereq=False)][:n] for c in categories}
    (workdir / "test_case_ids_to_generate.json").write_text(json.dumps(ids))


def read_score(path: Path) -> dict[str, Any]:
    """The header line of a score file: accuracy, correct_count, total_count."""
    with open(path) as f:
        return json.loads(f.readline())


def number(value: str) -> float | None:
    """A CSV cell as a fraction: bfcl-eval writes accuracies as "71.74%" and a missing one as "N/A"."""
    try:
        return float(value.removesuffix("%")) / 100
    except ValueError:
        return None


def group_scores(score_dir: Path) -> dict[str, float | None]:
    """The official group accuracies (0..1) from the CSVs bfcl-eval writes."""
    columns = {
        "data_non_live.csv": ("non_live", "Non-Live Overall Acc"),
        "data_live.csv": ("live", "Live Overall Acc"),
        "data_multi_turn.csv": ("multi_turn", "Multi Turn Overall Acc"),
        "data_agentic.csv": ("agentic", "Agentic Overall Acc"),
    }
    groups: dict[str, float | None] = {}
    for filename, (group, column) in columns.items():
        path = score_dir / filename
        if not path.exists():
            continue
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            groups[group] = number(rows[0][column])
    return groups


def summarize(version: str, registry: str, score_dir: Path) -> dict[str, Any]:
    """Per-category scores and the official groups; the overall BFCL v3 computed (bfcl-eval 3.0.0's
    total: the unweighted mean of the live, non-live and multi-turn groups), and for v4 only when
    bfcl-eval itself has all its groups (the agentic one needs web search)."""
    categories = {}
    for path in sorted(score_dir.glob(f"{registry}/*/*_score.json")):
        score = read_score(path)
        name = path.name.removesuffix("_score.json")
        categories[re.sub(r"^BFCL_v\d+_", "", name)] = {
            "accuracy": score["accuracy"], "correct": score["correct_count"], "total": score["total_count"],
        }
    groups = group_scores(score_dir)
    overall = None
    if version == "v3" and all(groups.get(g) is not None for g in ("non_live", "live", "multi_turn")):
        overall = sum(groups[g] for g in ("non_live", "live", "multi_turn")) / 3
    return {"version": version, "categories": categories, "groups": groups, "overall": overall}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    # Read when bfcl_eval is imported: where results, scores and the ids file live.
    os.environ["BFCL_PROJECT_ROOT"] = str(workdir)
    if args.api_base_url:
        os.environ["OPENAI_BASE_URL"] = args.api_base_url
    os.environ.setdefault("OPENAI_API_KEY", os.environ.get("TARGET_API_KEY") or "none")

    from bfcl_eval._llm_response_generation import main as generate
    from bfcl_eval.eval_checker.eval_runner import main as evaluate

    temperature, request_kwargs = split_llm_args(json.loads(args.llm_args))
    # A thinking model's long answers outlast the SDK's 600 s; EvalScope lost 3 samples per run to it.
    install_patches({"timeout": args.request_timeout, **request_kwargs})
    registry = register(args.model)
    categories = args.test_category or CATEGORIES[args.version]
    if args.num_tasks:
        first_entries(categories, args.num_tasks, workdir)

    generate(SimpleNamespace(
        model=[registry], test_category=categories, temperature=temperature, include_input_log=False,
        exclude_state_log=False, num_gpus=1, num_threads=args.num_threads, gpu_memory_utilization=0.9,
        backend="sglang", skip_server_setup=True, local_model_path=None, result_dir=None, allow_overwrite=False,
        run_ids=bool(args.num_tasks), enable_lora=False, max_lora_rank=None, lora_modules=None,
    ))
    evaluate([registry], categories, None, None, partial_eval=bool(args.num_tasks))

    summary = summarize(args.version, registry, workdir / "score")
    summary.update(model=args.model, api_base_url=args.api_base_url, num_tasks=args.num_tasks)
    (workdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
