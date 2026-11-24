"""Run BFCL (the Berkeley Function Call Leaderboard) with the official bfcl-eval package: its own generation
(the model's requests, the multi-turn loop with its simulated tool environments), its own checkers and its own
score aggregation. No other framework in between. Run by scripts/custom/run_bfcl.sh in a venv of
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
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# BFCL v3: the non-live, live and multi-turn categories. BFCL v4 adds the memory
# categories; its web search ones need a SerpAPI key and are not run (--test-category adds them).
V3_CATEGORIES = [
    "simple_python", "simple_java", "simple_javascript", "multiple", "parallel", "parallel_multiple", "irrelevance",
    "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple", "live_irrelevance", "live_relevance",
    "multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context",
]
CATEGORIES = {"v3": V3_CATEGORIES, "v4": V3_CATEGORIES + ["memory_kv", "memory_vector", "memory_rec_sum"]}

# The categories of each of bfcl-eval's groups; a group is only reported when all were run.
NEEDED = {
    "non_live": ["simple_python", "simple_java", "simple_javascript", "multiple", "parallel", "parallel_multiple", "irrelevance"],
    "live": ["live_simple", "live_multiple", "live_parallel", "live_parallel_multiple", "live_irrelevance", "live_relevance"],
    "multi_turn": ["multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context"],
    "agentic": ["memory_kv", "memory_vector", "memory_rec_sum", "web_search_base", "web_search_no_snippet"],
}

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
    """The registry key. bfcl-eval names the result directory after it with "/" as "_", and its evaluator maps
    that name back with every "_" as "/": a key with either would come back as another one (the official
    Qwen/Qwen3-8B-FC, with its own handler, for a served Qwen/Qwen3-8B). So letters, digits, "." and "-" only,
    and a prefix that no official key has."""
    return "custom-" + re.sub(r"[^A-Za-z0-9.-]", "-", model) + "-FC"


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


def select_ids(entry_ids: list[str], n: int) -> list[str]:
    """The first n questions of a category, and for memory categories the prerequisite conversations of their
    scenarios: those are the turns that write the memory the questions then ask about, and without them the
    model is asked about facts it was never told (every memory question scores 0)."""
    questions = [i for i in entry_ids if "_prereq_" not in i][:n]
    scenario = lambda i: i.split("-")[1]  # "memory_kv_0-customer-0" -> "customer"
    scenarios = {scenario(i) for i in questions if i.startswith("memory_")}
    return [i for i in entry_ids if "_prereq_" in i and scenario(i) in scenarios] + questions


def first_entries(categories: list[str], n: int, workdir: Path) -> None:
    """Write bfcl-eval's test_case_ids_to_generate.json: the first n entries of each category."""
    from bfcl_eval.utils import load_dataset_entry

    ids = {c: select_ids([e["id"] for e in load_dataset_entry(c)], n) for c in categories}
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


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def v4_without_web_search(categories: dict[str, dict[str, Any]]) -> dict[str, float] | None:
    """BFCL v4's overall with the web search categories left out, from the per-category accuracies; None unless
    every category it needs was run. bfcl-eval's own formula (eval_runner_helper.py, 2026.3.23) with one change:
    its agentic group is the mean of the web search and the memory scores, here the memory score alone. The rest
    is the package's: non-live is the mean of simple (the mean of Python, Java, JavaScript), multiple, parallel and
    parallel multiple; live weights its four categories by their size; irrelevance is the mean of the non-live and
    the live irrelevance scores; multi-turn and memory are plain means; the weights are 10/10/10/30/40 over
    non-live, live, irrelevance, multi-turn and agentic. Not the leaderboard's number: bfcl-eval counts a missing
    web search score as 0, which would put its overall far below this one."""
    names = [c for group in NEEDED.values() for c in group if c not in ("web_search_base", "web_search_no_snippet")]
    if not all(name in categories for name in names):
        return None
    accuracy = {name: categories[name]["accuracy"] for name in names}
    live = [categories[c] for c in NEEDED["live"] if c not in ("live_irrelevance", "live_relevance")]
    scores = {
        "non_live": mean([
            mean([accuracy["simple_python"], accuracy["simple_java"], accuracy["simple_javascript"]]),
            accuracy["multiple"], accuracy["parallel"], accuracy["parallel_multiple"],
        ]),
        "live": sum(c["accuracy"] * c["total"] for c in live) / sum(c["total"] for c in live),
        "irrelevance": mean([accuracy["irrelevance"], accuracy["live_irrelevance"]]),
        "multi_turn": mean([accuracy[c] for c in NEEDED["multi_turn"]]),
        "memory": mean([accuracy[c] for c in ("memory_kv", "memory_vector", "memory_rec_sum")]),
    }
    scores["overall_no_web_search"] = (
        10 * scores["non_live"] + 10 * scores["live"] + 10 * scores["irrelevance"]
        + 30 * scores["multi_turn"] + 40 * scores["memory"]
    ) / 100
    return scores


INFERENCE_ERROR_MARKERS = ("Error during inference", "Error occurred during inference")


def inference_errors(result_dir: Path, registry: str) -> dict[str, Any]:
    """How many generated entries ended in an inference error, by cause.

    bfcl-eval writes "Error during inference: ..." as the result of an entry whose generation raised, and
    scores it wrong like any other wrong answer (it stays in the category's total). A context overflow
    (the conversation outgrew the served window) is a limit of the model under test; any other error (a timeout,
    a server error) is not necessarily the model's, so those are reported apart: rerun the run if there are any."""
    entries = errors = 0
    by_cause = {"context_overflow": 0, "other": 0}
    for path in sorted(result_dir.glob(f"{registry}/**/*_result.json")):
        for line in path.read_text().splitlines():
            try:
                result = json.loads(line).get("result", "")
            except (json.JSONDecodeError, AttributeError):
                continue
            entries += 1
            text = json.dumps(result)[:1000]
            if any(marker in text for marker in INFERENCE_ERROR_MARKERS):
                errors += 1
                by_cause["context_overflow" if "context length" in text or "context_length" in text else "other"] += 1
    return {"n_entries": entries, "n_errors": errors, "by_cause": by_cause}


def summarize(version: str, registry: str, score_dir: Path) -> dict[str, Any]:
    """Per-category scores and the official groups; the overall BFCL v3 computed (bfcl-eval 3.0.0's
    total: the unweighted mean of the live, non-live and multi-turn groups). For v4 there is no official overall
    here (the agentic group needs web search); `v4_without_web_search` holds the scores of v4_without_web_search().

    bfcl-eval counts a category it was not asked to run as 0 in its group tables, so a group is only
    reported when all its categories were run: the agentic group needs both web search categories (memory alone
    is in the per-category scores)."""
    categories = {}
    for path in sorted(score_dir.glob(f"{registry}/**/*_score.json")):  # the memory ones sit one level deeper
        score = read_score(path)
        name = path.name.removesuffix("_score.json")
        categories[re.sub(r"^BFCL_v\d+_", "", name)] = {
            "accuracy": score["accuracy"], "correct": score["correct_count"], "total": score["total_count"],
        }
    groups = {g: v for g, v in group_scores(score_dir).items() if all(c in categories for c in NEEDED[g])}
    overall = None
    if version == "v3" and all(groups.get(g) is not None for g in ("non_live", "live", "multi_turn")):
        overall = sum(groups[g] for g in ("non_live", "live", "multi_turn")) / 3
    summary = {"version": version, "categories": categories, "groups": groups, "overall": overall}
    if version == "v4" and (no_web := v4_without_web_search(categories)):
        summary["v4_without_web_search"] = no_web
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
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
    # A thinking model's long answers outlast the SDK's 600 s; about 3 samples per run were lost to it.
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
    summary["inference_errors"] = inference_errors(workdir / "result", registry)
    summary.update(model=args.model, api_base_url=args.api_base_url, num_tasks=args.num_tasks, duration_s=int(time.time() - started))
    (workdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    errors = summary["inference_errors"]
    print(f"inference errors counted as wrong answers: {errors['n_errors']} of {errors['n_entries']} entries "
          f"(context overflow {errors['by_cause']['context_overflow']}, other {errors['by_cause']['other']})")
    if errors["by_cause"]["other"]:
        print("WARNING: errors other than a context overflow (a timeout, a server error) are not necessarily the "
              "model's fault; consider rerunning.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
