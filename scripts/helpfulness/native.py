"""Integration with the repository's pinned Swiss AI lm-eval harness."""

import hashlib
import importlib.metadata
import json
import os
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from scripts.helpfulness.adapters import NATIVE_TASKS
from scripts.helpfulness.core import (
    digest, endpoint_key, endpoint_model, endpoint_url, number, write_json,
)

NLTK_PACKAGES = ("punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger_eng")


def asset_hashes(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*")) if path.is_file()
    }


def prepare_assets(directory: Path) -> dict:
    import nltk

    directory.mkdir(parents=True, mode=0o700)
    for package in NLTK_PACKAGES:
        if not nltk.download(package, download_dir=str(directory), quiet=True):
            raise RuntimeError(
                f"Could not stage NLTK {package}; use an owner-private Linux bundle directory"
            )
    files = asset_hashes(directory)
    return {"files": files, "digest": digest(files)}


def prepare_alpaca_assets(directory: Path) -> dict:
    from alpaca_eval import constants
    from huggingface_hub import hf_hub_download

    source = Path(hf_hub_download(
        repo_id="tatsu-lab/alpaca_eval", filename="df_gamed.csv", repo_type="dataset",
        token=constants.DATASETS_TOKEN, cache_dir=constants.DEFAULT_CACHE_DIR,
    ))
    directory.mkdir(parents=True, mode=0o700)
    (directory / "df_gamed.csv").write_bytes(source.read_bytes())
    files = asset_hashes(directory)
    return {"files": files, "digest": digest(files), "revision": source.parent.name}


@contextmanager
def offline_alpaca_assets(directory: Path | None):
    if directory is None:
        yield
        return
    from alpaca_eval.metrics import glm_winrate

    def staged_download(*args, **kwargs):
        if (
            not args and kwargs.get("repo_id") == "tatsu-lab/alpaca_eval"
            and kwargs.get("filename") == "df_gamed.csv"
            and kwargs.get("repo_type") == "dataset"
        ):
            return str(directory / "df_gamed.csv")
        raise ValueError("AlpacaEval requested an unstaged scorer asset")

    with patch.object(glm_winrate, "hf_hub_download", staged_download):
        yield


@contextmanager
def offline_scorers(directory: Path | None):
    if directory is None:
        yield
        return
    with environment({"NLTK_DATA": str(directory)}):
        import nltk

        paths, download = list(nltk.data.path), nltk.download
        nltk.data.path.insert(0, str(directory))

        def refuse_download(*args, **kwargs):
            raise ValueError("Missing NLTK scorer assets; run prepare instead of downloading during evaluation")

        nltk.download = refuse_download
        try:
            yield
        finally:
            nltk.data.path[:] = paths
            nltk.download = download


def harness_pin() -> str:
    path = Path(__file__).resolve().parents[2] / "requirements" / "lm-eval-harness.txt"
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts and parts[0] == "swiss-ai/lm-evaluation-harness":
            return parts[1]
    raise ValueError("The repository has no Swiss AI harness pin")


def require_harness() -> None:
    root = os.environ.get("EVAL_PREBUILT_HARNESS_ROOT")
    if root:
        overlay = Path(root) / "swiss-ai" / "lm-evaluation-harness"
        if (overlay / ".commit").read_text(encoding="utf-8").strip() != harness_pin():
            raise ValueError("Prebuilt harness overlay does not match the repository pin")
        if str(overlay) not in sys.path:
            sys.path.insert(0, str(overlay))
    distribution = importlib.metadata.distribution("lm_eval")
    reference = json.loads(distribution.read_text("direct_url.json") or "{}")
    commit = reference.get("vcs_info", {}).get("commit_id")
    if commit != harness_pin():
        raise ValueError(
            "Install the exact git-pinned Swiss AI harness from "
            "requirements/lm-eval-harness.txt; an unverified editable/PyPI install is not accepted"
        )


@contextmanager
def environment(values: dict[str, str]):
    previous = {key: os.environ.get(key) for key in values}
    offline_constants = {
        "huggingface_hub.constants": "HF_HUB_OFFLINE",
        "datasets.config": "HF_DATASETS_OFFLINE",
    }
    saved_constants = {}
    for module_name, key in offline_constants.items():
        if key in values:
            module = sys.modules.get(module_name)
            saved_constants[module_name] = (
                getattr(module, key) if module is not None
                else (previous[key] or "").upper() in {"1", "ON", "YES", "TRUE"}
            )
            if module is not None:
                setattr(module, key, values[key].upper() in {"1", "ON", "YES", "TRUE"})
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        for module_name, value in saved_constants.items():
            module = sys.modules.get(module_name)
            if module is not None:
                setattr(module, offline_constants[module_name], value)


def load_tasks(name: str) -> dict:
    require_harness()
    from lm_eval.tasks import TaskManager

    return TaskManager().load(list(NATIVE_TASKS[name]))["tasks"]


def prepare_native(name: str) -> dict[str, list[dict]]:
    tasks = load_tasks(name)
    return {
        key: [dict(doc) for doc in (task.test_docs() if task.has_test_docs() else task.validation_docs())]
        for key, task in tasks.items()
    }


def validate_ifbench_cases(cases: list[dict]) -> None:
    from lm_eval.tasks.ifbench.instructions_registry import INSTRUCTION_DICT

    for case in cases:
        unknown = set(case["instruction_id_list"]) - set(INSTRUCTION_DICT)
        if unknown:
            raise ValueError(f"Unsupported held-out IFBench verifier IDs: {sorted(unknown)}")


def alpaca_aggregation(
    items: list[dict], judge: dict, output: Path, *, length_controlled: bool = True,
) -> float:
    from alpaca_eval import constants, evaluate, utils
    from lm_eval.tasks.alpaca_eval.metric import (
        _annotator_config, _rate_limited_openai_resources,
    )

    model = endpoint_model(judge)
    config = _annotator_config(model)
    annotator = next(iter(config))
    config[annotator]["prompt_template"] = str(
        constants.EVALUATORS_CONFIG_DIR / config[annotator]["prompt_template"]
    )
    client_path = output / "annotator" / "client-config.yaml"
    write_json(client_path, {"default": [{"base_url": endpoint_url(judge)}]})
    config[annotator]["completions_kwargs"].update({
        "client_config_path": str(client_path), "num_procs": 1,
    })
    candidates = [{
        "instruction": item["instruction"], "output": item["completion"],
        "generator": "candidate", "dataset": item["dataset"],
    } for item in items]
    references = [{
        "instruction": item["instruction"], "output": item["reference_output"],
        "generator": "gpt4", "dataset": item["dataset"],
    } for item in items]
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "annotator" / "configs.yaml"
    # JSON is valid YAML; keep this config local rather than modifying site-packages.
    write_json(config_path, config)
    with environment({
        "OPENAI_API_KEY": endpoint_key(judge),
        "OPENAI_BASE_URL": endpoint_url(judge),
        "OPENAI_API_BASE": endpoint_url(judge),
    }), _rate_limited_openai_resources(model):
        leaderboard, annotations = evaluate(
            model_outputs=candidates, reference_outputs=references,
            annotators_config=str(config_path), is_return_instead_of_print=True,
            fn_metric="get_length_controlled_winrate" if length_controlled else "get_winrate",
            sort_by="length_controlled_winrate" if length_controlled else "win_rate",
            output_path=str(output),
            metric_kwargs={"save_weights_dir": str(output / "glm_weights")} if length_controlled else {},
            precomputed_leaderboard=None, is_cache_leaderboard=False,
            caching_path=str(output / "cache.json"),
            annotation_kwargs={
                "n_retries": 3,
                "client_kwargs": {
                    "max_retries": 3, "timeout": judge.get("timeout_seconds", 180),
                },
            },
        )
    # Pandas emits unavailable metadata (e.g. custom-judge pricing) as null.
    # Required preference and leaderboard metrics are still checked below.
    annotations = json.loads(utils.convert_to_dataframe(annotations).to_json(orient="records", double_precision=15))
    if Counter(row["instruction"] for row in annotations) != Counter(
        row["instruction"] for row in candidates
    ):
        raise ValueError("AlpacaEval did not judge every selected instruction")
    for row in annotations:
        number(row.get("preference"), "AlpacaEval preference", 1, 2)
    raw = number(leaderboard["win_rate"].iloc[0], "AlpacaEval raw win rate", 0, 100) / 100
    controlled = number(
        leaderboard["length_controlled_winrate"].iloc[0], "AlpacaEval LC win rate", 0, 100
    ) / 100 if length_controlled else None
    write_json(output / "validated_metrics.json", {
        "raw_win_rate": raw, "length_controlled_winrate": controlled,
        "judged_cases": len(annotations),
    })
    write_json(output / "validated_annotations.json", annotations)
    return controlled if length_controlled else raw


def run_native(name: str, documents: dict[str, list[dict]], target: dict, judge: dict, output: Path) -> dict:
    from datasets import Dataset
    from lm_eval import evaluator
    from lm_eval.api.rate_limiter import JUDGE_REQUESTS_PER_MINUTE_ENV

    tasks = load_tasks(name)
    for key, task in tasks.items():
        split = task.get_config("test_split") or task.get_config("validation_split")
        task.dataset = {split: Dataset.from_list(documents[key])}
    if name == "alpaca_eval_2_lc":
        task = tasks["alpaca_eval"]
        # Replace only the aggregation hook; retain the pinned task's generation
        # and per-sample processing, but keep annotations instead of deleting them.
        original_aggregation = task.aggregation
        task.aggregation = lambda: {
            **original_aggregation(),
            "length_controlled_winrate": lambda items: alpaca_aggregation(items, judge, output / "alpaca"),
        }
    env = {
        "OPENAI_API_KEY": endpoint_key(target),
        "CSCS_SERVING_API": endpoint_key(judge) if judge else "",
        "ALPACA_EVAL_JUDGE_MODEL": endpoint_model(judge) if judge else "",
        "MT_BENCH_JUDGE_MODEL": endpoint_model(judge) if judge else "",
        "MT_BENCH_JUDGE_API_KEY": endpoint_key(judge) if judge else "",
        "MT_BENCH_JUDGE_API_BASE": endpoint_url(judge) if judge else "",
        "MT_BENCH_JUDGE_MAX_WORKERS": "1",
        "MT_BENCH_JUDGE_LOG_PATH": str(output / "mt_bench_judgments.jsonl"),
        "IS_ALPACA_EVAL_2": "True",
        JUDGE_REQUESTS_PER_MINUTE_ENV: str(judge.get("requests_per_minute", 14)) if judge.get("requests_per_minute", 14) else "",
    }
    likelihood = name == "sycophancy"
    args = {
        "model": endpoint_model(target),
        "base_url": endpoint_url(target) + ("/completions" if likelihood else "/chat/completions"),
        "num_concurrent": 1,
        "max_retries": 3,
        "requests_per_minute": target.get("requests_per_minute", 14) or None,
        "timeout": target.get("timeout_seconds", 180),
    }
    if likelihood:
        args["tokenizer"] = target["tokenizer"]
        args["tokenizer_backend"] = "huggingface"
    else:
        args["tokenizer_backend"] = None
    with environment(env):
        result = evaluator.simple_evaluate(
            model="local-completions" if likelihood else "local-chat-completions",
            model_args=args, tasks=list(tasks.values()), batch_size=1,
            log_samples=True, apply_chat_template=not likelihood,
            bootstrap_iters=0, random_seed=1234, numpy_random_seed=1234,
            torch_random_seed=1234, fewshot_random_seed=1234,
        )
    if result is None:
        raise ValueError("The native harness returned no result")
    from lm_eval.utils import handle_non_serializable

    return json.loads(json.dumps(result, default=handle_non_serializable))
