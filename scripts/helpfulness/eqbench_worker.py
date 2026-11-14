"""Private subprocess entrypoint for native EQ rubrics with a restricted API bridge."""

import importlib
import math
import os
from pathlib import Path
import runpy
import sys
import types

from scripts.helpfulness.core import ChatClient, append_jsonl, endpoint_model, read_json, write_json
from scripts.helpfulness.open_models import provenance


def bridge(target: dict, judge: dict, trace: Path, name: str = "eqbench3"):
    configs = {"test": target, "judge": judge}
    clients = {key: ChatClient(config) for key, config in configs.items()}

    class APIClient:
        def __init__(self, model_type=None, **kwargs):
            if model_type not in configs:
                raise ValueError("EQ-Bench must explicitly select the test or judge connection")
            self.role = model_type

        def request(self, model, messages, temperature, max_tokens, min_p):
            if model != endpoint_model(configs[self.role]):
                raise ValueError("Native EQ-Bench attempted an undeclared model")
            decoding = {"temperature": temperature, "max_tokens": max_tokens}
            if min_p is not None:
                decoding["min_p"] = min_p
            row = {"role": self.role, "model": model, "messages": messages, "decoding": decoding}
            try:
                response = clients[self.role].generate({"messages": messages}, decoding)
            except (ValueError, RuntimeError) as error:
                append_jsonl(trace, {**row, "error": str(error)})
                raise
            append_jsonl(trace, {**row, "response": response})
            return response

    if name == "eqbench_creative_v3":
        class CreativeAPIClient(APIClient):
            def generate(self, model, prompt, temperature=0.0, max_tokens=8096,
                         include_seed=True, min_p=0.1, system=None):
                if not isinstance(prompt, str):
                    raise ValueError("Creative Writing v3 requires its native string prompt")
                messages = ([{"role": "system", "content": system}] if system else []) + [
                    {"role": "user", "content": prompt}
                ]
                # The pinned native client's random judging-seed block is disabled.
                return self.request(model, messages, temperature, max_tokens, min_p)

        return CreativeAPIClient

    class ConversationAPIClient(APIClient):
        def generate(self, model, messages, temperature=0.7, max_tokens=4000, min_p=0.1):
            return self.request(
                model, messages, temperature, max_tokens, min_p if self.role == "test" else None,
            )

    return ConversationAPIClient


def valid_native_cases(name: str, runs: dict, native, expected_ids: set[str],
                       negative_criteria: list[str] | None = None) -> list[str]:
    valid = []
    key = "scenario_tasks" if name == "eqbench3" else "creative_tasks"
    if len(runs) != 1:
        raise ValueError("Native output must contain exactly one isolated run")
    for iteration, tasks in next(iter(runs.values())).get(key, {}).items():
        for case_id, task in tasks.items():
            identity = f"{iteration}/{case_id}"
            if identity not in expected_ids:
                raise ValueError("Native output contains an unplanned scenario/iteration")
            if name == "eqbench3":
                raw = [task.get("rubric_scores", {}) or {}]
                score, _ = native.calculate_final_rubric_score({key: {iteration: {case_id: task}}})
            else:
                raw = [block.get("judge_scores", {}) for block in task.get("results_by_modifier", {}).values()]
                if negative_criteria is None:
                    raise ValueError("Validate creative scores with the pinned native negative criteria")
                score = native.compute_single_benchmark_score_creative(
                    [task], negative_criteria,
                )["overall_score"] if raw else None
            if raw and all(scores and all(
                type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 20
                for value in scores.values()
            ) for scores in raw) and type(score) in (float, int) and math.isfinite(score) and 0 <= score <= 20:
                valid.append(identity)
    return valid


def main(path: Path) -> None:
    config = read_json(path)
    provenance(config["target"], {"chat", "min_p"})
    provenance(config["judge"], {"chat", "min_p"} if config["name"] == "eqbench_creative_v3" else {"chat"})
    working = Path(config["working"])
    os.chdir(working)
    sys.path.insert(0, str(working))
    if config.get("nltk_directory"):
        import nltk

        nltk.data.path[:] = [config["nltk_directory"]]
        for resource in ("corpora/cmudict", "tokenizers/punkt", "tokenizers/punkt_tab/english"):
            nltk.data.find(resource)
    module = types.ModuleType("utils.api")
    module.APIClient = bridge(config["target"], config["judge"], working / "requests.jsonl", config["name"])
    sys.modules["utils.api"] = module
    name, data = config["name"], config["data"]
    if name == "eqbench3":
        native = importlib.import_module("core.benchmark")
        case_ids = native.parse_scenario_prompts(native.C.STANDARD_SCENARIO_PROMPTS_FILE).keys()
        negative_criteria = None
    else:
        case_ids = read_json(working / "data" / "creative_writing_prompts_v3.json").keys()
        negative_criteria = [
            line.strip() for line in (working / "data" / "negative_criteria.txt").read_text(
                encoding="utf-8",
            ).splitlines() if line.strip()
        ]
    expected_ids = {f"{iteration}/{case_id}" for iteration in range(1, data["iterations"] + 1) for case_id in case_ids}
    write_json(working / "coverage.json", {"expected_cases": len(expected_ids)})
    for filename in ("empty-leaderboard.json", "native-runs.json"):
        if not (working / filename).exists():
            write_json(working / filename, {})
    sys.argv = [
        data["entrypoint"], "--test-model", endpoint_model(config["target"]),
        "--judge-model", endpoint_model(config["judge"]), "--threads", "1",
        "--iterations", str(data["iterations"]), "--no-elo", "--run-id", "teacher-rubric",
        "--runs-file", str(working / "native-runs.json"),
    ]
    if name == "eqbench3":
        sys.argv += ["--leaderboard-runs-file", str(working / "empty-leaderboard.json"),
                     "--leaderboard-elo-file", str(working / "empty-leaderboard.json")]
    try:
        runpy.run_path(str(working / data["entrypoint"]), run_name="__main__")
    except SystemExit as error:
        if error.code not in (None, 0):
            raise
    native = importlib.import_module("core.benchmark" if name == "eqbench3" else "core.scoring")
    valid = valid_native_cases(
        name, read_json(working / "native-runs.json"), native, expected_ids, negative_criteria,
    )
    write_json(working / "coverage.json", {"expected_cases": len(expected_ids), "valid_case_ids": valid})


if __name__ == "__main__":
    main(Path(sys.argv[1]))
