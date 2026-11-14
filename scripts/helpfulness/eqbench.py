"""Pinned native EQ-Bench rubric runs, with no inherited leaderboard/Elo state."""

import os
from pathlib import Path
import subprocess
import sys

from scripts.helpfulness.core import number, read_json, write_json

PINS = {
    "eqbench3": ("EQ-bench/eqbench3", "affc5a000bbdb965b807e09ca7875f8530fa8f56", "eqbench3.py"),
    "eqbench_creative_v3": ("EQ-bench/creative-writing-bench", "c7c3ceef54c40a8ae02dc1c2e1a5e40970fe5c0b", "creative_writing_bench.py"),
}


def prepare(options: dict) -> dict:
    name = options["name"]
    repo, revision, script = PINS[name]
    directory = Path(options["upstream_directory"])
    if options.get("protocol_mode") != "rubric_only":
        raise ValueError("This teacher adapter supports native rubric_only, not inherited Elo ratings")
    iterations = options.get("iterations")
    if type(iterations) is not int or iterations < 1:
        raise ValueError("Declare the number of native EQ-Bench iterations")
    if options["upstream_terms"].startswith("REPLACE_WITH"):
        raise ValueError("Confirm applicable upstream code/data terms before preparation")
    # Export committed bytes only; never execute untracked files or copy API credentials.
    actual = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
    if actual != revision:
        raise ValueError(f"{name}: require {repo}@{revision}")
    paths = subprocess.check_output(
        ["git", "-C", str(directory), "ls-tree", "-r", "--name-only", revision], text=True,
    ).splitlines()
    selected = [
        path for path in paths if (
            path.endswith(".py") or (path.startswith("data/") and path.endswith((".txt", ".json")))
            or path in ("LICENSE", "README.md", "requirements.txt")
        ) and not any(word in path.lower() for word in ("leaderboard", "runs.json", "elo_results"))
    ]
    sources = {
        path: subprocess.check_output(["git", "-C", str(directory), "show", f"{revision}:{path}"]).decode("utf-8")
        for path in selected
    }
    if script not in sources:
        raise ValueError("Native entrypoint missing from staged source")
    result = {"repository": repo, "revision": revision, "entrypoint": script,
              "sources": sources, "iterations": iterations, "upstream_terms": options["upstream_terms"]}
    if name == "eqbench_creative_v3":
        from scripts.helpfulness.native import asset_hashes

        if not options.get("nltk_directory"):
            raise ValueError("Creative Writing v3 requires a separately staged private nltk_directory")
        resources = asset_hashes(Path(options["nltk_directory"]))
        if not resources:
            raise ValueError("Stage CMUdict and Punkt resources before preparation")
        result["nltk_assets"] = resources
    return result


def summarize(name: str, runs: dict, expected: int, valid_case_ids: set[str] | None = None) -> dict:
    if len(runs) != 1:
        raise ValueError("Expected exactly one isolated native EQ-Bench run")
    row = next(iter(runs.values()))
    key = "scenario_tasks" if name == "eqbench3" else "creative_tasks"
    tasks = {
        f"{iteration}/{case_id}": task
        for iteration, cases in row.get(key, {}).items() for case_id, task in cases.items()
    }
    complete = []
    for identity, task in tasks.items():
        if name == "eqbench3":
            valid = task.get("status") == "rubric_scored" and bool(task.get("rubric_scores"))
        else:
            blocks = task.get("results_by_modifier", {})
            valid = task.get("status") in ("completed", "judged") and len(blocks) == 1 and all(
                block.get("judge_scores") and block.get("model_response", "").strip()
                for block in blocks.values()
            )
        if valid and (valid_case_ids is None or identity in valid_case_ids):
            complete.append(task)
    if not expected or len(tasks) > expected:
        raise ValueError("Native EQ-Bench task coverage disagrees with the prepared source")
    value = (
        row.get("results", {}).get("average_rubric_score") if name == "eqbench3"
        else row.get("results", {}).get("benchmark_results", {}).get("eqbench_creative_score")
    )
    maximum = 20 if name == "eqbench3" else 100
    value = number(value, "native rubric score", 0, maximum)
    return {
        "status": "succeeded" if len(complete) == expected else "incomplete",
        "selected_cases": expected, "scored_cases": len(complete),
        "failed_cases": len(tasks) - len(complete), "unprocessed_cases": expected - len(tasks),
        "metrics": {"native_rubric_score": {
            "value": value, "denominator": len(complete), "scale": [0, maximum],
        }},
        "published_protocol_equivalence": False, "scientific_acceptance": "not_performed",
        "elo_computed": False,
    }


def run(options: dict, data: dict, target: dict, judge: dict, output: Path, resume: bool) -> dict:
    from scripts.helpfulness.open_models import provenance

    provenance(target, {"chat", "min_p"})
    provenance(judge, {"chat", "min_p"} if options["name"] == "eqbench_creative_v3" else {"chat"})
    if options["name"] == "eqbench_creative_v3":
        from scripts.helpfulness.native import asset_hashes

        if asset_hashes(Path(options["nltk_directory"])) != data.get("nltk_assets"):
            raise ValueError("Creative-writing NLTK resources changed after preparation")
    working = output / "native"
    working.mkdir(parents=True, exist_ok=True)
    for relative, content in data["sources"].items():
        path = working / relative
        if not path.resolve().is_relative_to(working.resolve()):
            raise ValueError("Native source path escapes the run directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError("Native source changed since preparation")
        path.write_text(content, encoding="utf-8")
    config = output / "native-config.json"
    write_json(config, {
        "name": options["name"], "data": {key: value for key, value in data.items() if key != "sources"},
        "target": target, "judge": judge, "working": str(working.resolve()),
        "nltk_directory": options.get("nltk_directory"),
    })
    environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_MODE="disabled")
    # This subprocess replaces only upstream's network client, not its prompts/scoring.
    command = [sys.executable, "-m", "scripts.helpfulness.eqbench_worker", str(config.resolve())]
    with (output / "native.stdout").open("w", encoding="utf-8") as stdout, (
        output / "native.stderr"
    ).open("w", encoding="utf-8") as stderr:
        result = subprocess.run(command, env=environment, stdout=stdout, stderr=stderr)
    if result.returncode:
        raise RuntimeError(f"Native {options['name']} failed; see {output / 'native.stderr'}")
    coverage = read_json(working / "coverage.json")
    return summarize(
        options["name"], read_json(working / "native-runs.json"),
        coverage["expected_cases"], set(coverage["valid_case_ids"]),
    )
