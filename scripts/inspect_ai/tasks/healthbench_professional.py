"""HealthBench Professional, run via `inspect eval scripts/inspect_ai/tasks/healthbench_professional.py`.

HealthBench Professional (OpenAI, https://arxiv.org/abs/2604.27470, `openai/healthbench-professional`,
MIT) is 525 clinician conversations (care consult, writing and documentation, medical research),
each graded against physician-written rubrics. The data uses different field names from the
original HealthBench, so this task converts each record to the original format and runs
inspect_evals' own `healthbench` task on it: same prompt handling, same rubric scoring
(met points over positive points, clipped to [0, 1], bootstrap standard error).

OpenAI does not release an official implementation. Their reported numbers use an internal
one; the `simple-evals` HealthBench code is the reference, with an option matching the
internal settings (grader GPT-5.4 at low reasoning effort). Scores here use whatever grader
you pass, so they are not comparable with OpenAI's.

Pass the grader with `--task-arg judge_model=openai-api/swissai/<served model name>`. Each
example's use case, type, difficulty and specialty are kept as `example_tags`.

Per the dataset card, do not publish examples from this dataset in plain text online.
"""

from __future__ import annotations

import json
import tempfile

from huggingface_hub import hf_hub_download
from inspect_ai import Task, task


DATASET = "openai/healthbench-professional"
FILENAME = "healthbench_professional_eval.jsonl"
# Pinned dataset revision.
REVISION = "349962fd46dd02343a0d8a606491baf59154ea1a"


def to_healthbench_record(record: dict) -> dict:
    """One Professional record in the original HealthBench layout (`prompt`, `rubrics`, ...)."""
    return {
        "prompt_id": record["id"],
        "prompt": [{"role": m["role"], "content": m["content"]} for m in record["conversation"]["messages"]],
        "rubrics": [
            {"criterion": item["criterion_text"], "points": item["points"], "tags": []}
            for item in record["rubric_items"]
        ],
        "example_tags": [
            f"use_case:{record['use_case']}",
            f"type:{record['type']}",
            f"difficulty:{record['difficulty']}",
            f"specialty:{record['specialty']}",
        ],
    }


def convert(src_path: str, num_samples: int | None = None) -> str:
    """Write the converted records to a temp JSONL file and return its path."""
    with open(src_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if num_samples is not None:
        records = records[:num_samples]
    out = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    with out:
        for r in records:
            out.write(json.dumps(to_healthbench_record(r), ensure_ascii=False) + "\n")
    return out.name


@task
def healthbench_professional(
    judge_model: str,
    num_samples: int | None = None,
    data_path: str | None = None,
    **kwargs,
) -> Task:
    """
    Args:
        judge_model: Grader model name (required, no default).
        num_samples: Keep only the first N conversations.
        data_path: A local copy of the dataset file (used by the tests).
        **kwargs: Passed through to inspect_evals' `healthbench` (for example
            `bootstrap_samples`, `judge_max_tokens`).
    """
    from inspect_evals.healthbench import healthbench

    src = data_path or hf_hub_download(DATASET, FILENAME, repo_type="dataset", revision=REVISION)
    return healthbench(local_path=convert(src, num_samples), judge_model=judge_model, **kwargs)
