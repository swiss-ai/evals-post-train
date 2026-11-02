"""SuperGPQA, run via `inspect eval scripts/inspect_ai/tasks/supergpqa.py`.

SuperGPQA (https://arxiv.org/abs/2502.14739, `m-a-p/SuperGPQA`, public, one JSONL file):
26,529 graduate-level multiple-choice questions across 13 disciplines and 285 subfields,
with 4 to 10 options each. No framework in this repo has it (EvalScope does, but its
runner here only wires SWE-bench), so this is a from-scratch Inspect task.

Setup: Inspect's `multiple_choice()` solver with chain of thought (the model reasons,
then ends with "ANSWER: <letter>") and the `choice()` scorer, so the score is accuracy
over the whole set. The questions are shuffled once with a fixed seed before `--limit`
(or `num_samples`) is applied: the file is ordered by discipline, so the first N would
otherwise cover only a few of them. Each sample keeps its discipline, field, subfield,
difficulty and `is_calculation` in its metadata for breakdowns from the log.

Runs 26,529 chain-of-thought generations in full: test with `--limit 50`; a few
hundred to a thousand questions give a standard error of 1.5 to 3 points.
"""

from __future__ import annotations

import json
import random

from huggingface_hub import hf_hub_download
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice


DATASET = "m-a-p/SuperGPQA"
FILENAME = "SuperGPQA-all.jsonl"
# Pinned dataset revision.
REVISION = "4430d4458112c7d4497fdcf94d7cc223313d6acf"

LETTERS = "ABCDEFGHIJ"


def load_samples(
    path: str,
    num_samples: int | None = None,
    shuffle_seed: int | None = 0,
    discipline: str | None = None,
) -> list[Sample]:
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if discipline:
        rows = [r for r in rows if r["discipline"].lower() == discipline.lower()]
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(rows)
    if num_samples is not None:
        rows = rows[:num_samples]
    return [
        Sample(
            id=row["uuid"],
            input=row["question"],
            choices=row["options"],
            target=row["answer_letter"],
            metadata={
                k: row[k]
                for k in ("discipline", "field", "subfield", "difficulty", "is_calculation")
            },
        )
        for row in rows
    ]


@task
def supergpqa(
    num_samples: int | None = None,
    shuffle_seed: int | None = 0,
    discipline: str | None = None,
    cot: bool = True,
    data_path: str | None = None,
) -> Task:
    """
    Args:
        num_samples: Keep only this many questions (after shuffling).
        shuffle_seed: Seed for the one-off shuffle; None keeps the file order.
        discipline: Keep only one discipline, e.g. "Science" or "Engineering".
        cot: Chain-of-thought prompt (reason, then "ANSWER: <letter>").
        data_path: A local copy of SuperGPQA-all.jsonl (used by the tests).
    """
    path = data_path or hf_hub_download(
        DATASET, FILENAME, repo_type="dataset", revision=REVISION
    )
    return Task(
        dataset=MemoryDataset(
            load_samples(path, num_samples, shuffle_seed, discipline), name="supergpqa"
        ),
        solver=multiple_choice(cot=cot),
        scorer=choice(),
    )
