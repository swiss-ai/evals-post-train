"""PopQA, run via `inspect eval scripts/inspect_ai/tasks/popqa.py`.

PopQA (https://arxiv.org/abs/2212.10511, `akariasai/PopQA`, public, one TSV): 14,267
entity questions ("What is George Rankin's occupation?") over 16 Wikidata relations,
chosen so the subject's popularity (monthly Wikipedia page views, `s_pop`) spans from
very rare to very common. It measures how much long-tail factual knowledge a model
holds in its weights, with no retrieval. No framework in this repo has it, so this is a
from-scratch Inspect task.

Setup: closed-book and zero-shot with a short-answer instruction, greedy decoding, at
most 64 new tokens. Scoring is the standard PopQA metric: the answer is correct if any
of the question's gold aliases (`possible_answers`, 1 to 190 per question) appears in
the generation (case-insensitive substring match; Inspect's `includes()` scorer). Some
aliases are very short ("pol" for politician), so substring matching slightly
over-credits verbose answers; this is the metric the benchmark's authors and most
follow-up papers use, so it is kept as is. Each sample keeps its relation (`prop`) and
subject/object popularity in its metadata for accuracy-by-popularity breakdowns.

Runs 14,267 short generations in full (the generations are brief, so this is cheap);
use `--limit 100` to test.
"""

from __future__ import annotations

import csv
import json
import random

from huggingface_hub import hf_hub_download
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import includes
from inspect_ai.solver import generate


DATASET = "akariasai/PopQA"
FILENAME = "test.tsv"
# Pinned dataset revision.
REVISION = "098765c79ea10a2cb19c828324e33281b8336ec0"

PROMPT = "Answer the question with a short phrase only. Do not explain.\n\nQuestion: {question}\nAnswer:"


def load_samples(
    path: str, num_samples: int | None = None, shuffle_seed: int | None = None
) -> list[Sample]:
    csv.field_size_limit(10**9)
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(rows)
    if num_samples is not None:
        rows = rows[:num_samples]
    return [
        Sample(
            id=row["id"],
            input=PROMPT.format(question=row["question"]),
            target=json.loads(row["possible_answers"]),
            metadata={
                "prop": row["prop"],
                "s_pop": int(row["s_pop"]),
                "o_pop": int(row["o_pop"]),
            },
        )
        for row in rows
    ]


@task
def popqa(
    num_samples: int | None = None,
    shuffle_seed: int | None = None,
    max_tokens: int = 64,
    data_path: str | None = None,
) -> Task:
    """
    Args:
        num_samples: Keep only this many questions.
        shuffle_seed: Shuffle once with this seed before keeping num_samples.
        max_tokens: New-token limit per answer.
        data_path: A local copy of test.tsv (used by the tests).
    """
    path = data_path or hf_hub_download(
        DATASET, FILENAME, repo_type="dataset", revision=REVISION
    )
    return Task(
        dataset=MemoryDataset(load_samples(path, num_samples, shuffle_seed), name="popqa"),
        solver=generate(),
        scorer=includes(ignore_case=True),
        config=GenerateConfig(temperature=0.0, max_tokens=max_tokens),
    )
