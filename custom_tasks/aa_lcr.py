"""AA-LCR (Artificial Analysis Long Context Reasoning benchmark), run via
`inspect eval custom_tasks/aa_lcr.py` -- no inspect_evals implementation
exists to lean on (unlike hle/gpqa_diamond), so this is a from-scratch task.

https://artificialanalysis.ai/methodology/intelligence-benchmarking and the
dataset's own README (huggingface.co/datasets/ArtificialAnalysis/AA-LCR):
100 questions, each requiring reasoning across a ~100k-token Document Set
(2-16 real-world documents: company reports, government consultations,
legal, academic, marketing, survey materials), pass@1, equality-checker
grading. Full dataset is public (Apache 2.0 question set; the document set
"provided as a text representation of documents publicly available at time
of dataset creation, no copyright claimed") -- unlike AA-Omniscience there
is no held-back subset, so this one *can* be fully AAII-comparable.

Prompt template and document-loading order (data_source_filenames, in
order) are copied verbatim from the dataset README's own code sample, not
reconstructed -- verified against the actual dataset files (CSV +
extracted_text/AA-LCR_extracted-text.zip on the HF repo), not just the
docs, since the README's prose table uses "Company Documents"/"Marketing
Materials" while the CSV's own document_category values and the zip's
folder names both actually use "Company_Documents"/"Marketing" -- confirmed
these already agree with each other, so no category-name mapping is needed,
but this was verified, not assumed.

Grading model discrepancy, worth flagging rather than silently picking one:
the AAII methodology page states GPT-5.6 Luna (medium) as the judge across
AA's evals generally, including this one -- the same judge hle.py/
omniscience.py already use. The dataset's OWN README, however, documents
its original reference implementation's judge as "Qwen3 235B A22B 2507
Non-reasoning". These may simply be sequential (AA's evals moved to a
unified GPT-5.6 Luna judge policy after AA-LCR's 2025 dataset card was
written) rather than contradictory, but neither is independently confirmed
current here. Defaults to GPT-5.6 Luna for consistency with the rest of
this repo's AAII protocol implementations (settings.aa_lcr_grader_model is
its own setting, not a reuse of hle_grader_model/omniscience_grader_model,
same reasoning as omniscience.py: coinciding on today's judge model is not
a reason to couple them). The grading PROMPT text itself, unlike the judge
model choice, is copied verbatim from the dataset README's own "Scoring
Approach" section -- a primary source for this specific benchmark, not an
adaptation borrowed from a different one.

Cost/runtime note for anyone testing this: unlike every other task in this
repo, a single sample here is a ~100k-token prompt (the full Document Set)
-- test with --limit 1, not more, unless you mean to spend real time/money
on it.
"""

import csv
import difflib
import re
import zipfile

from huggingface_hub import hf_hub_download
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser, get_model
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState, generate

DATASET_REPO = "ArtificialAnalysis/AA-LCR"
CSV_FILENAME = "AA-LCR_Dataset.csv"
ZIP_FILENAME = "extracted_text/AA-LCR_extracted-text.zip"

# Verbatim from the dataset README's "Prompt Template" section.
_PROMPT_TEMPLATE = """BEGIN INPUT DOCUMENTS

{documents_text}

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

{question}

END QUESTION
"""

# Verbatim from the dataset README's "Scoring Approach" section.
_GRADING_TEMPLATE = """Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.
For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.

The question, for reference only: {question}
The OFFICIAL ANSWER: {official_answer}
CANDIDATE ANSWER TO ASSESS: {candidate_answer}

Reply only with CORRECT or INCORRECT.
"""

# The grading prompt asks for a terse reply, but a reasoning-capable judge
# may still think out loud first -- same last-match-wins defense as every
# other custom grader in this repo, for the same injection-resistance
# reason (an early, off-menu mention in the judge's own reasoning must not
# be picked up over its actual final verdict).
_GRADE_RE = re.compile(r"(?is).*\b(correct|incorrect)\b")


def _resolve_directory(directory: str, filenames: set[str], available: list[str]) -> dict[str, str]:
    """Every name in `filenames` resolved to its actual zip path among
    `available` (all real, non-directory zip entries under `directory`).

    `filenames` must be the UNION of every data_source_filenames entry any
    row references under this directory, not one row's list: document sets
    are shared across multiple questions, and different questions
    referencing the same set can use different subsets of its files, so
    resolving row-by-row leaves other rows' files sitting in the pool as
    decoys with no principled way to tell them apart from a real mismatch.
    Even at the union level the directory can have genuinely unreferenced
    extra files (confirmed: legal_eu_ai has one, present in the zip but
    named in no row's data_source_filenames at all) -- so an unmatched-name
    is resolved against whatever's left in the pool, not against a pool
    required to be the exact same size.

    At least two filenames in AA's own published zip are corrupted by a
    CP437-vs-UTF-8 mis-decode at zip-creation time (confirmed by
    inspecting the archive directly, not assumed) -- an apostrophe in one
    file, a Turkish cedilla-s in another, each mangled into different
    garbage bytes. No single normalization (stripping non-ASCII characters
    from both sides was tried first) matches every case, so unmatched
    names are instead paired against unclaimed files by string similarity
    (difflib), greedily, highest-confidence pair first -- safe here
    specifically because these filenames are long, largely-intact, and
    mutually dissimilar (a corrupted run of a handful of characters barely
    moves the overall similarity ratio, and there's nothing else in the
    same directory close enough to be confused for it). A pairing below
    the similarity floor, or a name left unmatched after pairing, raises
    rather than guess.
    """
    resolved: dict[str, str] = {}
    unmatched = []
    remaining = list(available)
    for name in filenames:
        full = f"{directory}/{name}"
        if full in remaining:
            resolved[name] = full
            remaining.remove(full)
        else:
            unmatched.append(name)
    if unmatched:
        pairs = sorted(
            (
                (difflib.SequenceMatcher(None, name, entry.rpartition("/")[2]).ratio(), name, entry)
                for name in unmatched
                for entry in remaining
            ),
            key=lambda p: p[0],
            reverse=True,
        )
        SIMILARITY_FLOOR = 0.85
        used_names: set[str] = set()
        used_entries: set[str] = set()
        for ratio, name, entry in pairs:
            if name in used_names or entry in used_entries:
                continue
            if ratio < SIMILARITY_FLOOR:
                break
            resolved[name] = entry
            used_names.add(name)
            used_entries.add(entry)
        missing = [n for n in unmatched if n not in used_names]
        if missing:
            raise KeyError(
                f"could not confidently resolve under {directory!r} (similarity < "
                f"{SIMILARITY_FLOOR}): {missing}"
            )
    return resolved


def _load_dataset() -> MemoryDataset:
    csv_path = hf_hub_download(DATASET_REPO, CSV_FILENAME, repo_type="dataset")
    zip_path = hf_hub_download(DATASET_REPO, ZIP_FILENAME, repo_type="dataset")

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    samples = []
    with zipfile.ZipFile(zip_path) as archive:
        entries_by_dir: dict[str, list[str]] = {}
        for entry in archive.namelist():
            directory, _, filename = entry.rpartition("/")
            if filename:  # skip directory-marker entries (name ends in "/")
                entries_by_dir.setdefault(directory, []).append(entry)

        # Union every row's filenames per directory first (see
        # _resolve_directory for why row-by-row resolution is unsafe), then
        # resolve each directory exactly once.
        filenames_by_dir: dict[str, set[str]] = {}
        for row in rows:
            directory = f"lcr/{row['document_category']}/{row['document_set_id']}"
            filenames_by_dir.setdefault(directory, set()).update(
                row["data_source_filenames"].split(";")
            )
        resolved_by_dir = {
            directory: _resolve_directory(directory, names, entries_by_dir.get(directory, []))
            for directory, names in filenames_by_dir.items()
        }

        for row in rows:
            directory = f"lcr/{row['document_category']}/{row['document_set_id']}"
            filenames = row["data_source_filenames"].split(";")
            paths = [resolved_by_dir[directory][name] for name in filenames]
            docs = [archive.read(path).decode("utf-8") for path in paths]
            documents_text = "\n\n".join(
                f"BEGIN DOCUMENT {i + 1}:\n{doc}\nEND DOCUMENT {i + 1}"
                for i, doc in enumerate(docs)
            )
            # answer is semicolon-separated acceptable criteria (per the
            # README's own load_questions()); rejoined for the single
            # {official_answer} slot the grading template expects.
            answer = "; ".join(row["answer"].split(";"))
            samples.append(
                Sample(
                    input=_PROMPT_TEMPLATE.format(
                        documents_text=documents_text, question=row["question"]
                    ),
                    target=answer,
                    id=int(row["question_id"]),
                    metadata={
                        "document_category": row["document_category"],
                        "document_set_id": row["document_set_id"],
                        "input_tokens_reported": row.get("input_tokens"),
                        # The short original question, not the ~100k-token
                        # documents-included prompt (state.input_text) --
                        # kept separately so the scorer can cite it in the
                        # grading prompt without re-sending every document
                        # to the grader too. See aa_lcr_scorer().
                        "question": row["question"],
                    },
                )
            )
    return MemoryDataset(samples, name="aa_lcr")


@scorer(metrics=[accuracy(), stderr()])
def aa_lcr_scorer(model_role: str = "grader") -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        # required=True: AA's protocol is a fixed external judge, not the
        # model under test grading itself -- same reasoning as
        # omniscience.py's scorer.
        grader = get_model(role=model_role, required=True)
        # The short original question (see _load_dataset), not
        # state.input_text -- that's the ~100k-token documents-included
        # prompt, and re-sending every document to the grader too would
        # roughly double this task's already-substantial cost for no
        # benefit: the README's own grading prompt calls this "the
        # question, for reference only," and the grader is checking
        # candidate-vs-official answer equivalence, not re-deriving the
        # answer from the source documents itself.
        prompt = _GRADING_TEMPLATE.format(
            question=state.metadata["question"],
            official_answer=target.text,
            candidate_answer=state.output.completion,
        )
        result = await grader.generate([ChatMessageUser(content=prompt)])
        match = _GRADE_RE.search(result.completion)
        # An unparseable grade is scored INCORRECT rather than raising --
        # same reasoning as omniscience.py: a single malformed judge
        # response shouldn't fail an entire run, and treating a non-answer
        # as a penalized guess is the conservative choice for a benchmark
        # with no "not attempted" category of its own.
        value = CORRECT if match and match.group(1).lower() == "correct" else INCORRECT
        return Score(
            value=value,
            answer=state.output.completion,
            explanation=result.completion,
        )

    return score


@task
def aa_lcr(model_role: str = "grader") -> Task:
    return Task(
        dataset=_load_dataset(),
        solver=[generate()],
        scorer=aa_lcr_scorer(model_role=model_role),
    )
