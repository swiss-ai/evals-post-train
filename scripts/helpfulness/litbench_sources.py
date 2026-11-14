"""Join authorized local stories to pinned LitBench IDs; never fetch story text.

Run with ``python -m scripts.helpfulness.litbench_sources --help``.
The local CSV/JSONL corpus uses upstream rehydrate.py's ``id`` and ``story``
fields. Supply exactly the held-out stories, not a mixed train/test corpus.
The terms-review file must document the user's lawful local-data authorization
and reviewed source/Reddit terms; this tool neither grants nor approves rights.
Outputs contain story text: keep them local and out of version control.
This prepares scalar creative-scorer validation, never teacher benchmarking.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys

from scripts.helpfulness.core import write_json

SOURCE = "SAA-Lab/LitBench-Test"
DATA_REVISION = "21e2739a2bce58bf0bc9317be412e8f33ed8f370"
NATIVE_INDEX_SHA256 = "02cfed095f70748ed72cb38f5e5e549238e1cd52c7b627ded6af405a359c6a9d"
PAIR_COUNT = 2381
ID_FIELDS = ("chosen_comment_id", "rejected_comment_id")
NATIVE_CODE = (
    "https://github.com/drfein/LitBench/blob/"
    "c1017dbe91eff178b3e318d775f600a89774089c/scripts/rehydrate.py"
)


def _native_id(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-z0-9]+", value) is not None


def validate_ids(rows: list[dict], expected_count: int | None = None) -> list[dict]:
    """Keep the original order and orientation; never normalize native IDs."""
    pairs, seen = [], set()
    for row in rows:
        if not isinstance(row, dict) or any(not _native_id(row.get(key)) for key in ID_FIELDS):
            raise ValueError("Expected exact native chosen/rejected comment IDs")
        chosen, rejected = (row[key] for key in ID_FIELDS)
        key = frozenset((chosen, rejected))
        if chosen == rejected or key in seen:
            raise ValueError("Duplicate, reversed duplicate, or self-paired native IDs")
        seen.add(key)
        pairs.append({key: row[key] for key in ID_FIELDS})
    if not pairs or (expected_count is not None and len(pairs) != expected_count):
        raise ValueError("Require complete coverage of the pinned 2381 held-out story pairs")
    return pairs


def load_native_ids(path: Path) -> list[dict]:
    """Read only the exact pinned local Parquet, using an existing Arrow runtime."""
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != NATIVE_INDEX_SHA256:
        raise ValueError("Native held-out index checksum differs from the pinned HF revision")
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "Reading the pinned local Parquet needs an already prepared pyarrow runtime; "
            "no dependencies or data are downloaded by this tool"
        ) from error
    table = parquet.read_table(io.BytesIO(content))
    if set(table.column_names) != set(ID_FIELDS) or any(
        str(table.schema.field(key).type) != "string" for key in ID_FIELDS
    ):
        raise ValueError("Unexpected native held-out index schema")
    return validate_ids(table.to_pylist(), PAIR_COUNT)


def _read_rows(path: Path) -> tuple[list[dict], str]:
    content = path.read_bytes()
    text = content.decode("utf-8-sig")
    if path.suffix.lower() == ".csv":
        reader = csv.DictReader(io.StringIO(text, newline=""))
        fields = reader.fieldnames or []
        if len(set(fields)) != len(fields):
            raise ValueError("Duplicate CSV column names")
        rows = list(reader)
    elif path.suffix.lower() == ".jsonl":
        def unique_object(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError("Duplicate JSON field")
                result[key] = value
            return result
        rows = [json.loads(line, object_pairs_hook=unique_object)
                for line in text.split("\n") if line.strip()]
    else:
        raise ValueError("Local stories/training IDs must be CSV or JSONL")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Expected local rows to be objects")
    return rows, hashlib.sha256(content).hexdigest()


def join_stories(
    ids: list[dict], stories: list[dict], training_ids: list[dict] | None = None,
) -> list[dict]:
    """Require exact story-ID coverage; preserve all supplied story bytes as text."""
    ids = validate_ids(ids)
    expected = {row[key] for row in ids for key in ID_FIELDS}
    if training_ids is not None:
        training = {row[key] for row in validate_ids(training_ids) for key in ID_FIELDS}
        if training & expected:
            raise ValueError("Detected train/test story-ID leakage")
    by_id = {}
    for row in stories:
        if not isinstance(row, dict) or not _native_id(row.get("id")):
            raise ValueError("Local stories require exact native id and story fields")
        key, story = row["id"], row.get("story")
        if key in by_id:
            raise ValueError("Duplicate local story ID")
        if key not in expected:
            raise ValueError("Unexpected local story ID; supply only the held-out corpus")
        if "comment_id" in row and row["comment_id"] != key:
            raise ValueError("Mismatched local id/comment_id fields")
        if (
            not isinstance(story, str) or not story.strip()
            or story.strip().lower() in {"[deleted]", "[removed]"}
        ):
            raise ValueError("Missing, empty, deleted, or removed local story")
        by_id[key] = story
    if set(by_id) != expected:
        raise ValueError("Missing local stories; exact expected-ID coverage is required")
    return [
        {**row, "chosen_story": by_id[row["chosen_comment_id"]],
         "rejected_story": by_id[row["rejected_comment_id"]]}
        for row in ids
    ]


def prepare(
    native_index: Path, stories_path: Path, terms_review: Path, output: Path,
    training_ids_path: Path | None = None,
) -> dict:
    """Validate everything before creating a fresh, local-only output directory."""
    if output.exists():
        raise ValueError("Use a fresh output directory; existing data is never overwritten")
    terms_bytes = terms_review.read_bytes()
    terms = terms_bytes.decode("utf-8-sig").strip()
    if not terms or terms.startswith("REPLACE_WITH"):
        raise ValueError("Provide a reviewed terms/local-data authorization record")
    ids = load_native_ids(native_index)
    stories, stories_sha256 = _read_rows(stories_path)
    training_ids, training_sha256 = None, None
    if training_ids_path is not None:
        training_ids, training_sha256 = _read_rows(training_ids_path)
    pairs = join_stories(ids, stories, training_ids)

    def encode(rows):
        return "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                       for row in rows).encode("utf-8")

    ids_bytes, pairs_bytes = encode(ids), encode(pairs)
    manifest = {
        "source": SOURCE, "revision": DATA_REVISION, "native_split": "train",
        "scope": "scorer_validation_not_teacher_quality",
        "native_index_sha256": NATIVE_INDEX_SHA256,
        "native_index_url": (
            f"https://huggingface.co/datasets/{SOURCE}/resolve/{DATA_REVISION}/"
            "data/train-00000-of-00001.parquet"
        ),
        "native_rehydration_code": NATIVE_CODE,
        "ids_sha256": hashlib.sha256(ids_bytes).hexdigest(),
        "rehydrated_sha256": hashlib.sha256(pairs_bytes).hexdigest(),
        "local_stories_sha256": stories_sha256,
        "terms": terms, "terms_review_sha256": hashlib.sha256(terms_bytes).hexdigest(),
        "pairs": len(pairs), "stories": len(stories),
        "coverage": "exact_native_ids_and_oriented_pairs",
        "training_ids_sha256": training_sha256,
        "train_test_leakage_check": (
            "no_story_id_overlap_with_supplied_training_index"
            if training_ids_path is not None else "not_checked_no_training_index_supplied"
        ),
        "limitations": "No authorization, text authenticity, or semantic leakage certification",
        "importer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    (output / "heldout-ids.jsonl").write_bytes(ids_bytes)
    (output / "pairs.jsonl").write_bytes(pairs_bytes)
    write_json(output / "data-manifest.json", manifest)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-index", required=True, type=Path,
                        help="Already-staged pinned LitBench-Test Parquet (IDs only)")
    parser.add_argument("--stories", required=True, type=Path,
                        help="Authorized local CSV/JSONL with native id, story fields")
    parser.add_argument("--terms-review", required=True, type=Path,
                        help="Local text record of reviewed terms and lawful-data authorization")
    parser.add_argument("--training-ids", type=Path,
                        help="Optional local native-pair CSV/JSONL index for ID leakage checks")
    parser.add_argument("--output-directory", required=True, type=Path,
                        help="Fresh local-only directory; outputs include story texts")
    args = parser.parse_args(argv)
    try:
        report = prepare(args.native_index, args.stories, args.terms_review,
                         args.output_directory, args.training_ids)
        print(f"Prepared {report['pairs']} local scorer-validation pairs; no scoring performed")
        return 0
    except (ValueError, RuntimeError, OSError, UnicodeError) as error:
        print(f"LitBench local import failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
