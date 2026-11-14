"""Offline local-only joining contracts; original synthetic prose, no Reddit texts."""

from contextlib import redirect_stderr, redirect_stdout
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.helpfulness import litbench, litbench_sources as sources
from scripts.helpfulness.core import read_json, read_jsonl


class LitBenchSourcesTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.ids = [
            {"chosen_comment_id": "aa", "rejected_comment_id": "bb"},
            {"chosen_comment_id": "cc", "rejected_comment_id": "aa"},
        ]
        self.stories = [
            {"id": "cc", "story": "A brass cloud rang above the harbor."},
            {"id": "aa", "story": "  A tiny moon lived in the baker's pocket.\n"},
            {"id": "bb", "story": "The clock planted a garden of blue hours."},
        ]

    def jsonl(self, name, rows):
        path = self.root / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def test_exact_join_preserves_native_orientation_order_and_text(self):
        pairs = sources.join_stories(self.ids, self.stories)
        self.assertEqual(pairs, [
            {**self.ids[0], "chosen_story": self.stories[1]["story"],
             "rejected_story": self.stories[2]["story"]},
            {**self.ids[1], "chosen_story": self.stories[0]["story"],
             "rejected_story": self.stories[1]["story"]},
        ])
        self.assertEqual(litbench.load_pairs(self.jsonl("pairs.jsonl", pairs)), pairs)

    def test_rejects_duplicate_missing_unexpected_malformed_and_deleted_stories(self):
        bad_corpora = [
            self.stories + [self.stories[0]],
            self.stories[:-1],
            self.stories + [{"id": "dd", "story": "A synthetic extra story."}],
            [{"id": "t1_cc", "story": "Wrong namespace."}] + self.stories[1:],
            [{"id": "cc", "comment_id": "dd", "story": "Wrong alias."}] + self.stories[1:],
        ]
        for story in ("", " \n ", None, "[deleted]", " [REMOVED] "):
            bad_corpora.append([{"id": "cc", "story": story}] + self.stories[1:])
        for rows in bad_corpora:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                sources.join_stories(self.ids, rows)

    def test_rejects_invalid_pair_indices_and_incomplete_coverage(self):
        for rows in (
            [], self.ids + [self.ids[0]],
            self.ids + [{"chosen_comment_id": "bb", "rejected_comment_id": "aa"}],
            [{"chosen_comment_id": "aa", "rejected_comment_id": "aa"}],
            [{"chosen_comment_id": "aa ", "rejected_comment_id": "bb"}],
            [{"chosen_comment_id": 1, "rejected_comment_id": "bb"}],
            [{"chosen_comment_id": "aa"}],
        ):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                sources.join_stories(rows, self.stories)
        with self.assertRaisesRegex(ValueError, "complete coverage"):
            sources.validate_ids(self.ids, sources.PAIR_COUNT)

    def test_train_test_leakage_checks_individual_stories_not_only_pairs(self):
        disjoint = [{"chosen_comment_id": "dd", "rejected_comment_id": "ee"}]
        self.assertEqual(sources.join_stories(self.ids, self.stories, disjoint),
                         sources.join_stories(self.ids, self.stories))
        for row in (
            {"chosen_comment_id": "dd", "rejected_comment_id": "aa"},
            {"chosen_comment_id": "bb", "rejected_comment_id": "aa"},
        ):
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, "leakage"):
                sources.join_stories(self.ids, self.stories, [row])
        with self.assertRaises(ValueError):
            sources.join_stories(self.ids, self.stories, [])

    def test_native_loader_anchors_bytes_before_reading_exact_schema(self):
        index = self.root / "native.parquet"
        index.write_bytes(b"synthetic native reader fixture, not a real Parquet")
        with self.assertRaisesRegex(ValueError, "checksum"):
            sources.load_native_ids(index)
        table = SimpleNamespace(
            column_names=list(sources.ID_FIELDS),
            schema=SimpleNamespace(field=lambda key: SimpleNamespace(type="string")),
            to_pylist=lambda: self.ids,
        )
        parquet = SimpleNamespace(read_table=lambda stream: table)
        modules = {"pyarrow": SimpleNamespace(parquet=parquet), "pyarrow.parquet": parquet}
        with (
            patch.object(sources, "NATIVE_INDEX_SHA256", hashlib.sha256(index.read_bytes()).hexdigest()),
            patch.object(sources, "PAIR_COUNT", len(self.ids)),
            patch.dict(sys.modules, modules),
        ):
            self.assertEqual(sources.load_native_ids(index), self.ids)
            table.column_names = ["chosen_story", "rejected_story"]
            with self.assertRaisesRegex(ValueError, "schema"):
                sources.load_native_ids(index)

    def test_prepared_outputs_match_existing_validator_and_record_leakage_limits(self):
        stories = self.jsonl("stories.jsonl", self.stories)
        terms = self.root / "terms.txt"
        terms.write_text("Original synthetic fixtures; author grants local testing only.", encoding="utf-8")
        for training in (None, self.jsonl("training.jsonl", [
            {"chosen_comment_id": "dd", "rejected_comment_id": "ee"},
        ])):
            output = self.root / ("with-training" if training else "without-training")
            with patch.object(sources, "load_native_ids", return_value=self.ids):
                report = sources.prepare(self.root / "native.parquet", stories, terms, output, training)
            self.assertEqual(read_json(output / "data-manifest.json"), report)
            self.assertEqual(read_jsonl(output / "heldout-ids.jsonl"), self.ids)
            pairs_path = output / "pairs.jsonl"
            self.assertEqual(litbench.load_pairs(pairs_path), sources.join_stories(self.ids, self.stories))
            self.assertEqual(report["rehydrated_sha256"], hashlib.sha256(pairs_path.read_bytes()).hexdigest())
            self.assertEqual(report["ids_sha256"],
                             hashlib.sha256((output / "heldout-ids.jsonl").read_bytes()).hexdigest())
            self.assertEqual(report["local_stories_sha256"], hashlib.sha256(stories.read_bytes()).hexdigest())
            self.assertEqual(report["scope"], "scorer_validation_not_teacher_quality")
            self.assertEqual(report["train_test_leakage_check"].startswith("not_checked"), training is None)
            with self.assertRaisesRegex(ValueError, "fresh"):
                sources.prepare(self.root / "native.parquet", stories, terms, output, training)

    @unittest.skipUnless(os.name == "posix", "POSIX permissions only; not Windows ACLs")
    def test_story_output_directory_is_owner_private(self):
        self.root.chmod(0o700)
        if stat.S_IMODE(self.root.stat().st_mode) != 0o700:
            self.skipTest("Filesystem does not enforce POSIX modes (e.g. WSL Windows mount)")
        stories = self.jsonl("stories.jsonl", self.stories)
        terms = self.root / "terms.txt"
        terms.write_text("Original synthetic fixtures; local testing permission.", encoding="utf-8")
        output = self.root / "private-output"
        with patch.object(sources, "load_native_ids", return_value=self.ids):
            sources.prepare(self.root / "native.parquet", stories, terms, output)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o700)

    def test_jsonl_preserves_unicode_separators_and_escaped_linebreaks(self):
        rows = [{"id": "aa", "story": "First\u2028second\u2029third\nfourth\r\nfifth."},
                {"id": "bb", "story": "Another synthetic story."}]
        path = self.root / "unicode.jsonl"
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        path.write_bytes(content.encode("utf-8"))
        parsed, checksum = sources._read_rows(path)
        self.assertEqual(parsed, rows)
        self.assertEqual(checksum, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_csv_and_jsonl_local_inputs_reject_ambiguous_fields(self):
        path = self.root / "stories.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["id", "story"])
            writer.writeheader()
            writer.writerows(self.stories)
        rows, _ = sources._read_rows(path)
        self.assertEqual(rows, self.stories)
        for content in ('{"id":"aa","id":"bb","story":"Synthetic."}\n', '["not an object"]\n'):
            path = self.root / "ambiguous.jsonl"
            path.write_text(content, encoding="utf-8")
            with self.assertRaises(ValueError):
                sources._read_rows(path)
        path = self.root / "ambiguous.csv"
        path.write_text("id,id,story\naa,bb,Synthetic.\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            sources._read_rows(path)

    def test_cli_fails_closed_without_writing_outputs_or_running_models(self):
        stories = self.jsonl("missing.jsonl", self.stories[:-1])
        terms = self.root / "terms.txt"
        terms.write_text("Synthetic testing permission only.", encoding="utf-8")
        output = self.root / "output"
        args = ["--native-index", str(self.root / "native.parquet"),
                "--stories", str(stories), "--terms-review", str(terms),
                "--output-directory", str(output)]
        with (
            patch.object(sources, "load_native_ids", return_value=self.ids),
            patch.object(litbench, "local_reward") as model,
            redirect_stderr(io.StringIO()) as errors,
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(sources.main(args), 1)
            self.assertIn("Missing local stories", errors.getvalue())
            self.assertFalse(output.exists())
            terms.write_text("REPLACE_WITH authorization", encoding="utf-8")
            self.assertEqual(sources.main(args), 1)
            self.assertFalse(output.exists())
            model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
