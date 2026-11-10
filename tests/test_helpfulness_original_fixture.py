"""Fixture normalization tests, not published benchmark approval."""

from copy import deepcopy
import unittest

from scripts.helpfulness.core import digest
from scripts.helpfulness.prepare_ifeval_replay import normalize_outputs, original_metrics


class OriginalFixtureTests(unittest.TestCase):
    def setUp(self):
        self.docs = [{"key": 9, "prompt": "Synthetic instruction",
                      "instruction_id_list": ["synthetic:a", "synthetic:b"], "kwargs": [{}, {}]}]
        self.responses = [{"prompt": "Synthetic instruction", "response": "Synthetic response"}]
        self.strict = [{"prompt": "Synthetic instruction", "response": "Synthetic response",
                        "instruction_id_list": ["synthetic:a", "synthetic:b"],
                        "follow_all_instructions": False, "follow_instruction_list": [True, False]}]
        self.loose = [dict(self.strict[0], follow_all_instructions=True, follow_instruction_list=[True, True])]
        self.metrics = original_metrics("prompt-level: 0.0\ninstruction-level: 0.5\n"
                                       "prompt-level: 1.0\ninstruction-level: 1.0\n")

    def normalize(self, docs=None, responses=None, strict=None, loose=None):
        return normalize_outputs(self.docs if docs is None else docs,
                                 self.responses if responses is None else responses,
                                 self.strict if strict is None else strict,
                                 self.loose if loose is None else loose, self.metrics)

    def test_exact_original_decisions_and_independent_denominators(self):
        rows, cases, aggregate = self.normalize()
        self.assertEqual(cases[0]["input_sha256"], digest(rows[0]))
        self.assertEqual(cases[0]["scores"]["inst_level_strict_acc"], [True, False])
        self.assertEqual(cases[0]["scores"]["prompt_level_loose_acc"], True)
        self.assertEqual(aggregate["denominators"]["inst_level_strict_acc"], 2)
        self.assertEqual(aggregate["denominators"]["prompt_level_strict_acc"], 1)
        self.assertEqual(aggregate["metrics"], self.metrics)
        self.assertEqual(rows[0]["requests"], [{"input_text": self.docs[0]["prompt"]}])

    def test_incomplete_original_coverage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.normalize(loose=[])

    def test_duplicate_prompt_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.normalize(responses=self.responses * 2)

    def test_duplicate_original_key_is_rejected(self):
        docs = [*self.docs, dict(self.docs[0], prompt="Another synthetic instruction")]
        with self.assertRaisesRegex(ValueError, "Duplicate original key"):
            self.normalize(docs=docs)

    def test_verdict_order_and_response_are_not_rewritten(self):
        for field, value in (("instruction_id_list", ["synthetic:b", "synthetic:a"]),
                             ("response", "Another response"),
                             ("follow_instruction_list", [True, 0])):
            strict = deepcopy(self.strict)
            strict[0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "alignment"):
                self.normalize(strict=strict)

    def test_inconsistent_original_prompt_boolean_is_rejected(self):
        strict = deepcopy(self.strict)
        strict[0]["follow_all_instructions"] = True
        with self.assertRaisesRegex(ValueError, "alignment"):
            self.normalize(strict=strict)

    def test_missing_original_aggregate_is_not_reconstructed(self):
        with self.assertRaisesRegex(ValueError, "aggregation"):
            original_metrics("prompt-level: 0.5\ninstruction-level: 0.7\n")


if __name__ == "__main__":
    unittest.main()
