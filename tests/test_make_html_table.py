import importlib.util
import sys
import types
import unittest
from unittest import mock

# The table only needs wandb to fetch runs; stub it for the import where it is missing.
with mock.patch.dict(sys.modules, {} if importlib.util.find_spec("wandb") else {"wandb": types.ModuleType("wandb")}):
    from scripts.results import make_html_table as table


class FailureRateTests(unittest.TestCase):
    def test_failure_rates_are_shown_but_never_scored(self) -> None:
        groups = [
            ("Accuracy", [("gsm8k/exact_match", "GSM8K")]),
            ("Truncated (%)", [("gsm8k/generation_truncated", "GSM8K")]),
        ]
        scores = {"m": {"gsm8k/exact_match": 0.8, "gsm8k/generation_truncated": 0.4}}
        group_data, overall = table._prepare_group_data(groups, ["m"], scores, lambda key: True, thinking=True)
        self.assertEqual([kind for *_, kind in group_data], ["rate", "failure"])
        self.assertEqual(overall, {"m": 0.8})
        cell = table._value_cell(0.4, 0, "failure", "")
        self.assertIn(">40.0<", cell)
        self.assertNotIn("data-val", cell)
        self.assertNotIn("score-row", table._row_classes("row", "failure"))


class RepeatHeadlineTests(unittest.TestCase):
    def test_oracle_scores_never_replace_headline_means(self):
        summary = {"mmlu_pro/exact_match_mean_repeats,none": .25,
                   "mmlu_pro/exact_match_pass@2,none": .5,
                   "code/pass@1_pass@2,none": 1.0}
        self.assertEqual(table.get_metric(summary, "mmlu_pro/exact_match"), .25)
        self.assertEqual(table.compute_mmlu_pro_detail(summary)["accuracy"], .25)
        self.assertIsNone(table.get_metric(summary, "code/pass@1"))
        self.assertIsNone(table.compute_degeneration_avg(summary))
        cell = table._value_cell(.25, 1, "rate", "", "mean@5")
        self.assertIn("mean@5", cell)
        self.assertIn('data-val="25.0"', cell)


if __name__ == "__main__":
    unittest.main()
