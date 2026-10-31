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


if __name__ == "__main__":
    unittest.main()
