import unittest

from make_table import get_metric as get_metric_txt
from make_html_table import get_metric as get_metric_html


class MetricTaskMatchingTest(unittest.TestCase):
    """get_metric's fuzzy fallback must not borrow scores across task names.

    Regression tests for swiss-ai/evals-post-train#22: a summary holding only
    `humaneval_instruct` used to fill the `humaneval` row, because the fallback
    matched task names by substring.
    """

    def get_metrics(self):
        return [get_metric_txt, get_metric_html]

    def test_missing_task_returns_none_instead_of_borrowing(self):
        summary = {"humaneval_instruct/exact_match,ordered-extract": 0.62}
        for get_metric in self.get_metrics():
            self.assertIsNone(get_metric(summary, "humaneval/exact_match"))

    def test_overlapping_task_names_from_issue(self):
        # Every overlapping pair reported in the issue must stay independent.
        summary = {
            "hellaswag_multilingual/acc": 0.55,
            "gsm8k_platinum/exact_match": 0.30,
            "global_mmlu_gen_0shot/exact_match": 0.44,
            "switzerland_qa_0shot/exact_match": 0.71,
            "minerva_math500/exact_match": 0.18,
        }
        for get_metric in self.get_metrics():
            self.assertIsNone(get_metric(summary, "hellaswag/acc"))
            self.assertIsNone(get_metric(summary, "gsm8k/exact_match"))
            self.assertIsNone(get_metric(summary, "global_mmlu/exact_match"))
            self.assertIsNone(get_metric(summary, "switzerland_qa/exact_match"))
            self.assertIsNone(get_metric(summary, "minerva_math/exact_match"))

    def test_exact_match_still_wins(self):
        summary = {
            "humaneval/exact_match": 0.41,
            "humaneval_instruct/exact_match,ordered-extract": 0.62,
        }
        for get_metric in self.get_metrics():
            self.assertEqual(get_metric(summary, "humaneval/exact_match"), 0.41)
            self.assertEqual(get_metric(summary, "humaneval_instruct/exact_match"), 0.62)

    def test_fuzzy_match_across_filter_suffix_still_works(self):
        # The fallback exists because main-table keys omit the filter suffix;
        # a request without it must still resolve the stored key that has it.
        summary = {"agieval/acc,none": 0.33}
        for get_metric in self.get_metrics():
            self.assertEqual(get_metric(summary, "agieval/acc"), 0.33)

    def test_filter_is_still_enforced(self):
        summary = {
            "gpqa_main_cot_zeroshot/exact_match,ordered-extract": 0.40,
            "gpqa_main_cot_zeroshot/exact_match,flexible-extract": 0.45,
        }
        for get_metric in self.get_metrics():
            self.assertEqual(
                get_metric(summary, "gpqa_main_cot_zeroshot/exact_match,flexible"),
                0.45,
            )
            self.assertIsNone(
                get_metric(summary, "gpqa_main_cot_zeroshot/exact_match,nonexistent")
            )

    def test_task_match_is_case_insensitive(self):
        summary = {"Hellaswag/acc": 0.60}
        for get_metric in self.get_metrics():
            self.assertEqual(get_metric(summary, "hellaswag/acc"), 0.60)

    def test_gsm8k_strict_match_preference(self):
        summary = {
            "gsm8k/exact_match,flexible-extract": 0.70,
            "gsm8k/exact_match,strict-match": 0.68,
        }
        for get_metric in self.get_metrics():
            self.assertEqual(get_metric(summary, "gsm8k/exact_match"), 0.68)

    def test_mgsm_aggregate_unchanged(self):
        summary = {
            "mgsm_de/exact_match": 0.50,
            "mgsm_fr/exact_match": 0.70,
        }
        for get_metric in self.get_metrics():
            self.assertEqual(get_metric(summary, "mgsm/exact_match"), 0.60)


if __name__ == "__main__":
    unittest.main()
