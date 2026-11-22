import math
from types import SimpleNamespace
import unittest

from scripts.helpfulness import eqbench_pilot as pilot


class EQBenchPilotTests(unittest.TestCase):
    def test_selection_is_deterministic_preserves_turns_and_native_families(self):
        C = SimpleNamespace(ANALYSIS_SCENARIO_IDS={"401", "402", "403", "404", "405", "406", "407", "408"},
                            MESSAGE_DRAFTING_SCENARIO_IDS={"208"})
        cases = {str(i): [f"First {i}", f"Second {i}"] for i in range(1, 13)}
        cases.update({sid: [sid] for sid in C.ANALYSIS_SCENARIO_IDS})
        cases["208"] = ["Draft", "Follow-up"]
        C.STANDARD_SCENARIO_PROMPTS_FILE = "unused"
        n = SimpleNamespace(C=C, parse_scenario_prompts=lambda _: cases)
        chosen = pilot.select(n)
        self.assertEqual(len(chosen), 20)
        self.assertEqual(chosen, pilot.select(n))
        self.assertEqual(sum(c["category"] == "message_drafting" for c in chosen), 1)
        for row in chosen:
            self.assertEqual(row["prompts"], cases[row["id"]])

    def test_missing_extra_nonfinite_and_out_of_range_scores_are_failures(self):
        result = {"empathy": 15, "correctness": 12}
        n = SimpleNamespace(Task=SimpleNamespace(_parse_rubric_scores=lambda _: result))
        sample = {"criteria": ["empathy", "correctness"]}
        response = {"choices": [{"finish_reason": "stop", "message": {"content": "native JSON"}}]}
        self.assertEqual(pilot.parse_scores(n, sample, response), result)
        for invalid in ({}, {"empathy": 15}, {"empathy": 15, "correctness": 12, "invented": 2},
                        {"empathy": math.nan, "correctness": 12}, {"empathy": -1, "correctness": 12},
                        {"empathy": 21, "correctness": 12}):
            result = invalid
            with self.assertRaises(ValueError):
                pilot.parse_scores(n, sample, response)
        response["choices"][0]["finish_reason"] = "length"
        with self.assertRaises(ValueError):
            pilot.parse_scores(n, sample, response)

    def test_judge_uses_the_full_frozen_native_prompt(self):
        prompt = "Native scenario and debrief\nFull rubric"
        request = pilot.request({"judge_prompt": prompt}, "open-judge")
        self.assertEqual(request["messages"], [{"role": "user", "content": prompt}])
        self.assertEqual(request["temperature"], 0)
        self.assertNotIn("min_p", request)


if __name__ == "__main__":
    unittest.main()
