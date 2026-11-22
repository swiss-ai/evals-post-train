import math
import copy
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

    def test_raw_replay_rejects_changed_requests_missing_calls_and_response_models(self):
        n = SimpleNamespace(Task=SimpleNamespace(_parse_rubric_scores=lambda _: {"correctness": 12}),
                            calculate_final_rubric_score=lambda _: (12, None))
        sample = {"case": {"id": "401"}, "status": "generated", "criteria": ["correctness"],
                  "judge_prompt": "Exact frozen native prompt", "task": {}}
        response = {"model": "open-judge", "choices": [
            {"finish_reason": "stop", "message": {"content": '{"correctness": 12}'}}]}
        calls = [{"phase": "requested", "request": pilot.request(sample, "open-judge")},
                 {"phase": "returned", "response": response}]
        rows, report = pilot.replay(n, [sample], calls, "open-judge")
        self.assertEqual(rows[0]["status"], "succeeded")
        self.assertFalse(report["automatic_scoring_ready"])
        for field, value in (("temperature", 1), ("max_tokens", 32),
                             ("messages", [{"role": "user", "content": "changed"}])):
            changed = copy.deepcopy(calls)
            changed[0]["request"][field] = value
            with self.assertRaises(ValueError):
                pilot.replay(n, [sample], changed, "open-judge")
        with self.assertRaises(ValueError):
            pilot.replay(n, [sample], calls[:1], "open-judge")
        calls[1]["response"]["model"] = "different-judge"
        with self.assertRaises(ValueError):
            pilot.replay(n, [sample], calls, "open-judge")

    def test_failed_native_scenario_has_no_invented_judge_credit(self):
        row = pilot.judgment(None, {"case": {"id": "3"}, "status": "model_no_final_reply"}, None)
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["scores"], {})
        self.assertNotIn("native_rubric_score", row)


if __name__ == "__main__":
    unittest.main()
