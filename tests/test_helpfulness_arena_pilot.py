from copy import deepcopy
import unittest

from scripts.helpfulness import arena_pilot as pilot


class ArenaPilotTests(unittest.TestCase):
    def test_selection_preserves_released_categories_and_open_references(self):
        rows = [
            {"uid": str(i), "prompt": "question", "category": category,
             "reference_answer": "reference", "reference_model": "Qwen/Qwen3-32B"}
            for category, offset, count in (("hard_prompt", 0, 500), ("creative_writing", 500, 250))
            for i in range(offset, offset + count)
        ]
        cases = pilot.select(rows)
        self.assertEqual(len(cases), 20)
        self.assertEqual(sum(case["category"] == "hard_prompt" for case in cases), 13)
        self.assertEqual(cases, pilot.select(list(reversed(rows))))
        rows[0]["reference_model"] = "closed-reference"
        with self.assertRaisesRegex(ValueError, "open Qwen"):
            pilot.select(rows)

    def test_terminal_preference_and_both_order_weighting(self):
        self.assertEqual(pilot.parse_preference("Reason. [[B>>A]]", "B"), 2)
        self.assertEqual(pilot.parse_preference("Reason. [[A>B]]", "A"), 1)
        self.assertEqual(pilot.battle_scores([2, 1]), {"win_credit": 4, "battle_weight": 4})
        self.assertEqual(pilot.battle_scores([-2, 0]), {"win_credit": 0.5, "battle_weight": 4})
        for raw in ("[[B>A]]", "Reason [[B>A]] more reasoning", "Reason [[A>B]] then [[B>A]]"):
            with self.assertRaises(ValueError):
                pilot.parse_preference(raw, "B")

    def test_invalid_order_is_not_credited_or_dropped(self):
        sample = {"case": {"id": "a", "category": "hard_prompt"}, "answer": "answer", "status": "generated"}
        responses = [
            {"choices": [{"finish_reason": "stop", "message": {"content": "Reason [[B>A]]"}}]},
            {"choices": [{"finish_reason": "length", "message": {"content": "Reason [[A>B]]"}}]},
        ]
        row = pilot.judgment(sample, responses)
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["scores"], {})
        result = pilot.summarize([row])
        self.assertEqual(result["selected_cases"], 20)
        self.assertIsNone(result["metrics"]["weighted_win_rate"]["value"])

    def test_replay_binds_messages_decoding_and_models(self):
        assets = {"files": {"utils/judge_utils.py":
                  'JUDGE_SETTINGS = {"hard_prompt": {"system_prompt": "Native rubric"}}'}}
        samples = [{"case": {"id": "a", "category": "hard_prompt", "question": "q",
                             "reference_answer": "r"}, "answer": "a", "status": "generated"}]
        calls = []
        for entry in pilot.judge_messages(samples, assets):
            content = "Reason [[B>>A]]" if entry["candidate_position"] == "B" else "Reason [[A>>B]]"
            calls.extend([
                {"phase": "requested", "request": {
                    "model": "open-model", "messages": entry["messages"],
                    "temperature": 0, "max_tokens": 2048, "seed": 1234}},
                {"phase": "returned", "response": {"model": "open-model",
                    "choices": [{"finish_reason": "stop", "message": {"content": content}}]}},
            ])
        rows, summary = pilot.replay(samples, calls, assets, "open-model")
        self.assertEqual(rows[0]["scores"], {"win_credit": 6, "battle_weight": 6})
        self.assertEqual(summary["position_exact_agreement"], 1)
        changed = deepcopy(calls)
        changed[0]["request"]["seed"] = 1
        with self.assertRaises(ValueError):
            pilot.replay(samples, changed, assets, "open-model")
        with self.assertRaises(ValueError):
            pilot.replay(samples, calls[:-1], assets, "open-model")
