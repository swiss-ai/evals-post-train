import math
import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile

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

    def test_constructed_controls_do_not_replace_checkpoint_aggregation(self):
        received = []
        n = SimpleNamespace(calculate_final_rubric_score=lambda data: (received.append(data) or 12, None))
        samples = [
            {"case": {"id": "401"}, "task": {}, "split": "pilot"},
            {"case": {"id": "401"}, "task": {}, "split": "development"},
        ]
        rows = [{"status": "succeeded", "scores": {"correctness": 12}},
                {"status": "succeeded", "scores": {"correctness": 1}}]
        report = pilot.summarize(n, samples, rows)
        self.assertEqual(received[0]["scenario_tasks"]["1"]["401"]["rubric_scores"], {"correctness": 12})
        self.assertEqual(report["scored_cases"], 1)
        self.assertEqual(report["valid_items"], 2)

    def test_calibration_is_explicit_and_does_not_change_native_user_prompt(self):
        request = pilot.request({"judge_prompt": "native rubric", "calibration_prompt": "calibration"}, "open")
        self.assertEqual(request["messages"], [
            {"role": "system", "content": "calibration"},
            {"role": "user", "content": "native rubric"},
        ])

    def test_structured_output_preserves_native_criteria_and_is_bound_in_raw_replay(self):
        sample = {"case": {"id": "401"}, "status": "generated", "task": {},
                  "judge_prompt": "unchanged native rubric", "criteria": ["correctness"],
                  "calibration_prompt": "compact profile", "structured_output": True}
        request = pilot.request(sample, "open")
        schema = request["response_format"]["json_schema"]["schema"]
        self.assertEqual(set(schema["required"]), {"correctness", "chain_of_thought_reasoning"})
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["correctness"], {
            "type": "number", "minimum": 0, "maximum": 20})
        self.assertEqual(request["messages"][-1]["content"], sample["judge_prompt"])
        n = SimpleNamespace(Task=SimpleNamespace(_parse_rubric_scores=lambda _: {"correctness": 12}),
                            calculate_final_rubric_score=lambda _: (12, None))
        response = {"model": "open", "choices": [{"finish_reason": "stop", "message": {
            "content": '{"correctness":12,"chain_of_thought_reasoning":"Grounded partial insight."}'}}]}
        calls = [{"phase": "requested", "request": request},
                 {"phase": "returned", "response": response}]
        self.assertEqual(pilot.replay(n, [sample], calls, "open")[0][0]["status"], "succeeded")
        del calls[0]["request"]["response_format"]
        with self.assertRaises(ValueError):
            pilot.replay(n, [sample], calls, "open")

    def test_judge_preflight_rejects_changed_native_criteria_and_source_identity(self):
        from scripts.helpfulness.core import write_json, append_jsonl
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = {"case": {"id": "401"}, "status": "generated", "task": {},
                      "criteria": ["correctness"], "judge_prompt": "native"}
            append_jsonl(root / "samples.jsonl", sample)
            sha = hashlib.sha256((root / "samples.jsonl").read_bytes()).hexdigest()
            identity = {"served_model": "gemma-local", "base_url": "http://127.0.0.1:8000/v1",
                        "source_samples_sha256": sha, "selection_sha256": "selection",
                        "benchmark": "eqbench3", "license": "Apache-2.0",
                        "public_model": "google/gemma-4-31B-it", "public_repository_revision": "pinned"}
            write_json(root / "identity.json", identity)
            args = SimpleNamespace(samples=root / "samples.jsonl", samples_sha256=sha,
                                   identity=root / "identity.json", selection_sha256="selection",
                                   model="gemma-local", base_url=identity["base_url"])
            with patch.object(pilot, "criteria", return_value=["correctness"]), patch.object(
                pilot, "rubric", return_value="native"
            ):
                samples, _, plans = pilot.judge_preflight(args, None, [sample["case"]])
                self.assertEqual(len(plans), 1)
                self.assertEqual(samples[0]["split"], "pilot")
                args.structured_output = True
                with self.assertRaises(ValueError):
                    pilot.judge_preflight(args, None, [sample["case"]])
                args.structured_output = False
                identity["source_samples_sha256"] = "changed"
                write_json(root / "identity.json", identity)
                with self.assertRaises(ValueError):
                    pilot.judge_preflight(args, None, [sample["case"]])
            with patch.object(pilot, "criteria", return_value=["missing"]):
                with self.assertRaises(ValueError):
                    pilot.judge_preflight(args, None, [sample["case"]])

    def test_partial_replay_preserves_pending_request_without_credit(self):
        n = SimpleNamespace(Task=SimpleNamespace(_parse_rubric_scores=lambda _: {"correctness": 12}),
                            calculate_final_rubric_score=lambda _: (12, None))
        first = {"case": {"id": "401"}, "status": "generated", "criteria": ["correctness"],
                 "judge_prompt": "first", "task": {}}
        second = {**first, "case": {"id": "402"}, "judge_prompt": "second"}
        response = {"model": "open", "choices": [
            {"finish_reason": "stop", "message": {"content": '{"correctness":12}'}}]}
        calls = [{"phase": "requested", "request": pilot.request(first, "open")},
                 {"phase": "returned", "response": response},
                 {"phase": "requested", "request": pilot.request(second, "open")}]
        rows, result = pilot.replay_partial(n, [first, second], calls, "open")
        self.assertEqual(len(rows), 1)
        self.assertEqual(result["status"], "incomplete")
        self.assertTrue(result["unreturned_request"])
        self.assertFalse(result["complete_raw_evidence"])
        calls[-1]["request"]["messages"] = []
        with self.assertRaises(ValueError):
            pilot.replay_partial(n, [first, second], calls, "open")

    def test_compact_profile_enforces_real_json_numeric_types_and_assessment_budget(self):
        n = SimpleNamespace(Task=SimpleNamespace(_parse_rubric_scores=lambda _: {"correctness": 12}))
        sample = {"criteria": ["correctness"], "calibration_prompt": "compact profile"}
        raw = {"correctness": 12, "chain_of_thought_reasoning": "grounded " * 60}
        response = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(raw)}}]}
        self.assertEqual(pilot.parse_scores(n, sample, response), {"correctness": 12})
        for changed in ({**raw, "chain_of_thought_reasoning": "grounded " * 61},
                        {**raw, "correctness": True}, {**raw, "correctness": "12"},
                        {**raw, "unexpected": 12}, {**raw, "chain_of_thought_reasoning": ""}):
            response["choices"][0]["message"]["content"] = json.dumps(changed)
            with self.assertRaises(ValueError):
                pilot.parse_scores(n, sample, response)


if __name__ == "__main__":
    unittest.main()
