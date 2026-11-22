import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from scripts.helpfulness import wildbench_pilot as w


class WildBenchPilotTests(unittest.TestCase):
    def fixtures(self):
        case = {"id": "a", "category": "Math", "task_groups": ["Math & Data Analysis"] * 2,
                "messages": [{"role": "user", "content": "Earlier"},
                             {"role": "assistant", "content": "Context"},
                             {"role": "user", "content": "Now"}], "checklist": ["Check the calculation"]}
        sample = {"case": case, "answer": "Four", "status": "generated",
                  "evaluation_id": "pilot-a", "split": "pilot", "calibration_prompt": "Bounded"}
        assets = {"files": {"evaluation/eval_template.score.v2.md":
                           "{$history}|{$user_query}|{$model_output}|{$checklist}"}}
        response = {"model": "open", "choices": [{"finish_reason": "stop", "message": {"content":
            '{"strengths":"Correct calculation.","weaknesses":"No material issue.","score":8}'}}]}
        return sample, assets, response

    def test_native_prompt_and_score_endpoints_are_preserved(self):
        sample, assets, response = self.fixtures()
        request = w.request(sample, assets, "open")
        self.assertEqual(request["messages"][-1]["content"],
                         "USER: Earlier\n\nASSISTANT: Context\n\n|Now|Four|- Check the calculation\n")
        for score, scaled in ((1, -80), (5, 0), (10, 100)):
            response["choices"][0]["message"]["content"] = json.dumps({
                "strengths": "Observed response.", "weaknesses": "Grounded limitation.", "score": score})
            row = w.judgment(sample, response, assets)
            self.assertEqual(row["scores"], {"score_1_10": score, "wb_score": scaled})

    def test_compact_failures_remain_failed_without_scores(self):
        sample, assets, response = self.fixtures()
        for score in (True, "8", 8.5, 0, 11, float("nan")):
            response["choices"][0]["message"]["content"] = json.dumps({
                "strengths": "Grounded.", "weaknesses": "Observed.", "score": score})
            row = w.judgment(sample, response, assets)
            self.assertEqual(row["status"], "failed")
            self.assertEqual(row["scores"], {})
        response["choices"][0]["message"]["content"] = json.dumps({
            "strengths": "x" * 33, "weaknesses": "Observed.", "score": 8})
        self.assertEqual(w.judgment(sample, response, assets)["status"], "failed")

    def test_replay_binds_schema_and_preserves_pending_request(self):
        sample, assets, response = self.fixtures()
        calls = [{"phase": "requested", "request": w.request(sample, assets, "open")},
                 {"phase": "returned", "response": response}]
        self.assertEqual(w.replay([sample], calls, assets, "open")[0][0]["status"], "succeeded")
        changed = copy.deepcopy(calls)
        del changed[0]["request"]["response_format"]
        with self.assertRaises(ValueError):
            w.replay([sample], changed, assets, "open")
        partial, report = w.replay([sample], calls[:1], assets, "open", partial=True)
        self.assertEqual(partial, [])
        self.assertTrue(report["unreturned_request"])
        self.assertFalse(report["complete_raw_evidence"])

    def test_controls_do_not_change_native_group_duplicate_weighting(self):
        sample, assets, response = self.fixtures()
        row = w.judgment(sample, response, assets)
        control = {**row, "split": "development", "scores": {"score_1_10": 1, "wb_score": -80}}
        result = w.summarize([row, control])
        self.assertEqual(result["metrics"]["wb_score"]["value"], 60)
        self.assertEqual(result["groups"]["Math & Data Analysis"]["wb_score"]["denominator"], 2)

    def test_audit_first_schema_keeps_historical_replay_strict(self):
        sample, assets, response = self.fixtures()
        response["choices"][0]["message"]["content"] = json.dumps({
            "strengths": " ".join(["Observed"] * 40), "weaknesses": "Grounded limitation.", "score": 6})
        self.assertEqual(w.judgment(sample, response, assets)["status"], "failed")
        historical = w.request(sample, assets, "open")
        sample["assessment_schema_version"] = 2
        current = w.request(sample, assets, "open")
        self.assertEqual(current["response_format"]["json_schema"]["schema"]["required"],
                         ["weaknesses", "strengths", "score"])
        self.assertEqual(w.judgment(sample, response, assets)["status"], "succeeded")
        calls = [{"phase": "requested", "request": historical},
                 {"phase": "returned", "response": response}]
        with self.assertRaises(ValueError):
            w.replay([sample], calls, assets, "open")
        response["choices"][0]["message"]["content"] = json.dumps({
            "strengths": " ".join(["Observed"] * 61), "weaknesses": "Grounded limitation.", "score": 6})
        self.assertEqual(w.judgment(sample, response, assets)["status"], "failed")

    def test_deployed_metadata_rejects_changed_local_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            index = root / "model.safetensors.index.json"
            config.write_bytes(b"{}")
            index.write_bytes(b"{}")
            for name in ("tokenizer_config.json", "chat_template.jinja", "tokenizer.json"):
                (root / name).write_bytes(b"{}")
            digest = hashlib.sha256(b"{}").hexdigest()
            identity = {"checkpoint_directory": directory, "public_repository_revision": "pinned",
                        "local_metadata_sha256": {"config.json": digest, "model.safetensors.index.json": digest},
                        "canonical_tokenizer": {"public_revision": "pinned", "files": {
                            name: {"sha256": digest, "bytes": 2} for name in
                            ("config.json", "tokenizer_config.json", "chat_template.jinja", "tokenizer.json")}}}
            self.assertFalse(w.deployed_metadata(identity, root)["full_deployed_weight_contents_verified"])
            config.write_bytes(b'{"changed":true}')
            with self.assertRaises(ValueError):
                w.deployed_metadata(identity, root)

    def test_generation_passes_all_history_to_native_checkpoint(self):
        sample, assets, _ = self.fixtures()
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(action="generate", output=Path(directory)/"output", checkpoint="checkpoint",
                                   base_url="http://localhost:8000/v1", model="target", ready_timeout=1,
                                   upstream="upstream")
            result = {"case": sample["case"], "answer": "Answer", "generation_info": {}}
            with (patch.object(w, "load_selection", return_value=[sample["case"]]),
                  patch.object(w.native, "staged_assets", return_value=assets),
                  patch.object(w.teacher_smoke, "preflight", return_value={}),
                  patch.object(w.teacher_smoke, "wait_for_server"),
                  patch.object(w.teacher_smoke, "native_model", return_value="model"),
                  patch.object(w.teacher_smoke, "generate_native", return_value=result) as generate,
                  patch.object(w, "outcome_status", return_value="generated")):
                w.run(args)
            self.assertEqual(generate.call_args.args[2], sample["case"]["messages"])
            self.assertEqual(generate.call_args.args[1], sample["case"])
            args.output = Path(directory)/"missing-reply"
            with (patch.object(w, "load_selection", return_value=[sample["case"]]),
                  patch.object(w.native, "staged_assets", return_value=assets),
                  patch.object(w.teacher_smoke, "preflight", return_value={}),
                  patch.object(w.teacher_smoke, "wait_for_server"),
                  patch.object(w.teacher_smoke, "native_model", return_value="model"),
                  patch.object(w.teacher_smoke, "generate_native", return_value=result),
                  patch.object(w, "outcome_status", return_value="model_no_final_reply")):
                report = w.run(args)
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["model_failure_cases"], 1)

    def test_judge_waits_for_declared_model_before_any_sdk_call(self):
        sample, assets, _ = self.fixtures()
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(action="judge", output=Path(directory)/"output", upstream="upstream",
                                   identity=Path(directory)/"identity.json",
                                   base_url="http://localhost:8000/v1", model="open", ready_timeout=1)
            with (patch.object(w, "load_selection", return_value=[sample["case"]]),
                  patch.object(w.native, "staged_assets", return_value=assets),
                  patch.object(w, "judge_preflight", return_value=([sample], {}, [])),
                  patch.object(w, "deployed_metadata", return_value={}),
                  patch.object(w.teacher_smoke, "wait_for_server", side_effect=TimeoutError("not ready")) as wait,
                  self.assertLogs(level="ERROR")):
                with self.assertRaises(TimeoutError):
                    w.run(args)
            wait.assert_called_once_with(args.base_url, args.model, 1)
            self.assertFalse((args.output/"judge-calls.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
