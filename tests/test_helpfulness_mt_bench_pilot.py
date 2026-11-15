from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.helpfulness import mt_bench_pilot as pilot
from scripts.helpfulness.core import append_jsonl, write_json
from scripts.helpfulness.teacher_smoke import FAILURE_FLAGS


class ScriptedTask:
    config = SimpleNamespace(generation_kwargs={"max_gen_toks": 1024})

    def init_multiturn_state(self, doc, ctx, decoding, **kwargs):
        return {"doc": doc, "history": [], "responses": [], "gen_kwargs": decoding}

    def multiturn_is_done(self, state):
        return len(state["responses"]) == 2

    def multiturn_next_request(self, state):
        state["history"].append({"role": "user", "content": state["doc"]["turns"][len(state["responses"])]})
        return json.dumps(state["history"]), deepcopy(state["gen_kwargs"])

    def multiturn_consume_response(self, state, answer):
        state["responses"].append(answer)
        state["history"].append({"role": "assistant", "content": answer})

    def multiturn_result(self, state):
        return {"responses": state["responses"], "history": state["history"]}


class MTBenchPilotTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        docs = [
            {"question_id": index * 3 + j, "category": category, "turns": [f"First {index}-{j}", f"Follow-up {index}-{j}"]}
            for index, category in enumerate(pilot.CATEGORIES) for j in range(3)
        ]
        docs.append({"question_id": 100, "category": "coding", "turns": ["Excluded", "Excluded"]})
        refs = [{"question_id": doc["question_id"], "model_id": "gpt-4",
                 "choices": [{"index": 0, "turns": ["Reference one", "Reference two"]}]}
                for doc in docs if doc["category"] in ("reasoning", "math")]
        source, references, selection = (self.root / name for name in ("source.jsonl", "refs.jsonl", "selection.json"))
        for doc in docs:
            append_jsonl(source, doc)
        for ref in refs:
            append_jsonl(references, ref)
        for name, path in (("SOURCE_SHA256", source), ("REFERENCES_SHA256", references)):
            patcher = patch.object(pilot, name, hashlib.sha256(path.read_bytes()).hexdigest())
            patcher.start()
            self.addCleanup(patcher.stop)
        write_json(selection, {
            "benchmark": "mt_bench_noncode", "source_sha256": pilot.SOURCE_SHA256,
            "references_sha256": pilot.REFERENCES_SHA256,
            "question_ids": [doc["question_id"] for doc in pilot.select_documents(docs)],
        })
        self.args = SimpleNamespace(
            source=source, references=references, selection=selection,
            selection_sha256=hashlib.sha256(selection.read_bytes()).hexdigest(),
            action="generate", output=self.root / "generated", checkpoint=self.root,
            max_tokens=1024, max_model_len=16384, ready_timeout=1,
            model="chonk-fixture", base_url="http://127.0.0.1:8000/v1", allow_truncated=True,
        )
        self.info = {flag: 0 for flag in FAILURE_FLAGS}
        self.info["apertus2_output"] = "Synthetic native evidence, not checkpoint inference"
        self.cases = pilot.load_selection(self.args)

    def generate(self, truncated=False, no_final=False, request_failed=False, claim=False):
        histories = []

        def answer(model, case, history, task, index, decoding, *, rendered_context):
            histories.append(deepcopy(history))
            self.assertEqual(json.loads(rendered_context), history)
            info = dict(self.info)
            info["generation_truncated"] = int(truncated and len(histories) == 1)
            info["generation_has_claim"] = int(claim and len(histories) == 1)
            text = f"Answer {len(histories) - 1}"
            if no_final and len(histories) == 1:
                text = ""
                info.update(generation_truncated=1, generation_think_truncated=1,
                            generation_request_failed=int(request_failed))
            return {"case": case, "answer": text, "generation_info": info}

        with (
            patch.object(pilot, "preflight", return_value=({}, ScriptedTask(), None)),
            patch.object(pilot.teacher_smoke, "wait_for_server"),
            patch.object(pilot.teacher_smoke, "native_model"),
            patch.object(pilot.teacher_smoke, "generate_native", side_effect=answer) as generated,
        ):
            result = pilot.run(self.args)
        return result, generated, histories

    def test_stratified_selection_and_math_references_are_preserved(self):
        counts = {category: sum(case["category"] == category for case in self.cases) for category in pilot.CATEGORIES}
        self.assertEqual(sorted(counts.values()), [2, 3, 3, 3, 3, 3, 3])
        self.assertTrue(all(len(case["reference"]) == 2 for case in self.cases if case["category"] in ("math", "reasoning")))
        original = self.args.references.read_text(encoding="utf-8")
        self.args.references.write_text(original.splitlines()[0] + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum"):
            pilot.load_selection(self.args)

    def test_turn_two_receives_actual_first_answer_and_bounded_truncations_continue(self):
        result, calls, histories = self.generate(truncated=True)
        self.assertEqual(calls.call_count, 40)
        self.assertEqual([call.args[4] for call in calls.call_args_list[:2]], [0, 0])
        self.assertEqual(histories[1][-2], {"role": "assistant", "content": "Answer 0"})
        self.assertEqual(histories[1][-1]["content"], self.cases[0]["turns"][1])
        self.assertEqual(result["completed_conversations"], 20)
        self.assertEqual(result["completed_turns"], 40)
        self.assertEqual(result["truncated_turns"], 1)
        self.args.samples = self.args.output / "samples.jsonl"
        self.args.samples_sha256 = hashlib.sha256(self.args.samples.read_bytes()).hexdigest()
        rows = pilot.load_samples(self.args, self.cases)
        self.assertEqual(rows[0]["status"], "generated_truncated")
        rows[0]["history"][1]["content"] = "Not the first answer"
        self.args.samples.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        self.args.samples_sha256 = hashlib.sha256(self.args.samples.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "history"):
            pilot.load_samples(self.args, self.cases)

    def judge(self, invalid=False, no_final=False):
        self.generate(no_final=no_final)
        self.args.samples = self.args.output / "samples.jsonl"
        self.args.samples_sha256 = hashlib.sha256(self.args.samples.read_bytes()).hexdigest()
        self.args.output = self.root / "judge"
        self.args.action = "judge"
        self.args.model = "gemma-fixture"
        metric = SimpleNamespace(
            _load_judge_prompts=lambda: {},
            _parse_score=lambda text: float(text.removeprefix("[[").removesuffix("]]")),
        )

        def grade(client, model, item, turn, prompts):
            empty = no_final and item["question_id"] == self.cases[0]["question_id"] and turn == 1
            if empty:
                self.assertEqual(item["responses"][0], "")
            score = 1 if empty else 8 if turn == 1 else 4
            return {
                "question_id": item["question_id"], "category": item["category"], "turn": turn,
                "judge_error": None, "score": float("nan") if invalid else score,
                "judgment": f"[[{score}]]",
            }

        metric._judge_one = grade
        with (
            patch.object(pilot, "preflight", return_value=({"judge_messages": []}, None, metric)),
            patch.object(pilot.teacher_smoke, "wait_for_server"),
            patch.object(pilot, "record_sdk_calls", return_value=nullcontext()),
            patch.dict("sys.modules", {
                "openai": SimpleNamespace(OpenAI=lambda **kwargs: nullcontext(object())),
                "lm_eval.api.rate_limiter": SimpleNamespace(JUDGE_REQUESTS_PER_MINUTE_ENV="FIXTURE_RATE"),
            }),
        ):
            return pilot.run(self.args)

    def test_judging_reports_separate_turns_and_never_averages_away_bad_judgments(self):
        result = self.judge()
        self.assertEqual(result["metrics"], {"turn_1": 8, "turn_2": 4, "score": 6})
        self.assertEqual(result["denominators"]["score"], 40)
        self.assertFalse(result["target_regenerated"])
        self.args.output = self.root / "fresh-generated"
        self.args.action = "generate"
        self.args.model = "chonk-fixture"
        original_judge = self.root / "judge"
        original_judge.rename(self.root / "successful-judge")
        with self.assertRaisesRegex(ValueError, "forty valid native"):
            self.judge(invalid=True)
        report = pilot.read_json(self.root / "judge/report.json")
        self.assertEqual(report["status"], "failed")
        self.assertNotIn("metrics", report)
        self.assertEqual(len(pilot.read_jsonl(self.root / "judge/judgments.jsonl")), 40)

    def test_native_helper_preserves_rendered_context_without_registering_it_twice(self):
        requests = []

        def instance(**kwargs):
            requests.append(kwargs)
            return SimpleNamespace(length_info=[self.info])

        with patch.dict("sys.modules", {"lm_eval.api.instance": SimpleNamespace(Instance=instance)}):
            for rendered in (None, [{"role": "user", "content": "Already rendered native context"}]):
                model = Mock()
                model.apply_chat_template.return_value = "Default rendered context"
                model.generate_until.return_value = ["Native final answer"]
                result = pilot.teacher_smoke.generate_native(
                    model, self.cases[0], [], "fixture", 0, {}, rendered_context=rendered,
                )
                self.assertEqual(model.apply_chat_template.call_count, 1 if rendered is None else 0)
                self.assertEqual(requests[-1]["arguments"][0],
                                 "Default rendered context" if rendered is None else rendered)
                self.assertEqual(result["answer"], "Native final answer")

    def test_hash_bound_resume_keeps_nine_conversations_and_only_generates_missing_turns(self):
        self.generate()
        original = self.args.output
        samples = pilot.read_jsonl(original / "samples.jsonl")[:9]
        evidence = pilot.read_jsonl(original / "generation-evidence.jsonl")[:19]
        for name, rows in (("saved.jsonl", samples), ("evidence.jsonl", evidence)):
            for row in rows:
                append_jsonl(self.root / name, row)
        report = pilot.read_json(original / "report.json")
        report.update(status="failed", completed_conversations=9, completed_turns=18)
        write_json(self.root / "failed-report.json", report)
        for name, filename in (("samples", "saved.jsonl"), ("report", "failed-report.json"),
                               ("evidence", "evidence.jsonl")):
            path = self.root / filename
            setattr(self.args, "resume_" + name, path)
            setattr(self.args, "resume_" + name + "_sha256", hashlib.sha256(path.read_bytes()).hexdigest())
        self.args.output, self.args.max_tokens = self.root / "resumed", 2048
        result, calls, _ = self.generate()
        self.assertEqual(calls.call_count, 22)
        self.assertEqual(calls.call_args_list[0].args[4], 9)
        self.assertTrue(all(call.args[5]["max_gen_toks"] == 2048 for call in calls.call_args_list))
        self.assertEqual(result["resumed_conversations"], 9)
        self.assertEqual(result["completed_turns"], 40)
        self.assertEqual(result["new_generation_attempts"], 22)
        self.assertTrue((self.args.output / "samples.jsonl").read_bytes().startswith(self.args.resume_samples.read_bytes()))
        self.assertEqual(pilot.read_jsonl(self.args.output / "generation-evidence.jsonl")[:18], evidence[:18])
        report["model"] = "different-checkpoint"
        write_json(self.args.resume_report, report)
        self.args.resume_report_sha256 = hashlib.sha256(self.args.resume_report.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "model or native preflight"):
            pilot.resume_data(self.args, self.cases, {})

    def test_opt_in_finishes_requests_but_blocks_grading_for_missing_final_reply(self):
        self.args.continue_model_failures = True
        with self.assertRaisesRegex(ValueError, "missing final replies"):
            self.generate(no_final=True, claim=True)
        report = pilot.read_json(self.args.output / "report.json")
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["attempted_turns"], 40)
        self.assertEqual(report["completed_turns"], 39)
        self.assertEqual(report["completed_conversations"], 19)
        self.assertEqual(report["no_final_reply_turns"], 1)
        self.assertEqual(report["auxiliary_claim_turns"], 1)
        self.assertNotIn("metrics", report)
        rows = pilot.read_jsonl(self.args.output / "samples.jsonl")
        self.assertEqual(rows[0]["status"], "model_no_final_reply")
        self.assertEqual(rows[0]["answers"][0], "")
        self.args.output = self.root / "transport-failed"
        with self.assertRaisesRegex(ValueError, "no final reply"):
            self.generate(no_final=True, request_failed=True)
        self.assertEqual(len(pilot.read_jsonl(self.args.output / "generation-evidence.jsonl")), 1)

    def test_native_auxiliary_claim_is_retained_without_rejecting_parsed_reply(self):
        result, calls, _ = self.generate(claim=True)
        self.assertEqual(calls.call_count, 40)
        self.assertEqual(result["auxiliary_claim_turns"], 1)
        self.args.samples = self.args.output / "samples.jsonl"
        self.args.samples_sha256 = hashlib.sha256(self.args.samples.read_bytes()).hexdigest()
        rows = pilot.load_samples(self.args, self.cases)
        info = rows[0]["generation_info"][0]
        self.assertEqual(info["generation_has_claim"], 1)
        self.assertEqual(rows[0]["answers"][0], "Answer 0")
        with self.assertRaisesRegex(ValueError, "generation_has_claim"):
            pilot.teacher_smoke.validate_generation("Answer 0", info)
        invalid = {**info, "generation_parse_failed": 1}
        with self.assertRaisesRegex(ValueError, "generation_parse_failed"):
            pilot.teacher_smoke.validate_generation("Answer 0", invalid, allow_claim=True)

    def test_explicit_empty_reply_policy_grades_actual_blank_without_imputing_score(self):
        self.args.allow_no_final = True
        result = self.judge(no_final=True)
        generation = pilot.read_json(self.root / "generated/report.json")
        self.assertEqual(generation["status"], "succeeded_with_model_failures")
        self.assertEqual(generation["recorded_conversations"], 20)
        self.assertEqual(generation["completed_turns"], 39)
        self.assertEqual(result["no_final_reply_turns"], 1)
        self.assertEqual(result["completed_turns"], 40)
        self.assertEqual(result["denominators"]["score"], 40)
        self.assertAlmostEqual(result["metrics"]["turn_1"], 7.65)
        rows = pilot.read_jsonl(self.args.samples)
        self.assertEqual(rows[0]["answers"][0], "")
        self.assertEqual(rows[0]["status"], "model_no_final_reply")
        self.args.allow_no_final = False
        with self.assertRaisesRegex(ValueError, "no final reply"):
            pilot.load_samples(self.args, self.cases)


if __name__ == "__main__":
    unittest.main()
