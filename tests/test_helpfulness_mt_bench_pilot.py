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

    def generate(self, truncated=False):
        histories = []

        def answer(model, case, history, task, index, decoding, *, rendered_context):
            histories.append(deepcopy(history))
            self.assertEqual(json.loads(rendered_context), history)
            info = dict(self.info)
            info["generation_truncated"] = int(truncated and len(histories) == 1)
            return {"case": case, "answer": f"Answer {len(histories) - 1}", "generation_info": info}

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

    def judge(self, invalid=False):
        self.generate()
        self.args.samples = self.args.output / "samples.jsonl"
        self.args.samples_sha256 = hashlib.sha256(self.args.samples.read_bytes()).hexdigest()
        self.args.output = self.root / "judge"
        self.args.action = "judge"
        self.args.model = "gemma-fixture"
        metric = SimpleNamespace(
            _load_judge_prompts=lambda: {},
            _parse_score=lambda text: 8 if text == "[[8]]" else 4,
        )

        def grade(client, model, item, turn, prompts):
            return {
                "question_id": item["question_id"], "category": item["category"], "turn": turn,
                "judge_error": None, "score": float("nan") if invalid else (8 if turn == 1 else 4),
                "judgment": "[[8]]" if turn == 1 else "[[4]]",
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


if __name__ == "__main__":
    unittest.main()
