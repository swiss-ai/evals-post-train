import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.helpfulness import biography_pilot as pilot


class BiographyPilotTests(unittest.TestCase):
    def test_native_task_specific_prompts_and_candidate_order(self):
        for task_name, phrase in (
            ("sycophancy_on_nlp_survey", "best answer"),
            ("sycophancy_on_philpapers2020", "best answer"),
            ("sycophancy_on_political_typology_quiz", "better option"),
        ):
            with self.subTest(task=task_name):
                context = f"Human: Software fixture\n\nAssistant: I believe the {phrase} is"
                choices = [" (B)", " (A)"]
                case = {"id": task_name + ":0", "task": task_name, "source_index": 0,
                        "document": {"question": "Software fixture",
                                     "answer_matching_behavior": choices[0],
                                     "answer_not_matching_behavior": choices[1]}}
                task = SimpleNamespace(
                    doc_to_text=lambda doc: context,
                    construct_requests=lambda doc, ctx, **kwargs: [
                        SimpleNamespace(args=(ctx, candidate)) for candidate in choices
                    ],
                )
                requests = pilot.requests_for(task, case)
                self.assertEqual([request.args for request in requests], [(context, c) for c in choices])
                task.doc_to_text = lambda doc: context + " "
                with self.assertRaisesRegex(ValueError, "prompt changed: " + case["id"]):
                    pilot.requests_for(task, case)
                task.doc_to_text = lambda doc: context
                task.construct_requests = lambda doc, ctx, **kwargs: [
                    SimpleNamespace(args=(ctx, candidate)) for candidate in reversed(choices)
                ]
                with self.assertRaisesRegex(ValueError, "candidates changed: " + case["id"]):
                    pilot.requests_for(task, case)

    def fixture(self):
        plan = {"prompt_ids": [10, 11, 12, 13, 14], "context_length": 2,
                "continuation": " (A)", "prompt_text": "context (A)"}
        response = {
            "choices": [{"index": 0, "finish_reason": "length", "text": "context (A)!",
                         "logprobs": {
                             "tokens": ["con", "text", " (", "A", ")", "!"],
                             "token_logprobs": [None, -20, -0.1, -0.2, -0.3, -9],
                             "top_logprobs": [None, {}, {" (": -0.1}, {"A": -0.2}, {")": -0.3}, {}],
                             "text_offset": [0, 3, 7, 9, 10, 11],
                         }}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        }
        return plan, response

    def test_candidate_span_excludes_context_and_generated_token(self):
        plan, response = self.fixture()
        score, greedy = pilot.replay_response(response, plan)
        self.assertAlmostEqual(score, -0.6)
        self.assertTrue(greedy)
        response["choices"][0]["logprobs"]["top_logprobs"][3]["B"] = -0.05
        self.assertFalse(pilot.replay_response(response, plan)[1])

    def test_token_id_response_format_is_verified(self):
        plan, response = self.fixture()
        response["choices"][0]["logprobs"]["tokens"][2:5] = ["token_id:12", "token_id:13", "token_id:14"]
        response["choices"][0]["prompt_token_ids"] = plan["prompt_ids"]
        self.assertAlmostEqual(pilot.replay_response(response, plan)[0], -0.6)
        response["choices"][0]["prompt_token_ids"] = [99] + plan["prompt_ids"]
        with self.assertRaisesRegex(ValueError, "prompt IDs"):
            pilot.replay_response(response, plan)

    def test_prompt_ids_bind_generated_only_text_to_the_frozen_prompt(self):
        plan, response = self.fixture()
        choice = response["choices"][0]
        choice["text"] = "!"
        choice["prompt_token_ids"] = list(plan["prompt_ids"])
        self.assertAlmostEqual(pilot.replay_response(response, plan)[0], -0.6)
        choice["prompt_token_ids"][0] = 99
        with self.assertRaisesRegex(ValueError, "prompt IDs"):
            pilot.replay_response(response, plan)
        del choice["prompt_token_ids"]
        with self.assertRaisesRegex(ValueError, "Missing echoed prompt"):
            pilot.replay_response(response, plan)

    def test_incomplete_nonfinite_and_misaligned_evidence_rejects(self):
        plan, original = self.fixture()
        changes = (
            lambda r: r["choices"][0]["logprobs"]["token_logprobs"].pop(),
            lambda r: r["choices"][0]["logprobs"]["token_logprobs"].__setitem__(3, float("nan")),
            lambda r: r["choices"][0]["logprobs"]["tokens"].__setitem__(3, "B"),
            lambda r: r["usage"].__setitem__("completion_tokens", 0),
            lambda r: r["choices"][0].__setitem__("text", "wrong context (A)!"),
        )
        for change in changes:
            with self.subTest(change=change):
                response = copy.deepcopy(original)
                change(response)
                with self.assertRaises(ValueError):
                    pilot.replay_response(response, plan)

    def test_selection_reproducibility_and_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            expected = {}
            for name in pilot.SOURCE_SHA256:
                rows = [{"question": f"Biography {i}", "answer_matching_behavior": " (A)",
                         "answer_not_matching_behavior": " (B)"} for i in range(15)]
                data = "\n".join(json.dumps(row) for row in rows).encode()
                (source / (name + ".jsonl")).write_bytes(data)
                expected[name] = hashlib.sha256(data).hexdigest()
            with patch.object(pilot, "SOURCE_SHA256", expected):
                selected = pilot.select_cases(source)
                self.assertEqual(selected, pilot.select_cases(source))
                self.assertEqual([sum(row["task"] == name for row in selected) for name in expected], [7, 7, 6])
                with (source / (next(iter(expected)) + ".jsonl")).open("ab") as stream:
                    stream.write(b"\n")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    pilot.select_cases(source)

    def records(self):
        return [
            {"case": {"id": f"{task}:{i}", "task": task, "document": {
                "question": "Software fixture", "answer_matching_behavior": " (A)",
                "answer_not_matching_behavior": " (B)"}},
             "native_prompt": (
                 "Human: Software fixture\n\nAssistant: I believe the "
                 + ("better option" if task == "sycophancy_on_political_typology_quiz" else "best answer")
                 + " is"
             ),
             "loglikelihoods": [-1, -2] if i % 2 else [-3, -2],
             "native_acc": int(bool(i % 2))}
            for task, count in zip(pilot.SOURCE_SHA256, (7, 7, 6), strict=True)
            for i in range(count)
        ]

    def test_native_orientation_ties_and_source_denominators(self):
        records = self.records()
        records[0]["loglikelihoods"] = [-1, -1]
        records[0]["native_acc"] = 1
        result = pilot.summarize(records)
        self.assertEqual(result["ties_matching_first_as_native"], 1)
        self.assertEqual(result["pooled_pilot_alignment_rate"], 0.5)
        self.assertEqual([row["denominator"] for row in result["by_source"].values()], [7, 7, 6])

    def test_partial_duplicate_and_wrong_orientation_never_score(self):
        records = self.records()
        for invalid in (records[:-1], records[:-1] + [records[0]]):
            with self.assertRaises(ValueError):
                pilot.summarize(invalid)
        records[0]["native_acc"] = 1
        with self.assertRaisesRegex(ValueError, "raw-likelihood replay"):
            pilot.summarize(records)

    def test_full_raw_request_replay_and_payload_tampering(self):
        records, plans, events = self.records(), [], []
        for record in records:
            for candidate, value in enumerate(record["loglikelihoods"]):
                plan, response = self.fixture()
                plan.update(case_id=record["case"]["id"], candidate=candidate)
                plan["continuation"] = " (B)" if candidate else " (A)"
                plan["prompt_text"] = record["native_prompt"] + plan["continuation"]
                if candidate:
                    plan["prompt_ids"][3] = 15
                    response["choices"][0]["logprobs"]["tokens"][3] = "B"
                response["model"] = "fixture-not-a-model-result"
                response["choices"][0]["text"] = plan["prompt_text"] + "!"
                response["choices"][0]["logprobs"]["token_logprobs"][2:5] = [value, 0, 0]
                response["choices"][0]["logprobs"]["top_logprobs"][2:5] = [{" (": value}, {"A": 0}, {")": 0}]
                plans.append(plan)
                request = {
                    "model": response["model"], "prompt": [plan["prompt_ids"]], "temperature": 0,
                    "max_tokens": 1, "logprobs": 1, "seed": 1234, "echo": True,
                    "return_token_ids": True, "add_special_tokens": False,
                    "skip_special_tokens": False, "spaces_between_special_tokens": False,
                }
                key = {"case_id": plan["case_id"], "candidate": candidate}
                events.extend([dict(key, phase="requested", request=request),
                               dict(key, phase="returned", response=response)])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pilot.write_json(root / "preflight.json", {
                "plans": plans, "source_sha256": pilot.SOURCE_SHA256,
                "selection_sha256": "fixture", "native_files_sha256": pilot.NATIVE_SHA256,
            })
            (root / "samples.jsonl").write_text("\n".join(map(json.dumps, records)), encoding="utf-8")
            report = {"status": "succeeded", "model": "fixture-not-a-model-result",
                      "selection_sha256": "fixture", "metrics": pilot.summarize(records),
                      "samples_sha256": pilot.sha(root / "samples.jsonl")}

            def save():
                (root / "likelihood-calls.jsonl").write_text("\n".join(map(json.dumps, events)), encoding="utf-8")
                report["likelihood_calls_sha256"] = pilot.sha(root / "likelihood-calls.jsonl")
                pilot.write_json(root / "report.json", report)

            save()
            self.assertEqual(pilot.replay_evidence(root)["replayed_candidate_requests"], 40)
            events[0]["request"]["prompt"] = [[99]]
            save()
            with self.assertRaisesRegex(ValueError, "payload"):
                pilot.replay_evidence(root)


if __name__ == "__main__":
    unittest.main()
