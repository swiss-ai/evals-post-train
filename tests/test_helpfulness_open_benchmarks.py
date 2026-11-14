"""Offline contract tests, not benchmark approvals or judge calibration."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import redirect_stderr
import io

from scripts.helpfulness import acceptance_scoring, eqbench, litbench, open_benchmarks as ob, open_models
from scripts.helpfulness.cli import custom_run
from scripts.helpfulness.core import read_jsonl, write_json
from scripts.helpfulness.eqbench_worker import bridge, valid_native_cases


class Client:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return next(self.responses)


class OpenBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def profile(self, name="gemma", model_id="google/gemma-3-27b-it"):
        card = self.root / (name + ".md")
        card.write_text("Synthetic unit-test model card; not an approved deployment.")
        config = {"base_url": "http://localhost:9000/v1", "model": name,
                  "provenance_file": str(self.root / (name + ".json"))}
        record = {
            "model_id": model_id, "revision": "a" * 40, "license": "unit-test",
            "license_url": "https://example.invalid/license", "weights_url": "https://example.invalid/weights",
            "company": "Test", "display_name": name, "reviewed_by": "test fixture",
            "open_weights": True, "license_reviewed": True, "served_model": name,
            "base_url": config["base_url"], "model_card_file": card.name,
            "model_card_sha256": hashlib.sha256(card.read_bytes()).hexdigest(),
            "capabilities": ["chat", "min_p", "text_completions", "token_logprobs"],
        }
        write_json(Path(config["provenance_file"]), record)
        return config, record

    def test_scoped_policy_requires_reviewed_exact_identity_and_card(self):
        config, record = self.profile()
        self.assertEqual(open_models.provenance(config, {"chat"})["model_id"], record["model_id"])
        for field, value in (("open_weights", False), ("license_reviewed", False),
                             ("served_model", "other"), ("revision", "main"),
                             ("model_card_sha256", "0" * 64), ("capabilities", []),
                             ("model_id", "openai/gpt-4o"), ("model_id", "openai/o1")):
            with self.subTest(field=field):
                write_json(Path(config["provenance_file"]), {**record, field: value})
                with self.assertRaises(ValueError):
                    open_models.provenance(config, {"chat"})

    def test_no_global_restriction_and_no_judge_alias_loophole(self):
        self.assertEqual(open_models.validate_suite({}, [], []), {})
        target, _ = self.profile("teacher")
        judge, _ = self.profile("judge")
        suite = {"purpose": "helpfulness_chitchat_teacher", "model_policy": "open_weights_only", "judge": judge}
        with self.assertRaisesRegex(ValueError, "independent"):
            open_models.validate_suite(suite, [{"name": "teacher", "endpoint": target}], [{"name": "mt_bench"}])

    def test_assets_reject_changed_bytes_before_import(self):
        with patch.dict(ob.SPECS, {"fixture": ("owner/repo", "a" * 40,
                              {"prompt.py": hashlib.sha256(b"TEXT = 'rubric'\n").hexdigest()})}):
            (self.root / "prompt.py").write_bytes(b"TEXT = 'rubric'\n")
            self.assertEqual(ob.staged_assets("fixture", self.root)["revision"], "a" * 40)
            (self.root / "prompt.py").write_bytes(b"TEXT = 'changed'\n")
            with self.assertRaisesRegex(ValueError, "differs"):
                ob.staged_assets("fixture", self.root)

    def test_conversation_preserves_context_and_rejects_missing_turns(self):
        history = [{"role": "user", "content": "remember"},
                   {"role": "assistant", "content": "fixed recorded answer"},
                   {"role": "user", "content": "last turn"}]
        case = ob.normalize("multichallenge", [{
            "QUESTION_ID": 7, "CONVERSATION": history, "AXIS": "RECOLLECTION",
            "TARGET_QUESTION": "did it forget?", "PASS_CRITERIA": "NO",
        }], {})[0]
        client = Client("completion")
        self.assertEqual(ob.generate("multichallenge", case, client, 100), ["completion"])
        self.assertEqual(client.calls[0][0], history)
        with self.assertRaises(ValueError):
            ob.messages(history[:-1])

    def test_multichallenge_preserves_inverted_criterion(self):
        assets = {"files": {"src/evaluator.py": 'JUDGE_PROMPT = "Response: {} Criterion: {}"'}}
        client = Client('{"reasoning":"criterion absent","verdict":"NO"}')
        scores, trace = ob.score("multichallenge", {"criterion": "harm?", "pass_criteria": "NO"},
                                "safe", client, assets, {}, {})
        self.assertTrue(scores["passed"])
        self.assertEqual(len(trace), 1)

    def test_prefix_profiles_are_case_sensitive_and_not_semantic_refusal(self):
        for profile, answer, expected in (("opensource", "  I cannot do that", True),
                                           ("opensource", "I'm sorry, but no", False),
                                           ("qwen", "I'm sorry, but no", True),
                                           ("opensource", "sorry", False)):
            scores, _ = ob.score("or_bench", {}, answer, None, {"files": {}},
                                 {"keyword_profile": profile}, {})
            self.assertEqual(scores["refused"], expected)

    def test_wildbench_scales_score_and_keeps_history_checklist(self):
        assets = {"files": {"evaluation/eval_template.score.v2.md":
                  "{$history}|{$user_query}|{$model_output}|{$checklist}"}}
        case = {"messages": [{"role": "user", "content": "earlier"},
                             {"role": "assistant", "content": "prior"},
                             {"role": "user", "content": "now"}], "checklist": ["check fact"]}
        client = Client('{"strengths":"good","weaknesses":"none","score":"8"}')
        scores, _ = ob.score("wildbench_v2", case, "answer", client, assets, {}, {})
        self.assertEqual(scores, {"score_1_10": 8, "wb_score": 60})
        prompt = client.calls[0][0][0]["content"]
        self.assertIn("ASSISTANT: prior", prompt)
        self.assertIn("- check fact", prompt)
        for bad in ('{"score":true}', '{"strengths":"","weaknesses":"","score":"8 or 9"}', 'not json'):
            with self.assertRaises(ValueError):
                ob.score("wildbench_v2", case, "answer", Client(bad), assets, {}, {})

    def test_arena_reverses_positions_without_changing_candidate_orientation(self):
        assets = {"files": {"utils/judge_utils.py":
                  'PROMPT = "native system"\nJUDGE_SETTINGS = {"hard_prompt": {"system_prompt": PROMPT}}'}}
        case = {"question": "q", "category": "hard_prompt", "reference_answer": "reference"}
        client = Client("reason [[B>>A]]", "reason [[A>B]]")
        scores, trace = ob.score("arena_hard_v2", case, "candidate", client, assets, {}, {})
        self.assertEqual(scores, {"win_credit": 4, "battle_weight": 4})
        self.assertIn("candidate", trace[0]["messages"][1]["content"])
        self.assertNotEqual(trace[0]["messages"], trace[1]["messages"])

    def test_arena_aggregation_weights_strong_preferences_not_prompts(self):
        records = [
            {"case": {"category": "hard_prompt"}, "status": "succeeded",
             "scores": {"win_credit": 6, "battle_weight": 6}},
            {"case": {"category": "hard_prompt"}, "status": "succeeded",
             "scores": {"win_credit": 0, "battle_weight": 2}},
        ]
        report = ob.summarize("arena_hard_v2", records, 2)
        self.assertEqual(report["metrics"]["weighted_win_rate"], {"value": 0.75, "denominator": 8})
        self.assertEqual(report["groups"]["hard_prompt"]["weighted_win_rate"]["value"], 0.75)
        self.assertFalse(report["headline_bradley_terry_score_computed"])

    def test_wildbench_macro_preserves_native_duplicate_group_memberships(self):
        cases = ob.normalize("wildbench_v2", [
            {"id": "a", "conversation_input": [{"role": "user", "content": "q"}],
             "checklist": ["c"], "primary_tag": "Creative Writing", "secondary_tags": ["Editing"]},
            {"id": "b", "conversation_input": [{"role": "user", "content": "q"}],
             "checklist": ["c"], "primary_tag": "Math", "secondary_tags": ["Creative Writing"]},
        ], {})
        records = [
            {"case": case, "status": "succeeded", "scores": {"wb_score": score}}
            for case, score in zip(cases, [100, 0])
        ]
        report = ob.summarize("wildbench_v2", records, 2)
        self.assertEqual(report["groups"]["Creative Tasks"]["wb_score"]["denominator"], 3)
        self.assertAlmostEqual(report["metrics"]["task_macro_wb_score"]["value"], 100 / 3)

    def test_abstention_scores_correctness_and_group_macro_f1(self):
        assets = {"files": {"recipe/evaluation_judge_prompts.py":
                  'LLM_JUDGE_ABSTENTION_COCONOT_STYLE_PROMPT = "{question} {model_answer} {ref_answer} {abstention_label}"\n'
                  'LLM_JUDGE_CORRECTNESS_PROMPT = "{question} {model_answer} {ref_answer}"'}}
        cases = ob.normalize("abstention_bench", [
            {"id": "1", "question": "answerable", "dataset": "one", "scenario": "uncertainty",
             "reference_answers": ["gold"], "should_abstain": False},
            {"id": "2", "question": "unknown", "dataset": "two", "scenario": "unknown",
             "reference_answers": None, "should_abstain": True},
        ], {})
        client = Client("No", "correct", "Yes.")
        records = []
        for case in cases:
            scores, _ = ob.score("abstention_bench", case, "answer", client, assets, {}, {})
            records.append({"case": case, "status": "succeeded", "scores": scores})
        report = ob.summarize("abstention_bench", records, 2)
        self.assertEqual(report["groups"]["one/uncertainty"]["answerable_accuracy"], 1)
        self.assertEqual(report["groups"]["two/unknown"]["f1"], 1)
        self.assertEqual(report["metrics"]["macro_dataset_scenario_f1"]["value"], 0.5)
        with self.assertRaises(ValueError):
            ob.binary("maybe", "yes", "no")

    def test_elephant_three_dimensions_keep_framing_orientation(self):
        assets = {"files": {"sycophancy_scorers.py":
                  'raise RuntimeError("module import forbidden")\n'
                  'def create_prompt(row, metric, prompt_col="question", response_col="response"):\n'
                  '    return metric + row[response_col]\n'}}
        scores, trace = ob.score("elephant", {"question": "q"}, "a",
                                 Client("1", "0", "1"), assets, {}, {})
        self.assertEqual(scores, {"validation": 1, "indirectness": 0, "framing": 1})
        self.assertEqual(len(trace), 3)
        with self.assertRaises(ValueError):
            ob.score("elephant", {"question": "q"}, "a", Client("reason: 1"), assets, {}, {})

    def test_elephant_moral_requires_original_pairs_and_reports_unclassified(self):
        options = {"variant": "moral"}
        cases = ob.normalize("elephant", [{"id": "pair", "side_a_prompt": "a",
                                         "side_b_prompt": "b"}], options)
        target = Client("NTA", "YTA")
        answers = ob.generate("elephant", cases[0], target, 64)
        self.assertEqual(target.calls[1][0][0]["content"], "b\nOutput only YTA or NTA.")
        scores, _ = ob.score("elephant", cases[0], answers[-1], None, {"files": {}},
                              options, {"answers": answers})
        self.assertTrue(scores["side_A_NTA_side_B_YTA"])
        self.assertFalse(scores["unclassified"])
        scores, _ = ob.score("elephant", cases[0], "no verdict", None, {"files": {}},
                              options, {"answers": ["not a verdict", "NTA"]})
        self.assertTrue(scores["unclassified"])
        with self.assertRaises(ValueError):
            ob.normalize("elephant", [{"id": "unpaired", "side_a_prompt": "a"}], options)

    def test_elephant_human_baseline_uses_same_three_native_rubrics(self):
        assets = {"files": {"sycophancy_scorers.py":
                  'def create_prompt(row, metric, prompt_col="question", response_col="response"):\n'
                  '    return metric + row[response_col]\n'}}
        client = Client("1", "0", "1", "0", "1", "1")
        scores, trace = ob.score("elephant", {"question": "q", "human_response": "human"},
                                 "teacher", client, assets, {}, {})
        self.assertEqual([scores["delta_" + key] for key in ("validation", "indirectness", "framing")],
                         [1, -1, 0])
        self.assertEqual(len(trace), 6)
        self.assertIn("human", trace[3]["messages"][1]["content"])

    def test_darkbench_binds_model_info_without_reformatting_untrusted_braces(self):
        assets = {"files": {
            "darkbench/dark_patterns.py": 'BIAS = DarkPattern(id="brand_bias", display_name="Brand", description="{company} {model}")',
            "darkbench/scorer.py": 'OVERSEER_TEMPLATE = "{issue_description} {prompt} {response} {{{{json}}}}"\n'
                                   'OVERSEER_SYSTEM_MESSAGE = "annotate"',
        }}
        case = {"question": "q {keep}", "issue": "brand_bias"}
        good = '{"brand_bias":1,"invalid":0,"egregious_example":1,"reasoning":"bias"}'
        scores, trace = ob.score("darkbench", case, '{"answer":1}', Client(good), assets,
                                 {}, {"company": "Swiss", "display_name": "Apertus"})
        self.assertEqual(scores, {"incidence": True, "egregious": True})
        self.assertIn('Swiss Apertus q {keep} {"answer":1} {json}', trace[0]["messages"][1]["content"])
        with self.assertRaisesRegex(ValueError, "invalid"):
            ob.score("darkbench", case, "a", Client(good.replace('"invalid":0', '"invalid":1')),
                     assets, {}, {"company": "Swiss", "display_name": "Apertus"})

    def test_judge_failure_is_persisted_and_resume_reuses_teacher_answer(self):
        case = {"id": "one", "criterion": "yes?", "pass_criteria": "YES",
                "messages": [{"role": "user", "content": "q"}]}
        assets = {"files": {"src/evaluator.py": 'JUDGE_PROMPT = "{} {}"'}}
        path = self.root / "samples.jsonl"
        target = Client("answer")
        report = custom_run("multichallenge", [case], "generate", target, Client("bad"),
                            path, 100, assets, {}, {})
        self.assertEqual(report["failed_cases"], 1)
        self.assertIn("bad", read_jsonl(path)[-1]["raw_judge"])
        report = custom_run("multichallenge", [case], "generate", target,
                            Client('{"reasoning":"ok","verdict":"YES"}'), path, 100, assets, {}, {})
        self.assertEqual(report["scored_cases"], 1)
        self.assertEqual(len(target.calls), 1)

    def test_extensions_do_not_silently_enter_existing_acceptance_contract(self):
        for name in ob.NAMES | set(eqbench.PINS):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "uncalibrated"):
                acceptance_scoring.validate_document(name, {})

    def test_eq_native_summary_rejects_partial_success_shaped_results(self):
        row = {"scenario_tasks": {"1": {
            "ok": {"status": "rubric_scored", "rubric_scores": {"empathy": 16}},
            "bad": {"status": "error"},
        }}, "results": {"average_rubric_score": 16}}
        report = eqbench.summarize("eqbench3", {"run": row}, 3)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["scored_cases"], 1)
        self.assertEqual(report["failed_cases"], 1)
        self.assertEqual(report["unprocessed_cases"], 1)
        self.assertFalse(report["elo_computed"])

    def test_eq_native_completion_does_not_hide_rubrics_excluded_by_native_aggregation(self):
        row = {"scenario_tasks": {"1": {
            "unknown": {"status": "rubric_scored", "rubric_scores": {"invented_metric": 16}},
        }}, "results": {"average_rubric_score": 0}}
        native = SimpleNamespace(calculate_final_rubric_score=lambda _: (None, "no allowed metrics"))
        valid = valid_native_cases("eqbench3", {"run": row}, native, {"1/unknown"})
        report = eqbench.summarize("eqbench3", {"run": row}, 1, set(valid))
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["failed_cases"], 1)
        self.assertEqual(report["metrics"]["native_rubric_score"]["scale"], [0, 20])
        with self.assertRaisesRegex(ValueError, "unplanned"):
            valid_native_cases("eqbench3", {"run": row}, native, {"1/different"})

    def test_eq_bridge_refuses_undeclared_role_and_closed_model_switch(self):
        fake = Client()
        with patch("scripts.helpfulness.eqbench_worker.ChatClient", return_value=fake):
            cls = bridge({"model": "teacher"}, {"model": "gemma"}, self.root / "trace.jsonl")
            with self.assertRaises(ValueError):
                cls()
            client = cls(model_type="judge")
            with self.assertRaisesRegex(ValueError, "undeclared"):
                client.generate("claude-anything", [{"role": "user", "content": "q"}])
        self.assertEqual(fake.calls, [])

    def test_creative_native_bridge_accepts_string_prompt_seed_and_system(self):
        class Fake:
            def __init__(self):
                self.calls = []

            def generate(self, request, decoding):
                self.calls.append((request, decoding))
                return "native rubric response"

        fake = Fake()
        with patch("scripts.helpfulness.eqbench_worker.ChatClient", return_value=fake):
            cls = bridge({"model": "teacher"}, {"model": "phi"}, self.root / "trace.jsonl",
                         "eqbench_creative_v3")
            result = cls(model_type="judge").generate(
                "phi", "native string prompt", temperature=0, max_tokens=4096,
                include_seed=True, system="native system",
            )
        self.assertEqual(result, "native rubric response")
        self.assertEqual(fake.calls, [(
            {"messages": [{"role": "system", "content": "native system"},
                          {"role": "user", "content": "native string prompt"}]},
            {"temperature": 0, "max_tokens": 4096, "min_p": 0.1},
        )])

    def test_litbench_scorer_validation_counts_ties_as_not_correct_and_resumes(self):
        pairs = [
            {"chosen_story": "best", "rejected_story": "bad", "chosen_comment_id": "a", "rejected_comment_id": "b"},
            {"chosen_story": "same", "rejected_story": "same", "chosen_comment_id": "c", "rejected_comment_id": "d"},
        ]
        scores = {"best": 3.0, "bad": -1.0, "same": 2.0}
        report = litbench.validate(pairs, scores.__getitem__, self.root / "validation", {"model": "fixture"})
        self.assertEqual(report["accuracy"], 0.5)
        self.assertEqual(report["ties"], 1)
        self.assertEqual(report["scope"], "scorer_validation_not_teacher_quality")
        def unexpected(_):
            self.fail("Completed reward pairs must not be rescored")
        litbench.validate(pairs, unexpected, self.root / "validation", {"model": "fixture"}, True)
        with self.assertRaisesRegex(ValueError, "resume"):
            litbench.validate(pairs, unexpected, self.root / "validation", {"model": "changed"}, True)
        records = read_jsonl(self.root / "validation" / "pairs.jsonl")
        records[0]["correct"] = False
        (self.root / "validation" / "pairs.jsonl").write_text(
            "\n".join(json.dumps(row) for row in records) + "\n",
        )
        with self.assertRaisesRegex(ValueError, "verdicts"):
            litbench.validate(pairs, unexpected, self.root / "validation", {"model": "fixture"}, True)

    def test_litbench_refuses_training_or_partial_ids_before_loading_any_model(self):
        pair = {"chosen_story": "yes", "rejected_story": "no",
                "chosen_comment_id": "a", "rejected_comment_id": "b"}
        pairs, ids = self.root / "pairs.jsonl", self.root / "ids.jsonl"
        pairs.write_text(json.dumps(pair) + "\n")
        ids.write_text(json.dumps({key: value for key, value in pair.items() if key.endswith("_id")}) + "\n")
        data = {"source": "SAA-Lab/LitBench-Test", "revision": litbench.DATA_REVISION,
                "terms": "synthetic fixture only",
                "rehydrated_sha256": hashlib.sha256(pairs.read_bytes()).hexdigest(),
                "ids_sha256": hashlib.sha256(ids.read_bytes()).hexdigest()}
        manifest = self.root / "data.json"
        write_json(manifest, data)
        with patch.object(litbench, "local_reward") as loader, redirect_stderr(io.StringIO()) as errors:
            result = litbench.main([
                "--pairs", str(pairs), "--heldout-ids", str(ids), "--data-manifest", str(manifest),
                "--model-directory", str(self.root / "absent-weights"),
                "--provenance", str(self.root / "absent-profile.json"),
                "--output", str(self.root / "output"),
            ])
        self.assertEqual(result, 1)
        self.assertIn("complete coverage", errors.getvalue())
        loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
