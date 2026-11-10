"""Synthetic acceptance fixtures are implementation tests, not benchmark approvals."""

import contextlib
import copy
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.helpfulness import acceptance as checks
from scripts.helpfulness import acceptance_scoring as scoring
from scripts.helpfulness import acceptance_producers as producers
from scripts.helpfulness import adapters, cli, core, native


def model_pin(model_id="reference-model"):
    return {
        "schema_version": 1, "model_id": model_id,
        "weights": {"source": "example/reference", "revision": "a" * 40},
        "tokenizer": {"source": "example/tokenizer", "revision": "b" * 40},
        "chat_template_sha256": "c" * 64,
        "inference": {
            "backend": "test-fixture", "revision": "d" * 40, "dtype": "float16",
            "settings": {"context_length": 8192, "tensor_parallel_size": 1},
        },
    }


def write_toml(path, value):
    lines = []

    def section(data, prefix=()):
        if prefix:
            lines.append("[" + ".".join(prefix) + "]")
        for key, item in data.items():
            if not isinstance(item, dict):
                lines.append(f"{key} = {json.dumps(item)}")
        for key, item in data.items():
            if isinstance(item, dict):
                lines.append("")
                section(item, (*prefix, key))

    section(value)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class Fixture:
    def __init__(self, root, kind="scorer_replay", outcomes=("CORRECT", "INCORRECT")):
        self.root, self.kind = root, kind
        self.rows = [{
            "id": str(index), "cluster_id": str(index), "stratum": "factual",
            "doc": {"question": f"Question {index}", "gold": "Gold"},
            "requests": [{"messages": [{"role": "user", "content": f"Question {index}"}]}],
            "answers": [f"Frozen answer {index}"], "judgments": [json.dumps({"outcome": outcome})],
        } for index, outcome in enumerate(outcomes)]
        self.outcomes = outcomes
        self.inputs = root / "inputs.jsonl"
        self.expected_path = root / "oracle.json"
        self.config_path = root / "config.toml"
        self.plan_path = root / "plan.json"
        self.deployment = model_pin("judge-model" if kind == "judge_calibration" else "reference-model")
        core.write_json(root / "deployment.json", self.deployment)
        self.config = {
            "schema_version": 1, "kind": kind, "benchmark": "simpleqa",
            "protocol": adapters.PROTOCOLS["simpleqa"], "scope": "synthetic_test_fixture",
            "data_license": "synthetic test data",
            "inputs": {"path": "inputs.jsonl"}, "expected": {"path": "oracle.json"},
            "policy": {"min_cases": 2, "float_tolerance": 0},
        }
        if kind != "scorer_replay":
            self.config["deployment"] = {
                "path": "deployment.json", "sha256": checks.file_hash(root / "deployment.json"),
            }
            self.config["decoding"] = {"temperature": 0, "max_tokens": 2048}
            self.config["policy"] = {
                "min_cases": 2, "min_clusters": 2, "confidence": 0.95,
                "bootstrap_iterations": 200, "seed": 41, "strata": [],
                "metrics": {
                    key: {"equivalence_margin": 0} if kind == "reference_regeneration" else {
                        "max_abs_bias": 0, "max_mae": 0, "min_cluster_agreement": 0.1,
                        "agreement_tolerance": 0,
                    } for key in ("correct", "incorrect", "not_attempted")
                },
            }
            if kind == "reference_regeneration":
                self.config["policy"]["reference_replay_tolerance"] = 0
                judge = model_pin("judge-model")
                self.config["judge"] = {
                    "deployment": judge, "decoding": {"temperature": 0, "max_tokens": 2048},
                    "rubric_sha256": core.digest(scoring.rubric("simpleqa", "judge-model")),
                }
        self.refresh()

    def refresh(self):
        self.inputs.write_text("".join(json.dumps(row) + "\n" for row in self.rows), encoding="utf-8")
        self.expected = {
            "schema_version": 1, "inputs_sha256": checks.file_hash(self.inputs),
            "source": "https://example.test/independent-oracle", "revision": "f" * 40,
            "protocol": "independent-test-reference", "label_origin": "human_annotations",
            "cases": [{
                "id": row["id"], "input_sha256": core.digest(row),
                "scores": {"outcome": outcome},
            } for row, outcome in zip(self.rows, self.outcomes)],
            "aggregate": {
                "metrics": {
                    key.lower(): self.outcomes.count(key) / len(self.outcomes)
                    for key in ("CORRECT", "INCORRECT", "NOT_ATTEMPTED")
                },
                "denominators": dict.fromkeys(("correct", "incorrect", "not_attempted"), len(self.rows)),
            },
        }
        if self.kind == "reference_regeneration":
            self.expected["deployment"] = self.deployment
            self.expected["decoding"] = self.config["decoding"]
        self.save_expected()

    def save_expected(self):
        core.write_json(self.expected_path, self.expected)
        self.config["inputs"]["sha256"] = checks.file_hash(self.inputs)
        self.config["expected"]["sha256"] = checks.file_hash(self.expected_path)
        write_toml(self.config_path, self.config)

    def seal(self):
        write_toml(self.config_path, self.config)
        self.plan = checks.create_plan(self.config_path, self.plan_path)
        return self.plan

    def candidate(self):
        result = {
            "schema_version": 1, "acceptance_plan_sha256": self.plan["sha256"],
            "inputs_sha256": self.plan["inputs"]["sha256"],
            "deployment": self.deployment, "decoding": self.config["decoding"],
            "cases": copy.deepcopy(self.rows),
        }
        if self.kind == "judge_calibration":
            result["rubric_sha256"] = self.plan["rubric_sha256"]
        else:
            result["judge"] = self.plan["judge"]
        return result

    def run(self, candidate=None, name="result"):
        path = None
        if candidate is not None:
            path = self.root / (name + "-candidate.json")
            core.write_json(path, candidate)
        return checks.run_plan(self.plan_path, path, self.root / name)


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = Fixture(Path(self.temporary.name))

    def test_exact_decisions_and_aggregation_pass(self):
        self.fixture.seal()
        result = self.fixture.run()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["scored_cases"], 2)
        self.assertFalse(result["published_protocol_equivalence"])
        self.assertTrue(all(not row["differences"] for row in result["cases"]))

    def test_same_aggregate_but_swapped_decisions_fails(self):
        fixture = self.fixture
        fixture.expected["cases"][0]["scores"]["outcome"] = "INCORRECT"
        fixture.expected["cases"][1]["scores"]["outcome"] = "CORRECT"
        fixture.save_expected()
        fixture.seal()
        result = fixture.run()
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["aggregate_differences"])
        self.assertTrue(all(row["differences"] for row in result["cases"]))

    def test_correct_decisions_wrong_denominator_fails(self):
        fixture = self.fixture
        fixture.expected["aggregate"]["denominators"]["correct"] = 3
        fixture.save_expected()
        fixture.seal()
        result = fixture.run()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["aggregate_differences"][0]["path"], "/denominators/correct")

    def test_invalid_judge_preserves_partial_coverage(self):
        fixture = self.fixture
        fixture.rows[1]["judgments"] = ["invalid JSON"]
        fixture.refresh()
        fixture.seal()
        result = fixture.run()
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["scored_cases"], 1)
        self.assertEqual(result["failed_cases"], 1)
        self.assertEqual(result["unprocessed_cases"], 0)
        self.assertEqual(result["failed_case_id"], "1")

    def test_changed_input_checksum_is_rejected(self):
        self.fixture.seal()
        self.fixture.inputs.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            self.fixture.run()

    def test_oracle_input_alignment_is_not_optional(self):
        self.fixture.expected["cases"][0]["input_sha256"] = "0" * 64
        self.fixture.save_expected()
        with self.assertRaisesRegex(ValueError, "alignment"):
            self.fixture.seal()

    def test_missing_and_duplicate_oracle_ids_are_rejected(self):
        self.fixture.expected["cases"][1]["id"] = "0"
        self.fixture.save_expected()
        with self.assertRaisesRegex(ValueError, "Oracle IDs"):
            self.fixture.seal()

    def test_plan_and_output_cannot_be_overwritten(self):
        self.fixture.seal()
        with self.assertRaisesRegex(ValueError, "overwritten"):
            self.fixture.seal()
        self.fixture.run()
        with self.assertRaisesRegex(ValueError, "overwrite"):
            self.fixture.run()

    def test_boolean_and_denominator_types_are_exact(self):
        self.assertTrue(checks.differences(True, 1, 1e-6))
        self.assertTrue(checks.differences({"denominators": {"x": 1}}, {"denominators": {"x": 1.0}}, 1e-6))
        self.assertFalse(checks.differences(0.5, 0.50000000001, 1e-6))
        self.assertTrue(checks.differences(1, 2, 1))

    def test_plan_source_runtime_and_policy_tampering_are_rejected(self):
        fixture = self.fixture
        fixture.seal()
        with patch.object(checks, "fingerprint", return_value={}):
            with self.assertRaisesRegex(ValueError, "source/runtime"):
                checks.load_plan(fixture.plan_path)
        changed = core.read_json(fixture.plan_path)
        changed["policy"]["float_tolerance"] = 1e-6
        core.write_json(fixture.plan_path, changed)
        with self.assertRaisesRegex(ValueError, "integrity"):
            checks.load_plan(fixture.plan_path)


class RegenerationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = Fixture(Path(self.temporary.name), "reference_regeneration")

    def test_matching_pinned_artifacts_pass_with_explicit_assertion_label(self):
        fixture = self.fixture
        fixture.seal()
        result = fixture.run(fixture.candidate())
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["deployment_assurance"], "operator_assertion_not_verified_server_state")
        self.assertEqual(result["uncertainty"]["groups"]["overall"]["metrics"]["correct"]["bias_interval"], [0, 0])
        self.assertFalse(result["published_protocol_equivalence"])

    def test_every_generation_pin_is_checked(self):
        fixture = self.fixture
        fixture.seal()
        mutations = (
            ("weights", lambda value: value["deployment"]["weights"].update(revision="0" * 40)),
            ("tokenizer", lambda value: value["deployment"]["tokenizer"].update(revision="0" * 40)),
            ("template", lambda value: value["deployment"].update(chat_template_sha256="0" * 64)),
            ("inference", lambda value: value["deployment"]["inference"].update(dtype="bfloat16")),
            ("decoding", lambda value: value["decoding"].update(temperature=0.7)),
            ("judge", lambda value: value["judge"].update(rubric_sha256="0" * 64)),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                candidate = copy.deepcopy(fixture.candidate())
                mutate(candidate)
                self.assertEqual(fixture.run(candidate, label)["status"], "incomplete")

    def test_same_model_name_is_insufficient_without_immutable_pins(self):
        invalid = model_pin()
        invalid["weights"]["revision"] = "main"
        with self.assertRaisesRegex(ValueError, "commit/content hash"):
            checks.deployment(invalid)

    def test_reference_pins_must_be_independently_available(self):
        fixture = self.fixture
        fixture.expected.pop("deployment")
        fixture.save_expected()
        with self.assertRaisesRegex(ValueError, "pins are missing"):
            fixture.seal()

    def test_point_difference_within_margin_but_uncertainty_outside_fails(self):
        fixture = self.fixture
        for criterion in fixture.config["policy"]["metrics"].values():
            criterion["equivalence_margin"] = 0.6
        fixture.seal()
        candidate = fixture.candidate()
        candidate["cases"][0]["answers"] = ["Different generation"]
        candidate["cases"][0]["judgments"] = ['{"outcome":"INCORRECT"}']
        result = fixture.run(candidate)
        metric = result["uncertainty"]["groups"]["overall"]["metrics"]["correct"]
        self.assertLess(abs(metric["bias"]), 0.6)
        self.assertLess(metric["bias_interval"][0], -0.6)
        self.assertFalse(metric["passed"])
        self.assertEqual(result["status"], "failed")

    def test_missing_extra_and_blank_answers_never_become_a_lower_score(self):
        fixture = self.fixture
        fixture.seal()
        for label in ("missing", "extra", "blank", "truncated"):
            candidate = fixture.candidate()
            if label == "missing":
                candidate["cases"].pop()
            elif label == "extra":
                extra = copy.deepcopy(candidate["cases"][0])
                extra["id"] = "extra"
                candidate["cases"].append(extra)
            elif label == "truncated":
                candidate["cases"][0]["generation_truncated"] = True
            else:
                candidate["cases"][0]["answers"] = [""]
            with self.subTest(label=label):
                self.assertEqual(fixture.run(candidate, label)["status"], "incomplete")

    def test_unknown_or_favorable_subset_metrics_cannot_pass(self):
        self.fixture.config["policy"]["metrics"].pop("not_attempted")
        with self.assertRaisesRegex(ValueError, "every benchmark metric"):
            self.fixture.seal()

    def test_evidence_only_metadata_is_not_a_generation_request(self):
        self.fixture.rows[0]["requests"] = [{"evidence_only": "no original prompt"}]
        self.fixture.refresh()
        with self.assertRaisesRegex(ValueError, "chat/plain completions only"):
            self.fixture.seal()

    def test_changed_requests_or_documents_are_rejected(self):
        self.fixture.seal()
        for key in ("requests", "doc"):
            candidate = self.fixture.candidate()
            candidate["cases"][0][key] = [{"prompt": "changed"}] if key == "requests" else {"gold": "changed"}
            with self.subTest(key=key):
                self.assertEqual(self.fixture.run(candidate, key)["status"], "incomplete")

    def test_clustering_does_not_count_many_summaries_as_independent_articles(self):
        self.fixture.rows[1]["cluster_id"] = "0"
        self.fixture.refresh()
        self.fixture.seal()
        result = self.fixture.run(self.fixture.candidate())
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("independent clusters", result["error"])


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = Fixture(Path(self.temporary.name), "judge_calibration")

    def test_matching_judge_can_be_frozen_and_enforced(self):
        fixture = self.fixture
        fixture.seal()
        result = fixture.run(fixture.candidate())
        self.assertEqual(result["status"], "passed")
        lock_path = fixture.root / "judge-lock.json"
        lock = checks.freeze(fixture.root / "result" / "report.json", lock_path)
        self.assertFalse(lock["published_protocol_equivalence"])
        judge = {"model": "judge-model", "deployment_manifest": "deployment.json"}
        verified = checks.validate_judge_lock(lock_path, "simpleqa", judge, fixture.root)
        self.assertEqual(verified["sha256"], lock["sha256"])
        with self.assertRaisesRegex(ValueError, "overwritten"):
            checks.freeze(fixture.root / "result" / "report.json", lock_path)

    def test_all_correct_on_tiny_fixture_does_not_report_certainty(self):
        self.fixture.seal()
        result = self.fixture.run(self.fixture.candidate())
        metric = result["uncertainty"]["groups"]["overall"]["metrics"]["correct"]
        self.assertEqual(metric["agreement"], 1)
        self.assertLess(metric["cluster_agreement_interval"][0], 1)
        self.assertGreater(metric["cluster_agreement_interval"][0], 0)

    def test_calibration_requires_independent_labels_not_its_own_judge(self):
        self.fixture.expected["label_origin"] = "replacement_judge"
        self.fixture.save_expected()
        with self.assertRaisesRegex(ValueError, "independently sourced"):
            self.fixture.seal()

    def test_changing_frozen_answers_or_rubric_invalidates_calibration(self):
        self.fixture.seal()
        for label in ("answers", "rubric"):
            candidate = self.fixture.candidate()
            if label == "answers":
                candidate["cases"][0]["answers"] = ["A different answer"]
            else:
                candidate["rubric_sha256"] = "0" * 64
            with self.subTest(label=label):
                self.assertEqual(self.fixture.run(candidate, label)["status"], "incomplete")

    def test_invalid_judgment_is_not_not_attempted_and_cannot_be_frozen(self):
        self.fixture.seal()
        candidate = self.fixture.candidate()
        candidate["cases"][0]["judgments"] = ["not JSON"]
        result = self.fixture.run(candidate)
        self.assertEqual(result["status"], "incomplete")
        with self.assertRaisesRegex(ValueError, "complete, passing"):
            checks.freeze(self.fixture.root / "result" / "report.json", self.fixture.root / "lock.json")

    def test_confusion_reports_abstention_bias(self):
        self.fixture.seal()
        candidate = self.fixture.candidate()
        for row in candidate["cases"]:
            row["judgments"] = ['{"outcome":"NOT_ATTEMPTED"}']
        result = self.fixture.run(candidate)
        group = result["uncertainty"]["groups"]["overall"]
        self.assertEqual(group["confusion"]["reference_rows_candidate_columns"]["CORRECT"]["NOT_ATTEMPTED"], 1)
        self.assertEqual(group["metrics"]["not_attempted"]["bias"], 1)
        self.assertEqual(result["status"], "failed")

    def test_evidence_changes_and_report_tampering_prevent_freeze(self):
        self.fixture.seal()
        self.fixture.run(self.fixture.candidate())
        report_path = self.fixture.root / "result" / "report.json"
        report = core.read_json(report_path)
        report["cases"][0]["actual"]["outcome"] = "NOT_ATTEMPTED"
        report.pop("sha256")
        report["sha256"] = core.digest(report)
        core.write_json(report_path, report)
        with self.assertRaisesRegex(ValueError, "results do not reproduce"):
            checks.freeze(report_path, self.fixture.root / "lock.json")

    def test_locks_reject_changed_model_rubric_and_runtime(self):
        self.fixture.seal()
        self.fixture.run(self.fixture.candidate())
        path = self.fixture.root / "lock.json"
        checks.freeze(self.fixture.root / "result" / "report.json", path)
        with self.assertRaisesRegex(ValueError, "Served judge model"):
            checks.validate_judge_lock(path, "simpleqa", {
                "model": "other-judge", "deployment_manifest": "deployment.json",
            }, self.fixture.root)
        with patch.object(scoring, "rubric", return_value={"instruction": "Changed"}):
            with self.assertRaisesRegex(ValueError, "rubric changed"):
                checks.validate_judge_lock(path, "simpleqa", {
                    "model": "judge-model", "deployment_manifest": "deployment.json",
                }, self.fixture.root)
        with patch.object(checks, "fingerprint", return_value={}):
            with self.assertRaisesRegex(ValueError, "source/runtime changed"):
                checks.validate_judge_lock(path, "simpleqa", {
                    "model": "judge-model", "deployment_manifest": "deployment.json",
                }, self.fixture.root)

    def test_default_judge_decoding_is_part_of_freeze(self):
        self.fixture.config["decoding"]["max_tokens"] = 32
        with self.assertRaisesRegex(ValueError, "decoding does not match"):
            self.fixture.seal()

    def test_human_labels_are_not_allowed_in_model_evidence(self):
        self.fixture.rows[0]["doc"]["human_scores"] = {"consistency": 5}
        self.fixture.refresh()
        with self.assertRaisesRegex(ValueError, "outside model/judge evidence"):
            self.fixture.seal()


class StatisticAndScorerTests(unittest.TestCase):
    def test_instruction_aggregation_flattens_instruction_denominators(self):
        scores = [
            dict(zip(adapters.INSTRUCTION_METRICS, (True, [True], True, [True]))),
            dict(zip(adapters.INSTRUCTION_METRICS, (False, [False, False, True], False, [False, False, True]))),
        ]
        result = scoring.aggregate("ifeval", [{}, {}], scores)
        self.assertEqual(result["metrics"]["inst_level_strict_acc"], 0.5)
        self.assertEqual(result["denominators"]["inst_level_strict_acc"], 4)
        self.assertEqual(result["denominators"]["prompt_level_strict_acc"], 2)

    def test_truthfulness_and_information_are_separately_validated(self):
        with self.assertRaisesRegex(ValueError, "conjunction"):
            scoring.validate_scores("truthfulqa_generation", {
                "truthful": True, "informative": False, "truthful_and_informative": True,
            }, {})

    def test_sycophancy_uses_aligned_total_likelihoods_with_stable_ties(self):
        row = {"doc": {"choices": ["A", "B"], "target": 0}, "answers": [], "likelihoods": [-1, -1]}
        self.assertTrue(scoring.replay_case("sycophancy", row)["biography_alignment"])
        row["likelihoods"] = [-2, -1]
        self.assertFalse(scoring.replay_case("sycophancy", row)["biography_alignment"])
        row["likelihoods"] = [float("nan"), -1]
        with self.assertRaises(ValueError):
            scoring.replay_case("sycophancy", row)

    def test_mt_bench_aggregation_preserves_turn_two(self):
        result = scoring.aggregate("mt_bench", [{}, {}], [
            {"turn_1": 8, "turn_2": 2}, {"turn_1": 6, "turn_2": 4},
        ])
        self.assertEqual(result["metrics"], {"turn_1": 7, "turn_2": 3, "score": 5})
        self.assertEqual(result["denominators"]["score"], 4)

    def test_stratum_bias_cannot_cancel_in_overall_average(self):
        rows, left, right = [], [], []
        for index in range(50):
            rows.append({"cluster_id": str(index), "stratum": "short" if index < 25 else "long"})
            left.append(dict.fromkeys(adapters.DIMENSIONS, 3.0))
            right.append(dict.fromkeys(adapters.DIMENSIONS, 2.0 if index < 25 else 4.0))
        plan = {
            "kind": "judge_calibration", "benchmark": "summeval",
            "policy": {
                "min_cases": 2, "min_clusters": 2, "confidence": 0.95,
                "bootstrap_iterations": 200, "seed": 5, "strata": ["short", "long"],
                "metrics": {
                    key: {"max_abs_bias": 0.8, "max_mae": 1.0, "min_cluster_agreement": 0.1, "agreement_tolerance": 1}
                    for key in adapters.DIMENSIONS
                },
            },
        }
        result = checks.statistics(plan, rows, left, right)
        self.assertTrue(result["groups"]["overall"]["metrics"]["consistency"]["passed"])
        self.assertFalse(result["groups"]["short"]["metrics"]["consistency"]["passed"])
        self.assertFalse(result["passed"])
        self.assertEqual(result, checks.statistics(plan, rows, left, right))

    def test_regeneration_bootstrap_uses_new_outputs_and_refits_length_control(self):
        rows = [{"cluster_id": str(index), "stratum": "all", "answers": ["old"]} for index in range(2)]
        candidate_rows = [{**row, "answers": ["new"]} for row in rows]
        scores = [{"preference": 1.5}] * 2
        plan = {
            "kind": "reference_regeneration", "benchmark": "alpaca_eval_2_lc",
            "policy": {
                "min_cases": 2, "min_clusters": 2, "confidence": 0.95,
                "bootstrap_iterations": 200, "seed": 5, "strata": [],
                "metrics": {"raw_win_rate": {"equivalence_margin": 1}, "length_controlled_winrate": {"equivalence_margin": 1}},
            },
        }
        seen = []

        def aggregate(name, examples, values, **kwargs):
            answer = examples[0]["answers"][0]
            seen.append(answer)
            return {"metrics": {"raw_win_rate": 0.5, "length_controlled_winrate": 0.25 if answer == "old" else 0.75}}

        with patch.object(scoring, "aggregate", side_effect=aggregate):
            result = checks.statistics(plan, rows, scores, scores, candidate_rows)
        self.assertEqual(seen.count("old"), 201)
        self.assertEqual(seen.count("new"), 201)
        self.assertEqual(result["groups"]["overall"]["metrics"]["length_controlled_winrate"]["bias_interval"], [0.5, 0.5])

    def test_unknown_criteria_and_secret_material_are_rejected(self):
        with self.assertRaises(ValueError):
            checks.validate_policy({"min_cases": 1, "float_tolerance": 0, "ignored_threshold": 1}, "scorer_replay")
        value = model_pin()
        value["inference"]["settings"]["api_key"] = "synthetic-secret"
        with self.assertRaisesRegex(ValueError, "credentials"):
            checks.deployment(value)

    def test_source_articles_are_required_and_clustered_by_article_not_summary(self):
        rows = [{
            "id": "summary-1", "cluster_id": "article-1", "stratum": "all",
            "doc": {"source": "source article", "source_id": "article-1"},
            "requests": [{"evidence_only": True}], "answers": ["summary"],
        }]
        checks.validate_rows(rows, "summeval")
        rows[0]["cluster_id"] = "summary-1"
        with self.assertRaisesRegex(ValueError, "source article IDs"):
            checks.validate_rows(rows, "summeval")
        with self.assertRaisesRegex(ValueError, "source"):
            scoring.validate_document("summeval", {"source_id": "article-1"})

    @unittest.skipUnless(importlib.util.find_spec("lm_eval"), "Pinned native runtime required")
    def test_real_mt_bench_parser_replays_both_turns_and_rejects_invalid_grade(self):
        native.require_harness()
        row = {
            "doc": {"turns": ["first", "second"], "category": "writing"},
            "answers": ["first answer", "second answer"], "judgments": ["Good. [[8]]", "[[3]]"],
        }
        self.assertEqual(scoring.replay_case("mt_bench", row), {"turn_1": 8, "turn_2": 3})
        row["judgments"][1] = "No rating"
        with self.assertRaises(ValueError):
            scoring.replay_case("mt_bench", row)

    @unittest.skipUnless(importlib.util.find_spec("alpaca_eval"), "Pinned AlpacaEval runtime required")
    def test_real_weighted_logprob_parser_and_orientation(self):
        import math

        row = {
            "doc": {}, "answers": ["candidate"], "candidate_position": 2,
            "judgments": [{"logprobs": {"content": [{"top_logprobs": [
                {"token": "m", "logprob": math.log(0.8)}, {"token": "M", "logprob": math.log(0.2)},
            ]}]}}],
        }
        self.assertAlmostEqual(scoring.replay_case("alpaca_eval_2_lc", row)["preference"], 1.2)
        row["candidate_position"] = 1
        self.assertAlmostEqual(scoring.replay_case("alpaca_eval_2_lc", row)["preference"], 1.8)

    @unittest.skipUnless(
        importlib.util.find_spec("lm_eval") and os.environ.get("HELPFULNESS_TEST_NLTK_DATA"),
        "Pinned native runtime and private NLTK assets required",
    )
    def test_real_ifeval_and_ifbench_replay_frozen_verdicts(self):
        with native.offline_scorers(Path(os.environ["HELPFULNESS_TEST_NLTK_DATA"])):
            for name, identifier, kwargs, answer in (
                ("ifeval", "keywords:existence", {"keywords": ["HELLO"]}, "HELLO"),
                ("ifbench", "format:no_whitespace", {}, "helloworld"),
            ):
                row = {
                    "doc": {"key": 1, "prompt": "prompt", "instruction_id_list": [identifier], "kwargs": [kwargs]},
                    "answers": [answer],
                }
                with self.subTest(name=name):
                    result = scoring.replay_case(name, row)
                    self.assertEqual(result, dict(zip(adapters.INSTRUCTION_METRICS, (True, [True], True, [True]))))

    @unittest.skipUnless(
        importlib.util.find_spec("alpaca_eval") and os.environ.get("HELPFULNESS_TEST_ALPACA_DATA"),
        "Pinned AlpacaEval runtime and staged GLM asset required",
    )
    def test_real_alpaca_annotation_aggregation_matches_upstream_glm(self):
        from alpaca_eval.metrics import get_length_controlled_winrate

        annotations, rows, scores = [], [], []
        for index in range(8):
            annotation = {
                "instruction": f"question-{index}", "index": index,
                "output_1": "reference " * (index + 2),
                "output_2": "candidate " * (index * index + 1),
                "generator_1": "gpt4", "generator_2": "candidate",
                "preference": 1.2 + index / 20,
            }
            annotations.append(annotation)
            row = {
                "doc": {"instruction": annotation["instruction"], "reference_output": annotation["output_1"]},
                "annotation": annotation, "replay_stage": "annotation", "candidate_position": 2,
                "answers": [annotation["output_2"]],
            }
            rows.append(row)
            scores.append(scoring.replay_case("alpaca_eval_2_lc", row))
        with native.offline_alpaca_assets(Path(os.environ["HELPFULNESS_TEST_ALPACA_DATA"])):
            upstream = get_length_controlled_winrate(annotations, save_weights_dir=None)
            result = scoring.aggregate("alpaca_eval_2_lc", rows, scores)
        self.assertAlmostEqual(result["metrics"]["raw_win_rate"], upstream["win_rate"] / 100)
        self.assertAlmostEqual(result["metrics"]["length_controlled_winrate"], upstream["length_controlled_winrate"] / 100)
        self.assertEqual(result["denominators"]["length_controlled_winrate"], 8)


class ProducerAndEnforcementTests(unittest.TestCase):
    def test_text_completion_forwards_pinned_decoding_and_rejects_unsupported_options(self):
        from test_helpfulness import LocalAPI

        response = {"choices": [{"index": 0, "text": "plain answer", "finish_reason": "stop"}]}
        with LocalAPI([response], []) as api:
            client = core.ChatClient({"base_url": api.url, "model": "target", "requests_per_minute": 0})
            decoding = {"temperature": 0.7, "max_tokens": 50, "top_p": 0.95, "seed": 12, "stop": ["Q:"]}
            self.assertEqual(client.generate({"prompt": "Q: question\nA:"}, decoding), "plain answer")
            self.assertEqual(api.requests[0], {"model": "target", "prompt": "Q: question\nA:", **decoding})
            with self.assertRaisesRegex(ValueError, "Unsupported decoding"):
                client.generate({"prompt": "q"}, {**decoding, "do_sample": True})
            self.assertEqual(len(api.requests), 1)

    @unittest.skipUnless(importlib.util.find_spec("lm_eval"), "Pinned native runtime required")
    def test_actual_alpaca_judging_preserves_global_indices_without_glm_download(self):
        import math
        from test_helpfulness import LocalAPI

        choice = {
            "index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "m"},
            "logprobs": {"content": [{"token": "m", "logprob": math.log(0.8), "top_logprobs": [
                {"token": "m", "logprob": math.log(0.8)}, {"token": "M", "logprob": math.log(0.2)},
            ]}]},
        }
        response = {"choices": [choice], "usage": {"total_tokens": 1, "prompt_tokens": 0, "completion_tokens": 1}}
        rows = [{
            "doc": {"instruction": f"question-{index}", "reference_output": "reference", "dataset": "test"},
            "annotation": {"index": index + 100}, "answers": [f"candidate-{index}"],
        } for index in range(2)]
        with tempfile.TemporaryDirectory() as directory, LocalAPI([], [response] * 2) as api, patch.dict(
            os.environ, {"ACCEPTANCE_TEST_KEY": "synthetic-key"}
        ):
            config = {"model": "local/judge", "base_url": api.url, "api_key_env": "ACCEPTANCE_TEST_KEY", "requests_per_minute": 0}
            with patch.object(native, "prepare_alpaca_assets", side_effect=AssertionError("must not acquire data")):
                producers.judge_alpaca(rows, config, Path(directory) / "annotator")
            self.assertEqual([row["annotation"]["index"] for row in rows], [100, 101])
            self.assertTrue(all(row["status"] == "succeeded" for row in rows))
            self.assertEqual(len(api.requests), 2)

    def test_frozen_multiturn_requests_use_new_assistant_answers(self):
        request = {"messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "answer_from_turn": 0},
            {"role": "user", "content": "second"},
        ]}
        self.assertEqual(producers.render(request, ["new answer"])["messages"][1]["content"], "new answer")
        self.assertEqual(producers.render({"prompt": ["prefix", {"answer_from_turn": 0}, "suffix"]}, ["new"])["prompt"], "prefixnewsuffix")
        with self.assertRaises(ValueError):
            producers.render(request, [])

    def test_hosted_judging_produces_actual_raw_artifacts_and_freezes(self):
        from test_helpfulness import LocalAPI

        with tempfile.TemporaryDirectory() as directory, LocalAPI([], [
            '{"outcome":"CORRECT"}', '{"outcome":"INCORRECT"}',
        ]) as api:
            root = Path(directory)
            fixture = Fixture(root, "judge_calibration")
            fixture.deployment["model_id"] = "judge"
            core.write_json(root / "deployment.json", fixture.deployment)
            fixture.config["deployment"]["sha256"] = checks.file_hash(root / "deployment.json")
            fixture.seal()
            endpoint_path = root / "endpoint.toml"
            write_toml(endpoint_path, {"endpoint": {
                "base_url": api.url, "model": "judge", "requests_per_minute": 0,
                "deployment_manifest": "deployment.json",
            }})
            candidate_path = root / "produced.json"
            artifact = producers.produce(fixture.plan_path, endpoint_path, candidate_path, "judge")
            self.assertEqual(artifact["status"], "succeeded")
            self.assertEqual(len(api.requests), 2)
            self.assertTrue(all("expected_scores" not in json.dumps(request) for request in api.requests))
            self.assertEqual(artifact["cases"][0]["judge_requests"][0], api.requests[0])
            result = checks.run_plan(fixture.plan_path, candidate_path, root / "result")
            self.assertEqual(result["status"], "passed")
            checks.freeze(root / "result" / "report.json", root / "lock.json")

    def test_hosted_regeneration_calls_target_and_independent_pinned_judge(self):
        from test_helpfulness import LocalAPI

        with tempfile.TemporaryDirectory() as directory, LocalAPI(["new answer"] * 2, [
            '{"outcome":"CORRECT"}', '{"outcome":"INCORRECT"}',
        ]) as api:
            root = Path(directory)
            fixture = Fixture(root, "reference_regeneration")
            fixture.deployment["model_id"] = "target"
            fixture.config["judge"]["deployment"]["model_id"] = "judge"
            fixture.config["judge"]["rubric_sha256"] = core.digest(scoring.rubric("simpleqa", "judge"))
            core.write_json(root / "deployment.json", fixture.deployment)
            core.write_json(root / "judge-deployment.json", fixture.config["judge"]["deployment"])
            fixture.config["deployment"]["sha256"] = checks.file_hash(root / "deployment.json")
            fixture.refresh()
            fixture.seal()
            endpoint_path = root / "endpoint.toml"
            write_toml(endpoint_path, {
                "endpoint": {"base_url": api.url, "model": "target", "requests_per_minute": 0, "deployment_manifest": "deployment.json"},
                "judge": {"base_url": api.url, "model": "judge", "requests_per_minute": 0, "deployment_manifest": "judge-deployment.json"},
            })
            path = root / "produced.json"
            artifact = producers.produce(fixture.plan_path, endpoint_path, path, "regenerate")
            self.assertEqual(artifact["status"], "succeeded")
            self.assertEqual([request["model"] for request in api.requests], ["target", "judge", "target", "judge"])
            self.assertEqual(artifact["cases"][0]["answers"], ["new answer"])
            self.assertEqual(checks.run_plan(fixture.plan_path, path, root / "result")["status"], "passed")

    def test_producer_preserves_failed_raw_judging_without_reusing_previous_scores(self):
        from test_helpfulness import LocalAPI

        with tempfile.TemporaryDirectory() as directory, LocalAPI([], ["invalid", '{"outcome":"INCORRECT"}']) as api:
            root = Path(directory)
            fixture = Fixture(root, "judge_calibration")
            fixture.deployment["model_id"] = "judge"
            core.write_json(root / "deployment.json", fixture.deployment)
            fixture.config["deployment"]["sha256"] = checks.file_hash(root / "deployment.json")
            fixture.seal()
            endpoint_path = root / "endpoint.toml"
            write_toml(endpoint_path, {"endpoint": {
                "base_url": api.url, "model": "judge", "requests_per_minute": 0,
                "deployment_manifest": "deployment.json",
            }})
            candidate = root / "candidate.json"
            result = producers.produce(fixture.plan_path, endpoint_path, candidate, "judge")
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(result["cases"][0]["judgments"], ["invalid"])
            self.assertEqual(checks.run_plan(fixture.plan_path, candidate, root / "result")["status"], "incomplete")

    def test_teacher_run_rejects_required_missing_profile_before_network_calls(self):
        suite = {
            "judge": {"model": "judge", "base_url": "http://127.0.0.1:1/v1", "require_calibration": True},
        }
        checkpoint = {
            "name": "test", "revision": "revision", "tokenizer_revision": "tokenizer",
            "chat_template_revision": "template", "endpoint": {"model": "target", "base_url": "http://127.0.0.1:1/v1"},
        }
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "frozen judge calibration is required"):
            cli.run(suite, {}, Path(directory) / "run", [checkpoint], [{"name": "simpleqa"}])

    def test_teacher_run_uses_valid_frozen_profile_and_records_its_identity(self):
        from test_helpfulness import LocalAPI

        with tempfile.TemporaryDirectory() as directory, LocalAPI(["reply"], ['{"outcome":"CORRECT"}']) as api:
            root = Path(directory)
            fixture = Fixture(root, "judge_calibration")
            fixture.deployment["model_id"] = "judge"
            core.write_json(root / "deployment.json", fixture.deployment)
            fixture.config["deployment"]["sha256"] = checks.file_hash(root / "deployment.json")
            fixture.seal()
            fixture.run(fixture.candidate())
            lock_path = root / "lock.json"
            lock = checks.freeze(root / "result" / "report.json", lock_path)
            suite = {
                "judge": {
                    "model": "judge", "base_url": api.url, "requests_per_minute": 0,
                    "require_calibration": True, "deployment_manifest": str(root / "deployment.json"),
                    "calibration_profiles": {"simpleqa": str(lock_path)},
                },
            }
            checkpoint = {
                "name": "test", "revision": "revision", "tokenizer_revision": "tokenizer",
                "chat_template_revision": "template", "endpoint": {
                    "model": "target", "base_url": api.url, "requests_per_minute": 0,
                },
            }
            data = {"simpleqa": {"cases": [{"id": "q", "question": "question", "gold": "gold"}]}}
            result = cli.run(suite, data, root / "teacher", [checkpoint], [{"name": "simpleqa"}])
            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(result["results"]["test"]["simpleqa"]["judge_calibration"]["sha256"], lock["sha256"])
            metadata = core.read_json(root / "teacher" / "run.json")
            self.assertEqual(metadata["identity"]["judge_calibrations"]["simpleqa"]["sha256"], lock["sha256"])

    def test_acceptance_cli_returns_failure_not_success_for_incomplete_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory), "judge_calibration")
            fixture.seal()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(checks.main([
                    "run", "--plan", str(fixture.plan_path), "--output", str(fixture.root / "result"),
                ]), 1)


if __name__ == "__main__":
    unittest.main()
