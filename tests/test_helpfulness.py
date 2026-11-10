"""Offline adapter contracts and a real local-HTTP CLI round trip."""

import contextlib
import io
import importlib.util
import json
import math
import os
import tempfile
import threading
import unittest
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.helpfulness import adapters, cli, core, native


class FixedJudge:
    def __init__(self, output):
        self.output = output
        self.evidence = None

    def judge(self, instruction, evidence):
        self.evidence = evidence
        return self.output


class RecordingTarget:
    def __init__(self):
        self.messages = []

    def complete(self, messages, **kwargs):
        self.messages.append(messages)
        return f"answer-{len(self.messages)}"


def scored(case_id, scores, case=None):
    return {"id": case_id, "status": "succeeded", "scores": scores, "case": case or {}}


class GradingTests(unittest.TestCase):
    def test_truth_and_information_are_independent(self):
        judge = FixedJudge('{"truthful":true,"informative":false}')
        result, raw = adapters.score_case("truthfulqa_generation", {"question": "q"}, ["abstain"], judge)
        self.assertTrue(result["truthful"])
        self.assertFalse(result["informative"])
        self.assertFalse(result["truthful_and_informative"])
        self.assertIn("true", raw)

    def test_truth_judge_rejects_numeric_or_missing_booleans(self):
        for output in ('{"truthful":1,"informative":1}', '{"truthful":true}', "[]", "not JSON"):
            with self.subTest(output=output), self.assertRaises(ValueError):
                adapters.score_case("truthfulqa_generation", {}, ["a"], FixedJudge(output))

    def test_simpleqa_reports_each_outcome(self):
        rows = []
        for outcome in ("CORRECT", "INCORRECT", "NOT_ATTEMPTED"):
            scores, _ = adapters.score_case("simpleqa", {}, ["a"], FixedJudge(json.dumps({"outcome": outcome})))
            rows.append(scored(outcome, scores))
        rows.append({"id": "bad", "status": "failed"})
        summary = adapters.summarize_custom("simpleqa", rows, 5, "generate")
        self.assertEqual(summary["status"], "incomplete")
        self.assertEqual(summary["failed_cases"], 1)
        self.assertEqual(summary["unprocessed_cases"], 1)
        for outcome in ("correct", "incorrect", "not_attempted"):
            self.assertEqual(summary["metrics"][outcome]["count"], 1)
            self.assertAlmostEqual(summary["metrics"][outcome]["rate"], 1 / 3)

    def test_invalid_simpleqa_judge_is_not_abstention(self):
        for output in ('{"outcome":"UNKNOWN"}', '"NOT_ATTEMPTED"', '{"outcome":"CORRECT","extra":1}', "C"):
            with self.subTest(output=output), self.assertRaises(ValueError):
                adapters.score_case("simpleqa", {}, ["a"], FixedJudge(output))

    def test_summary_scores_are_four_finite_dimensions(self):
        expected = dict.fromkeys(adapters.DIMENSIONS, 4)
        result, _ = adapters.score_case("summeval", {"source": "article"}, ["summary"], FixedJudge(json.dumps(expected)))
        self.assertEqual(result, expected)
        for value in (True, 0, 6, float("nan"), "4"):
            invalid = {**expected, "consistency": value}
            with self.subTest(value=value), self.assertRaises(ValueError):
                adapters.score_case("summeval", {}, ["a"], FixedJudge(json.dumps(invalid)))

    def test_judge_is_not_given_human_scores(self):
        judge = FixedJudge(json.dumps(dict.fromkeys(adapters.DIMENSIONS, 4)))
        adapters.score_case("summeval", {"source": "article", "human_scores": {"consistency": 1}}, ["s"], judge)
        self.assertNotIn("human_scores", judge.evidence)
        self.assertEqual(judge.evidence["source"], "article")

    def test_two_turn_generation_retains_assistant_history(self):
        target = RecordingTarget()
        answers = adapters.generate_case("ifbench_multiturn", {"turns": ["initial", "constraint"]}, target, 128)
        self.assertEqual(answers, ["answer-1", "answer-2"])
        self.assertEqual(target.messages[1], [
            {"role": "user", "content": "initial"},
            {"role": "assistant", "content": "answer-1"},
            {"role": "user", "content": "constraint"},
        ])

    def test_multiturn_scores_second_answer_with_native_verifiers(self):
        case = {"key": 1, "prompt": "combined", "instruction_id_list": ["x"], "kwargs": [{}]}
        expected = {
            "prompt_level_strict_acc": False, "inst_level_strict_acc": [False],
            "prompt_level_loose_acc": True, "inst_level_loose_acc": [True],
        }
        utils = SimpleNamespace(process_results=lambda doc, responses: expected)
        with patch.object(adapters.importlib, "import_module", return_value=utils):
            result, _ = adapters.score_case("ifbench_multiturn", case, ["first", "second"], None)
        self.assertEqual(result, expected)

    def test_instruction_denominator_is_not_prompt_average(self):
        rows = [
            scored("one", {"prompt_level_strict_acc": True, "inst_level_strict_acc": [True],
                           "prompt_level_loose_acc": True, "inst_level_loose_acc": [True]}),
            scored("two", {"prompt_level_strict_acc": False, "inst_level_strict_acc": [False, False, True],
                           "prompt_level_loose_acc": False, "inst_level_loose_acc": [False, False, True]}),
        ]
        result = adapters.summarize_custom("ifbench_multiturn", rows, 2, "generate")
        self.assertEqual(result["metrics"]["inst_level_strict_acc"], {"value": 0.5, "denominator": 4})
        self.assertEqual(result["metrics"]["prompt_level_strict_acc"]["denominator"], 2)

    def test_no_successful_judgments_has_null_metrics(self):
        result = adapters.summarize_custom("simpleqa", [{"status": "failed"}], 1, "generate")
        self.assertIsNone(result["metrics"]["not_attempted"]["rate"])

    def test_correlations_and_tied_ranks(self):
        self.assertAlmostEqual(adapters.correlation([1, 2, 3], [3, 2, 1]), -1)
        self.assertEqual(adapters.ranks([1, 1, 4]), [1.5, 1.5, 3])
        self.assertIsNone(adapters.correlation([1, 1], [2, 3]))
        self.assertIsNone(adapters.correlation([1], [2]))

    def test_human_validation_reports_correlations(self):
        rows = [
            scored(str(value), dict.fromkeys(adapters.DIMENSIONS, value),
                   {"human_scores": dict.fromkeys(adapters.DIMENSIONS, 6 - value)})
            for value in (1, 3, 5)
        ]
        result = adapters.summarize_custom("summeval", rows, 3, "validate")
        for dimension in adapters.DIMENSIONS:
            self.assertAlmostEqual(result["metrics"][dimension]["spearman"], -1)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)

    def source(self, rows):
        path = self.root / "source.jsonl"
        for row in rows:
            core.append_jsonl(path, row)
        return path

    def test_truthfulqa_csv_schema(self):
        path = self.root / "truth.csv"
        path.write_text("Question,Correct Answers,Incorrect Answers\nq,true,false\n", encoding="utf-8")
        cases = adapters.load_custom("truthfulqa_generation", path)
        self.assertEqual(cases[0]["true_answers"], "true")

    def test_missing_summary_source_is_rejected(self):
        with self.assertRaises(ValueError):
            adapters.load_custom("summeval", self.source([{"id": "s", "source_id": "a", "summary": "s"}]))

    def test_conflicting_source_join_is_rejected(self):
        with self.assertRaises(ValueError):
            adapters.load_custom("summeval", self.source([
                {"id": "s1", "source_id": "a", "source": "first"},
                {"id": "s2", "source_id": "a", "source": "wrong"},
            ]))

    def test_summary_generation_deduplicates_articles_not_validation_summaries(self):
        rows = [
            {"id": str(index), "source_id": "a", "source": "article", "summary": "s",
             "human_scores": dict.fromkeys(adapters.DIMENSIONS, 4)}
            for index in (1, 2)
        ]
        path = self.source(rows)
        self.assertEqual(len(adapters.load_custom("summeval", path)), 1)
        self.assertEqual(len(adapters.load_custom("summeval", path, "validate")), 2)

    def test_summary_validation_requires_human_scores(self):
        with self.assertRaises(ValueError):
            adapters.load_custom("summeval", self.source([
                {"id": "s", "source_id": "a", "source": "article", "summary": "s"}
            ]), "validate")

    def test_duplicate_case_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            adapters.load_custom("simpleqa", self.source([
                {"id": "same", "problem": "q", "answer": "a"},
                {"id": "same", "problem": "q2", "answer": "a2"},
            ]))

    def test_bad_multiturn_constraints_are_rejected(self):
        with self.assertRaises(ValueError):
            adapters.load_custom("ifbench_multiturn", self.source([
                {"turns": ["first"], "prompt": "p", "instruction_id_list": ["x"], "kwargs": []}
            ]))

    def test_mt_category_filter_precedes_limit(self):
        data = {"mt_bench": [
            {"category": "coding", "id": 1}, {"category": "writing", "id": 2},
            {"category": "writing", "id": 3},
        ]}
        result = cli.select_documents("mt_bench", data, ["writing"], 1)
        self.assertEqual(result["mt_bench"][0]["id"], 2)
        with self.assertRaises(ValueError):
            cli.select_documents("mt_bench", data, ["unknown"], None)


class NativeReportTests(unittest.TestCase):
    def test_all_instruction_metrics_and_denominators(self):
        result = {"results": {"ifeval": {key + ",none": 0.5 for key in adapters.INSTRUCTION_METRICS}},
                  "samples": {"ifeval": [
                      {"doc": {"instruction_id_list": ["a", "b"]}, "filtered_resps": ["answer"]},
                      {"doc": {"instruction_id_list": ["a"]}, "filtered_resps": ["answer"]},
                  ]}}
        summary = adapters.summarize_native("ifeval", result, {"ifeval": 2})
        self.assertEqual(summary["status"], "succeeded")
        self.assertEqual(summary["denominators"]["ifeval/inst_level_strict_acc"], 3)

    def test_missing_native_metric_is_not_success(self):
        with self.assertRaises(ValueError):
            adapters.summarize_native("ifbench", {"results": {"ifbench": {}}}, {"ifbench": 1})

    def test_native_missing_samples_are_incomplete(self):
        result = {"results": {"ifeval": {key + ",none": 1 for key in adapters.INSTRUCTION_METRICS}},
                  "samples": {"ifeval": []}}
        self.assertEqual(adapters.summarize_native("ifeval", result, {"ifeval": 1})["status"], "incomplete")

    def test_mt_turn_two_and_judge_failures_are_visible(self):
        metrics = {
            "mt_bench_score,none": 7, "mt_bench_turn_1_score,none": 8,
            "mt_bench_turn_2_score,none": 6, "mt_bench_judge_success_rate,none": 0.5,
        }
        result = {"results": {"mt_bench": metrics}, "samples": {"mt_bench": [{"filtered_resps": ["a"]}]}}
        summary = adapters.summarize_native("mt_bench", result, {"mt_bench": 1})
        self.assertEqual(summary["status"], "incomplete")
        self.assertEqual(summary["metrics"]["mt_bench/mt_bench_turn_2_score"], 6)

    def test_sycophancy_alignment_is_not_named_correctness(self):
        result = {"results": {"probe": {"acc,none": 0.8}}, "samples": {"probe": [{}]}}
        summary = adapters.summarize_native("sycophancy", result, {"probe": 1})
        self.assertEqual(summary["metrics"], {"probe/biography_alignment_rate": 0.8})

    def test_native_nan_is_rejected(self):
        with self.assertRaises(ValueError):
            adapters.summarize_native("sycophancy", {"results": {"p": {"acc,none": math.nan}}}, {"p": 1})

    def test_native_out_of_range_accuracy_is_rejected(self):
        with self.assertRaises(ValueError):
            adapters.summarize_native("sycophancy", {"results": {"p": {"acc,none": 2}}}, {"p": 1})

    def test_native_nested_truncation_is_incomplete(self):
        result = {
            "results": {"ifeval": {key + ",none": 1 for key in adapters.INSTRUCTION_METRICS}},
            "samples": {"ifeval": [{
                "doc": {"instruction_id_list": ["a"]}, "filtered_resps": [["answer"]],
                "length_info": [[{"generation_truncated": True}]],
            }]},
        }
        summary = adapters.summarize_native("ifeval", result, {"ifeval": 1})
        self.assertEqual(summary["status"], "incomplete")
        self.assertEqual(summary["coverage"]["ifeval"]["truncated_samples"], 1)

    def test_mt_nested_answers_require_two_nonempty_turns(self):
        metrics = dict.fromkeys((
            "mt_bench_score,none", "mt_bench_turn_1_score,none",
            "mt_bench_turn_2_score,none", "mt_bench_judge_success_rate,none",
        ), 1)
        for answers, expected in ((["one", "two"], "succeeded"), (["one", ""], "incomplete"), (["one"], "incomplete")):
            result = {
                "results": {"mt_bench": metrics},
                "samples": {"mt_bench": [{"filtered_resps": [{"responses": answers}]}]},
            }
            with self.subTest(answers=answers):
                self.assertEqual(adapters.summarize_native("mt_bench", result, {"mt_bench": 1})["status"], expected)

    def test_asset_hashes_detect_changes(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "asset.bin"
            path.write_bytes(b"original")
            original = native.asset_hashes(Path(root))
            path.write_bytes(b"changed")
            self.assertNotEqual(original, native.asset_hashes(Path(root)))

    def test_offline_scorers_block_downloads_and_restore_state(self):
        def original_download(*args, **kwargs):
            raise AssertionError("must not download")

        module = SimpleNamespace(data=SimpleNamespace(path=["existing"]), download=original_download)
        with patch.dict("sys.modules", {"nltk": module}):
            with native.offline_scorers(Path("prepared")):
                self.assertEqual(module.data.path[0], "prepared")
                with self.assertRaisesRegex(ValueError, "prepare"):
                    module.download("missing")
            self.assertEqual(module.data.path, ["existing"])
            self.assertIs(module.download, original_download)

    def test_offline_flags_update_already_imported_hf_modules(self):
        hub = SimpleNamespace(HF_HUB_OFFLINE=False)
        datasets = SimpleNamespace(HF_DATASETS_OFFLINE=False)
        with patch.dict("sys.modules", {"huggingface_hub.constants": hub, "datasets.config": datasets}):
            with native.environment({"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}):
                self.assertTrue(hub.HF_HUB_OFFLINE)
                self.assertTrue(datasets.HF_DATASETS_OFFLINE)
            self.assertFalse(hub.HF_HUB_OFFLINE)
            self.assertFalse(datasets.HF_DATASETS_OFFLINE)

    def test_harness_pin_is_checked(self):
        distribution = SimpleNamespace(read_text=lambda name: json.dumps({"vcs_info": {"commit_id": "wrong"}}))
        with patch.dict(os.environ, {}, clear=True), patch.object(
            native.importlib.metadata, "distribution", return_value=distribution
        ), self.assertRaises(ValueError):
            native.require_harness()

    def test_harness_exact_pin_is_accepted(self):
        distribution = SimpleNamespace(read_text=lambda name: json.dumps({"vcs_info": {"commit_id": native.harness_pin()}}))
        with patch.dict(os.environ, {}, clear=True), patch.object(
            native.importlib.metadata, "distribution", return_value=distribution
        ):
            native.require_harness()


class LocalAPI:
    def __init__(self, target_outputs, judge_outputs):
        self.target_outputs = deque(target_outputs)
        self.judge_outputs = deque(judge_outputs)
        self.requests = []
        self.authorizations = []

    def __enter__(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                response = json.dumps({"data": [{"id": model} for model in (
                    "target", "judge", "local/target", "local/judge",
                )]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append(payload)
                owner.authorizations.append(self.headers.get("Authorization"))
                queue = owner.judge_outputs if payload["model"].rsplit("/", 1)[-1] == "judge" else owner.target_outputs
                item = queue.popleft()
                if isinstance(item, dict):
                    result = item
                else:
                    content, finish = item if isinstance(item, tuple) else (item, "stop")
                    result = {"choices": [{
                        "index": 0, "message": {"role": "assistant", "content": content},
                        "finish_reason": finish,
                    }], "usage": {"total_tokens": 1, "prompt_tokens": 0, "completion_tokens": 1}}
                response = json.dumps(result).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


class IntegrationTests(unittest.TestCase):
    def test_custom_client_sends_configured_bearer_credential(self):
        with LocalAPI(["answer"], []) as api, patch.dict(os.environ, {"TEST_API_KEY": "synthetic-key"}):
            client = core.ChatClient({
                "base_url": api.url, "model": "target", "requests_per_minute": 0,
                "api_key_env": "TEST_API_KEY",
            })
            client.complete([{"role": "user", "content": "question"}])
            self.assertEqual(api.authorizations, ["Bearer synthetic-key"])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)

    def manifest(self, url):
        path = self.root / "suite.toml"
        path.write_text(f"""
schema_version = 1
max_tokens = 128
[judge]
base_url = "{url}"
model = "judge"
requests_per_minute = 0
[[checkpoints]]
name = "fixture"
path = "fixture-checkpoint"
revision = "fixture-v1"
tokenizer_revision = "fixture-tokenizer-v1"
chat_template_revision = "fixture-chat-v1"
[checkpoints.endpoint]
base_url = "{url}"
model = "target"
requests_per_minute = 0
[[benchmarks]]
name = "simpleqa"
source_file = "questions.jsonl"
source_origin = "synthetic-test-fixture"
source_revision = "fixture-v1"
data_license = "test-fixture"
""", encoding="utf-8")
        return path

    def command(self, command, manifest, extra=None):
        args = [command, "--suite", str(manifest), "--bundle", str(self.root / "bundle")]
        if command == "run":
            args += ["--output", str(self.root / "run")]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return cli.main(args + (extra or []))

    def test_prepare_http_run_failure_reporting_and_resume(self):
        for index in range(4):
            core.append_jsonl(self.root / "questions.jsonl", {"id": str(index), "problem": f"q{index}", "answer": "gold"})
        grades = ['{"outcome":"CORRECT"}', '{"outcome":"INCORRECT"}', '{"outcome":"NOT_ATTEMPTED"}', "invalid"]
        with LocalAPI(["a"] * 4, grades) as api:
            manifest = self.manifest(api.url)
            self.assertEqual(self.command("prepare", manifest), 0)
            self.assertEqual(self.command("run", manifest), 1)
            summary = core.read_json(self.root / "run" / "summary.json")
            result = summary["results"]["fixture"]["simpleqa"]
            self.assertEqual(result["scored_cases"], 3)
            self.assertEqual(result["failed_cases"], 1)
            self.assertAlmostEqual(result["metrics"]["not_attempted"]["rate"], 1 / 3)
            api.judge_outputs.append('{"outcome":"CORRECT"}')
            self.assertEqual(self.command("run", manifest, ["--resume"]), 0)
            self.assertEqual(sum(row["model"] == "target" for row in api.requests), 4)
            self.assertTrue((self.root / "run" / "summary.csv").exists())
            events = core.read_jsonl(self.root / "run" / "fixture" / "simpleqa" / "samples.jsonl")
            self.assertTrue(any(row.get("raw_judge") == "invalid" for row in events))
            self.assertEqual(self.command("run", manifest), 1)

    def test_bundle_content_tampering_is_rejected(self):
        core.append_jsonl(self.root / "questions.jsonl", {"problem": "q", "answer": "a"})
        manifest = self.manifest("http://127.0.0.1:1/v1")
        self.assertEqual(self.command("prepare", manifest), 0)
        path = self.root / "bundle" / "simpleqa.json"
        content = core.read_json(path)
        content["cases"][0]["gold"] = "different"
        core.write_json(path, content)
        self.assertEqual(self.command("run", manifest), 1)
        self.assertFalse((self.root / "run").exists())

    def test_self_judging_is_refused_before_requests(self):
        core.append_jsonl(self.root / "questions.jsonl", {"problem": "q", "answer": "a"})
        manifest = self.manifest("http://127.0.0.1:1/v1")
        manifest.write_text(manifest.read_text().replace('model = "judge"', 'model = "target"'))
        self.assertEqual(self.command("prepare", manifest), 0)
        self.assertEqual(self.command("run", manifest), 1)
        self.assertFalse((self.root / "run").exists())

    def test_generation_truncation_is_not_a_successful_abstention(self):
        with LocalAPI([("partial", "length")], []) as api:
            client = core.ChatClient({"base_url": api.url, "model": "target", "requests_per_minute": 0})
            with self.assertRaisesRegex(ValueError, "truncated"):
                client.complete([{"role": "user", "content": "q"}])

    def test_resume_rejects_changed_target_identity(self):
        core.append_jsonl(self.root / "questions.jsonl", {"problem": "q", "answer": "a"})
        with LocalAPI(["a"], ['{"outcome":"CORRECT"}']) as api:
            manifest = self.manifest(api.url)
            self.assertEqual(self.command("prepare", manifest), 0)
            self.assertEqual(self.command("run", manifest), 0)
            suite = cli.load_suite(manifest)
            data = cli.load_bundle(suite, self.root / "bundle", suite["benchmarks"])
            suite["checkpoints"][0]["endpoint"]["model"] = "different"
            with self.assertRaisesRegex(ValueError, "Cannot resume"):
                cli.run(suite, data, self.root / "run", suite["checkpoints"], suite["benchmarks"], resume=True)

    def test_unverified_checkpoint_provenance_is_refused(self):
        core.append_jsonl(self.root / "questions.jsonl", {"problem": "q", "answer": "a"})
        manifest = self.manifest("http://127.0.0.1:1/v1")
        manifest.write_text(manifest.read_text().replace("fixture-chat-v1", "REPLACE_WITH_VERIFIED_TEMPLATE"))
        self.assertEqual(self.command("prepare", manifest), 0)
        self.assertEqual(self.command("run", manifest), 1)

    def test_secret_is_not_saved_in_public_endpoint(self):
        with patch.dict(os.environ, {"TEST_EVAL_KEY": "synthetic-secret"}):
            config = {"base_url": "http://127.0.0.1:1/v1", "model": "target", "api_key_env": "TEST_EVAL_KEY"}
            self.assertNotIn("synthetic-secret", json.dumps(core.public_endpoint(config)))

    def test_endpoint_urls_do_not_accept_embedded_credentials(self):
        for url in ("https://key@example.test/v1", "https://example.test/v1?api_key=secret"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                core.endpoint_url({"base_url": url})


@unittest.skipUnless(importlib.util.find_spec("lm_eval"), "requires the pinned native evaluation environment")
class NativeIntegrationTests(unittest.TestCase):
    def test_real_alpaca_annotation_and_length_control_through_local_judge(self):
        directory = os.environ.get("HELPFULNESS_TEST_ALPACA_DATA")
        if not directory:
            self.skipTest("stage AlpacaEval features explicitly and set HELPFULNESS_TEST_ALPACA_DATA")
        responses = []
        for index in range(8):
            probability = 0.2 if index % 2 else 0.8
            responses.append({
                "choices": [{
                    "index": 0, "message": {"role": "assistant", "content": "m"},
                    "finish_reason": "stop",
                    "logprobs": {"content": [{
                        "token": "m", "logprob": math.log(probability),
                        "top_logprobs": [
                            {"token": "m", "logprob": math.log(probability)},
                            {"token": "M", "logprob": math.log(1 - probability)},
                        ],
                    }]},
                }],
                "usage": {"total_tokens": 1, "prompt_tokens": 0, "completion_tokens": 1},
            })
        items = [{
            "instruction": f"Fixture instruction {index}", "completion": "candidate " * (index + 1),
            "reference_output": "reference output", "dataset": "test-fixture",
        } for index in range(8)]
        with tempfile.TemporaryDirectory() as root, LocalAPI([], responses) as api:
            judge = {
                "base_url": api.url, "model": "local/judge", "requests_per_minute": 0,
                "api_key_env": "TEST_NATIVE_JUDGE_KEY",
            }
            with (
                patch.dict(os.environ, {"TEST_NATIVE_JUDGE_KEY": "synthetic-test-key"}),
                native.offline_alpaca_assets(Path(directory)),
            ):
                score = native.alpaca_aggregation(items, judge, Path(root))
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)
            annotations = core.read_json(Path(root) / "validated_annotations.json")
            self.assertEqual(len(annotations), 8)
            self.assertEqual(len(api.requests), 8)
            self.assertTrue(all(request["model"] == "local/judge" for request in api.requests))
            self.assertTrue((Path(root) / "glm_weights").exists())
            self.assertNotIn("synthetic-test-key", (Path(root) / "annotator" / "client-config.yaml").read_text())

    def test_real_ifbench_second_turn_scorer(self):
        case = {
            "key": 1, "prompt": "Use no whitespace.",
            "instruction_id_list": ["format:no_whitespace"], "kwargs": [{}],
        }
        directory = os.environ.get("HELPFULNESS_TEST_NLTK_DATA")
        if not directory:
            self.skipTest("stage NLTK assets explicitly and set HELPFULNESS_TEST_NLTK_DATA")
        with native.offline_scorers(Path(directory)):
            scores, _ = adapters.score_case("ifbench_multiturn", case, ["kiwi", "no fruit"], None)
            self.assertFalse(scores["prompt_level_strict_acc"])
            self.assertEqual(scores["inst_level_strict_acc"], [False])
            scores, _ = adapters.score_case("ifbench_multiturn", case, ["no fruit", "kiwi"], None)
            self.assertTrue(scores["prompt_level_strict_acc"])

    @native.environment({"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    def test_real_ifeval_harness_through_local_chat_api(self):
        from lm_eval.api.task import ConfigurableTask
        from lm_eval.tasks.ifeval import utils

        with tempfile.TemporaryDirectory() as directory, LocalAPI(["kiwi"], []) as api:
            root = Path(directory)
            doc = {
                "key": 1, "prompt": "Include kiwi.", "instruction_id_list": ["keywords:existence"],
                "kwargs": [{"keywords": ["kiwi"]}],
            }
            source = root / "cases.jsonl"
            core.append_jsonl(source, doc)
            task = ConfigurableTask(config={
                "task": "ifeval", "dataset_path": "json",
                "dataset_kwargs": {"data_files": {"test": str(source)}},
                "test_split": "test", "output_type": "generate_until",
                "doc_to_text": "prompt", "doc_to_target": 0,
                "process_results": utils.process_results,
                "generation_kwargs": {"until": [], "temperature": 0, "max_gen_toks": 128},
                "metric_list": [{
                    "metric": key,
                    "aggregation": utils.agg_inst_level_acc if key.startswith("inst_") else "mean",
                    "higher_is_better": True,
                } for key in adapters.INSTRUCTION_METRICS],
            })
            target = {"base_url": api.url, "model": "local/target", "requests_per_minute": 0}
            with patch.object(native, "load_tasks", return_value={"ifeval": task}):
                result = native.run_native("ifeval", {"ifeval": [doc]}, target, {}, root)
            summary = adapters.summarize_native("ifeval", result, {"ifeval": 1})
            self.assertEqual(summary["status"], "succeeded", repr(summary) + repr(result.get("samples")))
            self.assertEqual(summary["metrics"]["ifeval/prompt_level_strict_acc"], 1)
            self.assertEqual(summary["denominators"]["ifeval/inst_level_strict_acc"], 1)

    def test_real_mtbench_harness_preserves_turn_two_and_judgments(self):
        doc = {"question_id": 81, "category": "writing", "turns": ["first question", "second question"]}
        with tempfile.TemporaryDirectory() as directory, LocalAPI(["first answer", "second answer"], ["[[8]]", "[[6]]"]) as api:
            root = Path(directory)
            task = native.load_tasks("mt_bench")["mt_bench"]
            target = {"base_url": api.url, "model": "local/target", "requests_per_minute": 0}
            judge = {
                "base_url": api.url, "model": "local/judge", "requests_per_minute": 0,
                "api_key_env": "TEST_NATIVE_JUDGE_KEY",
            }
            with (
                patch.dict(os.environ, {"TEST_NATIVE_JUDGE_KEY": "synthetic-test-key"}),
                patch.object(native, "load_tasks", return_value={"mt_bench": task}),
            ):
                result = native.run_native("mt_bench", {"mt_bench": [doc]}, target, judge, root)
            summary = adapters.summarize_native("mt_bench", result, {"mt_bench": 1})
            self.assertEqual(summary["status"], "succeeded")
            self.assertEqual(summary["metrics"]["mt_bench/mt_bench_turn_2_score"], 6)
            generation = [request for request in api.requests if request["model"] == "local/target"]
            self.assertEqual(generation[1]["messages"][-2]["content"], "first answer")
            self.assertTrue((root / "mt_bench_judgments.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
