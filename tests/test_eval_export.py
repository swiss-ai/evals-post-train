import json
import tempfile
import unittest
from pathlib import Path

from scripts.eval_export.exporter import ExportError, export_results, validate_export


def fixture_results() -> dict:
    return {
        "results": {
            "mmlu_pro": {
                "alias": "mmlu_pro",
                "exact_match,none": 0.42,
                "exact_match_stderr,none": 0.01,
            },
            "gsm8k_cot": {
                "alias": "gsm8k_cot",
                "exact_match,strict-match": 0.75,
                "exact_match_stderr,strict-match": 0.02,
            },
            "unmapped_benchmark": {
                "alias": "unmapped_benchmark",
                "acc,none": 0.5,
            },
        },
        "configs": {
            "mmlu_pro": {
                "task": "mmlu_pro",
                "dataset_path": "TIGER-Lab/MMLU-Pro",
                "test_split": "test",
                "num_fewshot": 5,
                "output_type": "generate_until",
                "doc_to_text": "Question: {{question}}",
                "generation_kwargs": {
                    "do_sample": False,
                    "temperature": 0.0,
                    "max_gen_toks": 1024,
                },
            },
            "gsm8k_cot": {
                "task": "gsm8k_cot",
                "dataset_path": "openai/gsm8k",
                "test_split": "test",
                "num_fewshot": 8,
                "output_type": "generate_until",
                "generation_kwargs": {
                    "do_sample": False,
                    "max_new_tokens": 512,
                },
            },
        },
        "higher_is_better": {
            "mmlu_pro": {"exact_match": True},
            "gsm8k_cot": {"exact_match": True},
        },
        "n-samples": {
            "mmlu_pro": {"original": 100, "effective": 100},
            "gsm8k_cot": {"original": 200, "effective": 200},
        },
        "n-shot": {"mmlu_pro": 5, "gsm8k_cot": 8},
        "config": {
            "model": "vllm",
            "model_args": "pretrained=swiss-ai/Test-Model,dtype=bfloat16",
            "limit": None,
            "model_revision": "main",
            "model_sha": "abc123",
            "model_dtype": "bfloat16",
        },
        "model_name": "swiss-ai/Test-Model",
        "lm_eval_version": "0.4.9.2",
        "git_hash": "deadbee",
        "date": 1768964383.0,
        "task_hashes": {"mmlu_pro": "hash-mmlu", "gsm8k_cot": "hash-gsm"},
        "fewshot_as_multiturn": True,
        "chat_template": "{{ messages }}",
        "chat_template_sha": "chat-sha",
    }


def fixture_sample() -> dict:
    return {
        "doc_id": 7,
        "target": "4",
        "arguments": {"gen_args_0": {"arg_0": "What is 2 + 2?"}},
        "filtered_resps": [["4"]],
        "metrics": ["exact_match"],
        "exact_match": 1.0,
        "filter": "strict-match",
        "doc_hash": "doc",
        "prompt_hash": "prompt",
        "target_hash": "target",
    }


class EvalExportTests(unittest.TestCase):
    def test_export_writes_canonical_records_samples_and_hf_previews(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            run.mkdir()
            results = run / "results_2026-01-21T03-44-18.json"
            results.write_text(json.dumps(fixture_results()), encoding="utf-8")
            (run / "samples_gsm8k_cot_2026-01-21T03-44-18.jsonl").write_text(
                json.dumps(fixture_sample()) + "\n", encoding="utf-8"
            )
            output = root / "export"

            manifest = export_results(
                results,
                output,
                include_samples=True,
                retrieved_timestamp="1770000000.0",
            )

            self.assertFalse(manifest["publishing_performed"])
            self.assertEqual(manifest["skipped_unmapped_tasks"], ["unmapped_benchmark"])
            self.assertEqual(
                {item["benchmark"] for item in manifest["records"]},
                {"MMLU-Pro", "gsm8k"},
            )
            self.assertTrue((output / "manifest.yaml").is_file())
            self.assertFalse((output / "manifest.json").exists())
            self.assertEqual(validate_export(output), [])

            mmlu_item = next(
                item for item in manifest["records"] if item["benchmark"] == "MMLU-Pro"
            )
            mmlu = json.loads(
                (output / mmlu_item["eee_record"]).read_text(encoding="utf-8")
            )
            result = mmlu["evaluation_results"][0]
            self.assertEqual(
                mmlu["source_metadata"]["evaluator_relationship"], "first_party"
            )
            self.assertEqual(result["evaluation_name"], "MMLU-Pro")
            self.assertEqual(result["source_data"]["dataset_name"], "MMLU-Pro")
            self.assertEqual(result["metric_config"]["metric_id"], "mmlu_pro/overall")
            self.assertEqual(
                result["generation_config"]["generation_args"]["max_tokens"], 1024
            )
            self.assertEqual(
                result["generation_config"]["generation_args"]["temperature"], 0.0
            )
            self.assertEqual(result["score_details"]["uncertainty"]["num_samples"], 100)

            gsm_item = next(
                item for item in manifest["records"] if item["benchmark"] == "gsm8k"
            )
            sample_lines = (
                (output / gsm_item["instance_results"])
                .read_text(encoding="utf-8")
                .splitlines()
            )
            sample = json.loads(sample_lines[0])
            self.assertEqual(sample["evaluation_name"], "gsm8k/strict-match")
            self.assertEqual(sample["input"]["raw"], "What is 2 + 2?")
            self.assertTrue(sample["evaluation"]["is_correct"])
            gsm_record = json.loads(
                (output / gsm_item["eee_record"]).read_text(encoding="utf-8")
            )
            self.assertEqual(len(gsm_record["evaluation_results"]), 1)
            self.assertEqual(
                gsm_record["evaluation_results"][0]["generation_config"][
                    "generation_args"
                ]["max_tokens"],
                512,
            )

            hf_yaml = (output / gsm_item["huggingface_preview"]).read_text(
                encoding="utf-8"
            )
            self.assertIn('id: "openai/gsm8k"', hf_yaml)
            self.assertIn("value: 0.75", hf_yaml)
            self.assertIn("/flat/objects/", hf_yaml)

    def test_validate_accepts_legacy_manifest_and_export_migrates_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results.json"
            results.write_text(json.dumps(fixture_results()), encoding="utf-8")
            output = root / "export"

            export_results(
                results,
                output,
                retrieved_timestamp="1770000000.0",
            )
            current = output / "manifest.yaml"
            legacy = output / "manifest.json"
            current.rename(legacy)

            self.assertEqual(validate_export(output), [])

            export_results(
                results,
                output,
                retrieved_timestamp="1770000000.0",
            )
            self.assertTrue(current.is_file())
            self.assertFalse(legacy.exists())

    def test_local_checkpoint_requires_model_override(self):
        raw = fixture_results()
        raw["model_name"] = "/checkpoints/model"
        raw["config"]["model_args"] = "pretrained=/checkpoints/model"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results_local.json"
            results.write_text(json.dumps(raw), encoding="utf-8")

            with self.assertRaisesRegex(ExportError, "--model-id"):
                export_results(results, root / "failed")

            manifest = export_results(
                results,
                root / "ok",
                model_id_override="swiss-ai/Released-Model",
                retrieved_timestamp="1770000000.0",
            )
            self.assertEqual(manifest["model_id"], "swiss-ai/Released-Model")

    def test_external_model_defaults_to_third_party(self):
        raw = fixture_results()
        raw["model_name"] = "external-org/External-Model"
        raw["config"]["model_args"] = "pretrained=external-org/External-Model"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results_external.json"
            results.write_text(json.dumps(raw), encoding="utf-8")
            manifest = export_results(
                results,
                root / "export",
                retrieved_timestamp="1770000000.0",
            )
            record_path = root / "export" / manifest["records"][0]["eee_record"]
            record = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertEqual(
                record["source_metadata"]["evaluator_relationship"],
                "third_party",
            )

    def test_strict_mapping_rejects_unmapped_tasks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results.json"
            results.write_text(json.dumps(fixture_results()), encoding="utf-8")
            with self.assertRaisesRegex(ExportError, "unmapped_benchmark"):
                export_results(
                    results,
                    root / "export",
                    strict_mappings=True,
                )

    def test_self_consistency_suffix_uses_mean_at_logged_repeats(self):
        raw = fixture_results()
        task_name = "gsm8k_cot_zeroshot_self_consistency"
        raw["results"] = {
            task_name: {
                "exact_match,score-first": 0.7,
                "exact_match,mean@32": 0.8,
                "exact_match,maj@32": 0.85,
                "exact_match,pass@32": 0.9,
            }
        }
        raw["configs"] = {
            task_name: {
                "task": task_name,
                "dataset_path": "openai/gsm8k",
                "test_split": "test",
                "num_fewshot": 0,
                "repeats": 32,
                "output_type": "generate_until",
                "generation_kwargs": {
                    "do_sample": True,
                    "temperature": 0.6,
                    "max_gen_toks": 32768,
                },
            }
        }
        raw["n-samples"] = {task_name: {"original": 1319, "effective": 1319}}
        raw["higher_is_better"] = {task_name: {"exact_match": True}}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results_self_consistency.json"
            results.write_text(json.dumps(raw), encoding="utf-8")

            manifest = export_results(
                results,
                root / "export",
                retrieved_timestamp="1770000000.0",
            )

            self.assertEqual(
                manifest["resolved_task_aliases"],
                {task_name: "gsm8k_cot_zeroshot"},
            )
            self.assertEqual(manifest["skipped_unmapped_tasks"], [])
            record_path = root / "export" / manifest["records"][0]["eee_record"]
            record = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertEqual(len(record["evaluation_results"]), 1)
            result = record["evaluation_results"][0]
            self.assertEqual(result["evaluation_name"], "gsm8k/mean@32")
            self.assertEqual(result["score_details"]["score"], 0.8)
            details = result["generation_config"]["additional_details"]
            self.assertEqual(details["repeats"], "32")
            self.assertEqual(details["evaluation_variant"], "self_consistency")

    def test_all_unmapped_error_lists_tasks_and_mapping_file(self):
        raw = fixture_results()
        raw["results"] = {
            "aime24": {"exact_match,none": 0.25},
            "aime25": {"exact_match,none": 0.1},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results_aime.json"
            results.write_text(json.dumps(raw), encoding="utf-8")

            with self.assertRaises(ExportError) as caught:
                export_results(results, root / "export")

            message = str(caught.exception)
            self.assertIn("No exportable benchmark results", message)
            self.assertIn("Detected 2 task(s)", message)
            self.assertIn("aime24", message)
            self.assertIn("aime25", message)
            self.assertIn("task_mappings.json", message)
            self.assertIn("exact task names", message)


if __name__ == "__main__":
    unittest.main()
