import json
import tempfile
import unittest
from pathlib import Path

from scripts.eval_export.exporter import (
    ExportError,
    _evaluation_name,
    _internal_task_mapping,
    export_results,
    validate_export,
)


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
            "unmapped_benchmark": {
                "task": "unmapped_benchmark",
                "dataset_path": "internal-org/unmapped-benchmark",
                "test_split": "test",
                "output_type": "multiple_choice",
            },
        },
        "higher_is_better": {
            "mmlu_pro": {"exact_match": True},
            "gsm8k_cot": {"exact_match": True},
            "unmapped_benchmark": {"acc": True},
        },
        "n-samples": {
            "mmlu_pro": {"original": 100, "effective": 100},
            "gsm8k_cot": {"original": 200, "effective": 200},
            "unmapped_benchmark": {"original": 50, "effective": 50},
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
    def test_evaluation_name_scheme_handles_optional_composite_and_defaults(self):
        self.assertEqual(
            _evaluation_name({"benchmark": "MATH"}),
            "MATH.MATH.overall",
        )
        self.assertEqual(
            _evaluation_name(
                {
                    "composite": "ArtificialAnalysis",
                    "family": "MMLU",
                    "benchmark": "MMLU-Pro",
                    "split": "algebra",
                }
            ),
            "ArtificialAnalysis.MMLU.MMLU-Pro.algebra",
        )

    def test_internal_task_name_must_be_a_safe_identifier(self):
        with self.assertRaisesRegex(ExportError, "safe internal benchmark"):
            _internal_task_mapping({}, "../escape")

    def test_internal_task_name_preserves_spaces(self):
        task_name = "include_base_44_north macedonian_gen_0shot"
        mapping = _internal_task_mapping({}, task_name)
        self.assertEqual(mapping["eee"]["benchmark"], task_name)
        self.assertEqual(
            _evaluation_name(mapping["eee"]),
            (
                "include_base_44_north macedonian_gen_0shot."
                "include_base_44_north macedonian_gen_0shot.overall"
            ),
        )

    def test_aggregate_only_task_excludes_nested_subtasks(self):
        group = "include_base_44_gen_0shot"
        subgroup = "include_base_44_south_slavic_gen_0shot"
        leaf_one = "include_base_44_north macedonian_gen_0shot"
        leaf_two = "include_base_44_serbian_gen_0shot"
        raw = fixture_results()
        task_names = [group, subgroup, leaf_one, leaf_two, "aime24"]
        raw["results"] = {name: {"acc,none": 0.5} for name in task_names}
        raw["configs"] = {
            name: {
                "task": name,
                "dataset_path": f"internal/{name}",
                "test_split": "test",
                "output_type": "multiple_choice",
            }
            for name in task_names
        }
        raw["higher_is_better"] = {name: {"acc": True} for name in task_names}
        raw["n-samples"] = {
            name: {"original": 10, "effective": 10} for name in task_names
        }
        raw["group_subtasks"] = {
            group: [subgroup, leaf_one],
            subgroup: [leaf_two],
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results_aggregate_only.json"
            results.write_text(json.dumps(raw), encoding="utf-8")
            manifest = export_results(
                results,
                root / "export",
                aggregate_only_tasks=[group],
                exclude_tasks=["aime24"],
                retrieved_timestamp="1770000000.0",
            )

            self.assertEqual(manifest["aggregate_only_tasks"], [group])
            self.assertEqual(
                manifest["excluded_tasks"],
                sorted(["aime24", subgroup, leaf_one, leaf_two]),
            )
            self.assertEqual(len(manifest["records"]), 1)
            self.assertEqual(manifest["records"][0]["benchmark"], group)
            self.assertEqual(validate_export(root / "export"), [])

    def test_aggregate_only_task_falls_back_to_generated_suite_prefix(self):
        raw = fixture_results()
        aggregate = "include_base_new_45_gen_0shot"
        leaf_one = "include_base_new_45_amharic_gen_0shot"
        leaf_two = "include_base_new_45_czech_gen_0shot"
        raw["results"].update(
            {
                aggregate: {"acc,none": 0.4},
                leaf_one: {"acc,none": 0.3},
                leaf_two: {"acc,none": 0.5},
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results_no_groups.json"
            results.write_text(json.dumps(raw), encoding="utf-8")
            manifest = export_results(
                results,
                root / "export",
                aggregate_only_tasks=["include_base_new_45"],
                retrieved_timestamp="1770000000.0",
            )

            self.assertEqual(manifest["aggregate_only_tasks"], [aggregate])
            self.assertIn(leaf_one, manifest["excluded_tasks"])
            self.assertIn(leaf_two, manifest["excluded_tasks"])
            exported_tasks = {
                task
                for record in manifest["records"]
                for task in record["lm_eval_tasks"]
            }
            self.assertIn(aggregate, exported_tasks)
            self.assertNotIn(leaf_one, exported_tasks)
            self.assertNotIn(leaf_two, exported_tasks)

    def test_aggregate_only_task_requires_a_scored_aggregate(self):
        raw = fixture_results()
        raw["results"]["include_base_44_albanian_gen_0shot"] = {
            "acc,none": 0.5
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results_no_aggregate.json"
            results.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ExportError, "no scored aggregate task"):
                export_results(
                    results,
                    root / "export",
                    aggregate_only_tasks=["include_base_44"],
                )

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
            self.assertEqual(manifest["skipped_unmapped_tasks"], [])
            self.assertEqual(
                manifest["internally_named_tasks"], ["unmapped_benchmark"]
            )
            self.assertEqual(
                {item["benchmark"] for item in manifest["records"]},
                {"MMLU-Pro", "gsm8k", "unmapped_benchmark"},
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
            self.assertEqual(result["evaluation_name"], "MMLU.MMLU-Pro.overall")
            self.assertEqual(
                result["source_data"]["dataset_name"], "TIGER-Lab/MMLU-Pro"
            )
            self.assertEqual(
                result["source_data"]["hf_repo"], "TIGER-Lab/MMLU-Pro"
            )
            self.assertEqual(
                result["source_data"]["additional_details"]["hf_dataset_url"],
                "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro",
            )
            self.assertEqual(mmlu["model_info"]["id"], "swiss-ai/Test-Model")
            self.assertEqual(
                mmlu["model_info"]["additional_details"]["hf_model_url"],
                "https://huggingface.co/swiss-ai/Test-Model",
            )
            self.assertEqual(
                mmlu["model_info"]["additional_details"]["deployment_type"],
                "self_deployed",
            )
            self.assertEqual(
                mmlu["model_info"]["additional_details"]["model_availability"],
                "open_weights",
            )
            self.assertEqual(result["metric_config"]["metric_id"], "mmlu_pro/overall")
            self.assertEqual(
                result["generation_config"]["generation_args"]["max_tokens"], 1024
            )
            self.assertEqual(
                result["generation_config"]["generation_args"]["temperature"], 0.0
            )
            self.assertEqual(result["score_details"]["uncertainty"]["num_samples"], 100)

            internal_item = next(
                item
                for item in manifest["records"]
                if item["benchmark"] == "unmapped_benchmark"
            )
            internal = json.loads(
                (output / internal_item["eee_record"]).read_text(encoding="utf-8")
            )
            internal_result = internal["evaluation_results"][0]
            self.assertEqual(
                internal_result["evaluation_name"],
                "unmapped_benchmark.unmapped_benchmark.overall",
            )
            self.assertEqual(
                internal_result["source_data"]["dataset_name"],
                "internal-org/unmapped-benchmark",
            )
            self.assertEqual(
                internal_result["source_data"]["additional_details"][
                    "hf_dataset_url"
                ],
                "https://huggingface.co/datasets/internal-org/unmapped-benchmark",
            )

            gsm_item = next(
                item for item in manifest["records"] if item["benchmark"] == "gsm8k"
            )
            sample_lines = (
                (output / gsm_item["instance_results"])
                .read_text(encoding="utf-8")
                .splitlines()
            )
            sample = json.loads(sample_lines[0])
            self.assertEqual(sample["evaluation_name"], "gsm8k.gsm8k.overall")
            self.assertEqual(sample["input"]["raw"], "What is 2 + 2?")
            self.assertTrue(sample["evaluation"]["is_correct"])
            gsm_record = json.loads(
                (output / gsm_item["eee_record"]).read_text(encoding="utf-8")
            )
            self.assertEqual(len(gsm_record["evaluation_results"]), 1)
            self.assertTrue(
                gsm_record["detailed_evaluation_results"]["file_path"].startswith(
                    "data/gsm8k/"
                )
            )
            self.assertEqual(
                gsm_record["evaluation_results"][0]["evaluation_name"],
                "gsm8k.gsm8k.overall",
            )
            self.assertEqual(
                gsm_record["evaluation_results"][0]["metric_config"][
                    "evaluation_description"
                ],
                "exact_match (filter: strict-match)",
            )
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

    def test_api_model_does_not_claim_an_unverified_hugging_face_page(self):
        raw = fixture_results()
        raw["model_name"] = "openai/gpt-4o"
        raw["config"]["model"] = "local-completions"
        raw["config"]["model_args"] = (
            "model=openai/gpt-4o,base_url=https://api.example.test/v1/completions"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results_api.json"
            results.write_text(json.dumps(raw), encoding="utf-8")
            manifest = export_results(
                results,
                root / "export",
                retrieved_timestamp="1770000000.0",
            )
            record_path = root / "export" / manifest["records"][0]["eee_record"]
            record = json.loads(record_path.read_text(encoding="utf-8"))
            details = record["model_info"]["additional_details"]
            self.assertEqual(details["deployment_type"], "externally_managed")
            self.assertEqual(details["model_availability"], "unknown")
            self.assertNotIn("hf_model_url", details)

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

    def test_mapped_metric_mismatch_falls_back_to_all_numeric_metrics(self):
        raw = fixture_results()
        raw["results"] = {"mmlu_pro": {"acc_norm,none": 0.33}}
        raw["higher_is_better"] = {"mmlu_pro": {"acc_norm": True}}
        raw["n-samples"] = {"mmlu_pro": {"original": 100, "effective": 100}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results_metric_mismatch.json"
            results.write_text(json.dumps(raw), encoding="utf-8")

            manifest = export_results(
                results,
                root / "export",
                retrieved_timestamp="1770000000.0",
            )

            record_path = root / "export" / manifest["records"][0]["eee_record"]
            record = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertEqual(len(record["evaluation_results"]), 1)
            self.assertEqual(
                record["evaluation_results"][0]["evaluation_name"],
                "MMLU.MMLU-Pro.overall",
            )
            self.assertEqual(
                record["evaluation_results"][0]["metric_config"]["metric_id"],
                "normalized_accuracy",
            )
            self.assertTrue(
                any(
                    "exported all numeric metrics" in warning
                    for warning in manifest["warnings"]
                )
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
            self.assertEqual(result["evaluation_name"], "gsm8k.gsm8k.overall")
            self.assertEqual(result["score_details"]["score"], 0.8)
            details = result["generation_config"]["additional_details"]
            self.assertEqual(details["repeats"], "32")
            self.assertEqual(details["evaluation_variant"], "self_consistency")

    def test_all_unmapped_tasks_export_with_internal_names(self):
        raw = fixture_results()
        raw["results"] = {
            "aime24": {"exact_match,none": 0.25},
            "aime25": {"exact_match,none": 0.1},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results_aime.json"
            results.write_text(json.dumps(raw), encoding="utf-8")

            manifest = export_results(
                results,
                root / "export",
                retrieved_timestamp="1770000000.0",
            )

            self.assertEqual(manifest["skipped_unmapped_tasks"], [])
            self.assertEqual(manifest["internally_named_tasks"], ["aime24", "aime25"])
            self.assertEqual(
                {item["benchmark"] for item in manifest["records"]},
                {"aime24", "aime25"},
            )
            names = set()
            for item in manifest["records"]:
                record = json.loads(
                    (root / "export" / item["eee_record"]).read_text(encoding="utf-8")
                )
                names.add(record["evaluation_results"][0]["evaluation_name"])
            self.assertEqual(
                names,
                {"aime24.aime24.overall", "aime25.aime25.overall"},
            )

    def test_posttrain_final_inventory_is_fully_exported(self):
        task_file = (
            Path(__file__).resolve().parents[1]
            / "configs/apertus/tasks_posttrain_final.txt"
        )
        task_names = [
            line.strip()
            for line in task_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertEqual(len(task_names), 50)

        raw = fixture_results()
        raw["results"] = {name: {"acc,none": 0.5} for name in task_names}
        raw["configs"] = {
            name: {
                "task": name,
                "dataset_path": f"internal/{name}",
                "test_split": "test",
                "output_type": "multiple_choice",
            }
            for name in task_names
        }
        raw["higher_is_better"] = {name: {"acc": True} for name in task_names}
        raw["n-samples"] = {
            name: {"original": 10, "effective": 10} for name in task_names
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results_posttrain_final.json"
            results.write_text(json.dumps(raw), encoding="utf-8")
            manifest = export_results(
                results,
                root / "export",
                retrieved_timestamp="1770000000.0",
            )

            exported_tasks = {
                task
                for record in manifest["records"]
                for task in record["lm_eval_tasks"]
            }
            self.assertEqual(exported_tasks, set(task_names))
            self.assertEqual(len(manifest["records"]), len(task_names))
            self.assertEqual(len(manifest["internally_named_tasks"]), 41)
            self.assertIn("gpqa_main_cot_zeroshot", manifest["internally_named_tasks"])
            self.assertIn("bfcl_v3", manifest["internally_named_tasks"])
            self.assertIn(
                "swiss_ai_charter_alignment", manifest["internally_named_tasks"]
            )

            gpqa_item = next(
                item
                for item in manifest["records"]
                if item["benchmark"] == "gpqa_main_cot_zeroshot"
            )
            gpqa = json.loads(
                (root / "export" / gpqa_item["eee_record"]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                gpqa["evaluation_results"][0]["evaluation_name"],
                "gpqa_main_cot_zeroshot.gpqa_main_cot_zeroshot.overall",
            )


if __name__ == "__main__":
    unittest.main()
