import unittest

from scripts.alignment.wandb_names import (
    WANDB_ARTIFACT_NAME_LIMIT,
    WANDB_RUN_ID_MAX_LENGTH,
    make_sample_table_key,
    make_wandb_run_id,
    maximum_generated_artifact_name_length,
)


class WandbNamesTests(unittest.TestCase):
    def test_reported_qwen_failure_keeps_readable_task_key(self):
        model = "Qwen3.5-27B-release_evals-best_of_k-no-think"
        task = "gsm8k_cot_zeroshot_self_consistency"

        run_id = make_wandb_run_id(model)
        key = make_sample_table_key(run_id, task)

        self.assertEqual(run_id, f"{model}-001")
        self.assertEqual(key, f"samples/{task}")
        self.assertLessEqual(
            maximum_generated_artifact_name_length(run_id, key),
            WANDB_ARTIFACT_NAME_LIMIT,
        )

    def test_long_model_and_task_names_are_stably_shortened(self):
        model = (
            "Apertus-1.5-70B-SFT-RL-DPO-FINAL-with-an-extra-long-checkpoint-name-"
            "release_evals-best_of_k-no-think"
        )
        other_model = model.removesuffix("no-think") + "think"
        first_task = "very_long_benchmark_variant_self_consistency_first"
        second_task = "very_long_benchmark_variant_self_consistency_second"

        run_id = make_wandb_run_id(model)
        first_key = make_sample_table_key(run_id, first_task)
        second_key = make_sample_table_key(run_id, second_task)

        self.assertEqual(len(run_id), WANDB_RUN_ID_MAX_LENGTH)
        self.assertEqual(run_id, make_wandb_run_id(model))
        self.assertNotEqual(run_id, make_wandb_run_id(other_model))
        self.assertNotEqual(first_key, second_key)
        for key in (first_key, second_key):
            self.assertTrue(key.startswith("samples/"))
            self.assertLessEqual(
                maximum_generated_artifact_name_length(run_id, key),
                WANDB_ARTIFACT_NAME_LIMIT,
            )


if __name__ == "__main__":
    unittest.main()
