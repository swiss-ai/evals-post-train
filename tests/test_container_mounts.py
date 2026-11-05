import tomllib
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTAINER_CONFIGS = (
    "env_hf.toml",
    "env_eval.toml",
    "env_vllm.toml",
    "env_sglang.toml",
)


def container_path(mount: str) -> str:
    """Return the destination of an identity or host:container mount."""

    return mount.rsplit(":", 1)[-1]


def path_is_mounted(mounts: list[str], required: str) -> bool:
    return any(
        required == destination
        or required.startswith(destination.rstrip("/") + "/")
        for destination in map(container_path, mounts)
    )


class ContainerMountTests(unittest.TestCase):
    def test_eval_repo_and_default_scratch_cache_are_visible(self):
        required_paths = (
            "/users/${USER}",
            "/iopsstor/scratch/cscs/${USER}",
        )
        for filename in CONTAINER_CONFIGS:
            with self.subTest(config=filename):
                path = REPO_ROOT / "containers" / filename
                with path.open("rb") as handle:
                    mounts = tomllib.load(handle)["mounts"]
                for required in required_paths:
                    self.assertTrue(
                        path_is_mounted(mounts, required),
                        f"{filename} does not expose {required}",
                    )

    def test_harness_overlay_is_archived_and_staged_on_node_local_storage(self):
        build_script = (REPO_ROOT / "scripts" / "lm_eval_harness" / "build_eval_env.sh").read_text(
            encoding="utf-8"
        )
        evaluate_script = (REPO_ROOT / "scripts" / "lm_eval_harness" / "run_lm_eval.sbatch").read_text(
            encoding="utf-8"
        )

        self.assertIn('HARNESS_ARCHIVE="$EVAL_ENV_CACHE_ROOT/harness/', build_script)
        self.assertIn('tar -C "$HARNESS_OVERLAY" -cf "$archive_tmp" .', build_script)
        self.assertIn("EVAL_HARNESS_ARCHIVE", evaluate_script)
        self.assertIn("SLURM_TMPDIR", evaluate_script)
        self.assertIn("EVAL_RUNTIME_HARNESS_OVERLAY", evaluate_script)

    def test_environment_build_ignores_the_user_site(self):
        # Jobs point PYTHONUSERBASE elsewhere, so a package pip found in ~/.local is missing there.
        build_script = (REPO_ROOT / "scripts" / "lm_eval_harness" / "build_eval_env.sh").read_text(
            encoding="utf-8"
        )
        self.assertLess(build_script.index("export PYTHONNOUSERSITE=1"), build_script.index("python -m venv"))

    def test_model_free_work_runs_in_the_prebuilt_image(self):
        # The openai backend, the result merge and the W&B upload share one image, tagged with the
        # commit the scripts come from; its venv is on PATH only for a non-login shell.
        with (REPO_ROOT / "containers" / "env_eval.toml").open("rb") as handle:
            image = tomllib.load(handle)["image"]
        self.assertEqual(image, "ghcr.io#swiss-ai/evals-post-train-eval:${EVAL_IMAGE_TAG}")
        scripts = REPO_ROOT / "scripts" / "lm_eval_harness"
        evaluate_script = (scripts / "run_lm_eval.sbatch").read_text(encoding="utf-8")
        self.assertIn('openai) CONTAINER_ENV=./containers/env_eval.toml; CONTAINER_SHELL="bash -c"', evaluate_script)
        self.assertIn("--environment=./containers/env_eval.toml bash -c", evaluate_script)
        aggregate_script = (scripts / "aggregate_chunks.sbatch").read_text(encoding="utf-8")
        self.assertEqual(aggregate_script.count("--environment=./containers/env_eval.toml bash -c"), 2)
        inspect_script = (REPO_ROOT / "scripts" / "inspect_ai" / "run_inspect_eval.sbatch").read_text(encoding="utf-8")
        self.assertIn("--environment=./containers/env_eval.toml", inspect_script)
        self.assertNotIn("env.toml", evaluate_script + aggregate_script + inspect_script)


if __name__ == "__main__":
    unittest.main()
