import tomllib
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTAINER_CONFIGS = (
    "env.toml",
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

    def test_locked_environment_is_isolated_from_the_container(self):
        # The container's torchvision and torch libraries are built for its own torch, not the lock's.
        scripts = REPO_ROOT / "scripts" / "lm_eval_harness"
        build_script = (scripts / "build_eval_env.sh").read_text(encoding="utf-8")
        lock_branch = build_script[build_script.index('if [[ "$EVAL_RUNTIME_REQUIREMENTS" == *.lock ]]'):]
        lock_branch = lock_branch[:lock_branch.index("\nfi\n")]
        self.assertIn("BASE_VENV_ARGS=()", lock_branch)
        self.assertIn("grep -vx '.*/torch/lib'", lock_branch)
        self.assertIn('python -m venv "${BASE_VENV_ARGS[@]}"', build_script)
        self.assertIn("isolated venv", build_script[:build_script.index("OVERLAY_KEY=")])
        evaluate_script = (scripts / "run_lm_eval.sbatch").read_text(encoding="utf-8")
        openai_branch = evaluate_script[evaluate_script.rindex('elif [[ $LM_EVAL_BACKEND == "openai" ]]'):]
        self.assertLess(openai_branch.index("grep -vx '.*/torch/lib'"), openai_branch.index("$LOAD_ENV_CMD"))


if __name__ == "__main__":
    unittest.main()
