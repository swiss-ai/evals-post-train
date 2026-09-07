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
        build_script = (REPO_ROOT / "scripts" / "build_eval_env.sh").read_text(
            encoding="utf-8"
        )
        evaluate_script = (REPO_ROOT / "scripts" / "evaluate.sbatch").read_text(
            encoding="utf-8"
        )

        self.assertIn('HARNESS_ARCHIVE="$EVAL_ENV_CACHE_ROOT/harness/', build_script)
        self.assertIn('tar -C "$HARNESS_OVERLAY" -cf "$archive_tmp" .', build_script)
        self.assertIn("EVAL_HARNESS_ARCHIVE", evaluate_script)
        self.assertIn("SLURM_TMPDIR", evaluate_script)
        self.assertIn("EVAL_RUNTIME_HARNESS_OVERLAY", evaluate_script)


if __name__ == "__main__":
    unittest.main()
