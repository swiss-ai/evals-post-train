import shlex
import unittest
from unittest.mock import patch

from scripts import launch_judge
from swiss_ai_model_launch import JobStatus, LaunchArgs


class LaunchArgsTests(unittest.TestCase):
    def test_nodes_override_is_translated_to_sml_topology(self) -> None:
        args = launch_judge._build_launch_args("qwen3.5-27b", {"nodes": 3})

        self.assertIs(type(args), LaunchArgs)
        self.assertEqual(args.topology.replicas, 1)
        self.assertEqual(args.topology.nodes_per_replica, 3)
        self.assertEqual(args.total_nodes, 3)
        self.assertIn("--nodes=3", args.to_sbatch_args())

    def test_sml_injects_the_framework_port(self) -> None:
        args = launch_judge._build_launch_args("qwen3.5-27b", {})

        self.assertNotIn("--port", args.framework_args)
        self.assertTrue(args.to_job_env()["FRAMEWORK_ARGS"].startswith("--port 8080 "))

    def test_presets_use_registry_paths_and_consistent_served_names(self) -> None:
        for preset_name, preset in launch_judge.JUDGE_PRESETS.items():
            with self.subTest(preset=preset_name):
                tokens = shlex.split(preset["framework_args"])
                model_path = tokens[tokens.index("--model") + 1]
                framework_served_name = tokens[
                    tokens.index("--served-model-name") + 1
                ]

                self.assertTrue(
                    model_path.startswith(f"{launch_judge.MODEL_REGISTRY}/")
                )
                self.assertEqual(framework_served_name, preset["served_model_name"])
                self.assertEqual(
                    preset["served_model_name"],
                    preset["served_model_name"].strip(),
                )
                self.assertIn("--host", tokens)


class _FailedLauncher:
    last_init_kwargs = None

    def __init__(self, **kwargs: object) -> None:
        type(self).last_init_kwargs = kwargs

    async def launch_with_args(self, args: LaunchArgs) -> tuple[int, str]:
        return 12345, args.served_model_name

    async def get_job_status(self, _job_id: int) -> JobStatus:
        return JobStatus.FAILED

    async def get_job_logs(self, _job_id: int) -> tuple[str, str]:
        return "master stdout", "master stderr"

    async def read_job_file(self, _job_id: int, filename: str) -> str | None:
        return {
            "replica_0.out": "vLLM stdout",
            "replica_0.err": "vLLM startup error",
        }.get(filename)


class _UnknownLauncher(_FailedLauncher):
    last_instance = None

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.status_polls = 0
        type(self).last_instance = self

    async def get_job_status(self, _job_id: int) -> JobStatus:
        self.status_polls += 1
        return JobStatus.UNKNOWN


class LaunchFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_terminal_failure_raises_immediately_with_logs(self) -> None:
        args = launch_judge._build_launch_args("qwen3.5-27b", {})

        with (
            patch.object(launch_judge, "SlurmLauncher", _FailedLauncher),
            self.assertRaises(RuntimeError) as raised,
        ):
            await launch_judge.launch_judge(
                args,
                api_key="unused",
                health_timeout=900,
                health_interval=15,
                reservation="test-reservation",
            )

        message = str(raised.exception)
        self.assertEqual(
            _FailedLauncher.last_init_kwargs["reservation"],
            "test-reservation",
        )
        self.assertIn("SLURM status: FAILED", message)
        self.assertIn("master stderr", message)
        self.assertIn("vLLM startup error", message)

    async def test_persistent_unknown_status_uses_grace_period_then_shows_logs(
        self,
    ) -> None:
        args = launch_judge._build_launch_args("qwen3.5-27b", {})

        with (
            patch.object(launch_judge, "SlurmLauncher", _UnknownLauncher),
            self.assertRaises(RuntimeError) as raised,
        ):
            await launch_judge.launch_judge(
                args,
                api_key="unused",
                health_timeout=900,
                health_interval=0,
            )

        self.assertEqual(
            _UnknownLauncher.last_instance.status_polls,
            launch_judge.UNKNOWN_STATUS_GRACE_POLLS,
        )
        self.assertIn("SLURM status: UNKNOWN", str(raised.exception))
        self.assertIn("vLLM startup error", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
