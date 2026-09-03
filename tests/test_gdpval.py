"""GDPval task (inspect_evals/gdpval -- a standard task from the upstream
inspect_evals package, not repo-owned code): its @task decorator hardcodes a
Docker sandbox (see its own Dockerfile, whose build takes up to 10 minutes
per the README's "Docker Build Times" note), which is too heavy for a
routine test. This swaps in a local sandbox instead -- mockllm's fixed
response never calls the bash/python tools the task wires up, so nothing
actually touches the sandbox besides start-up/teardown -- to confirm the
task still constructs, loads its real HuggingFace dataset, and runs
end-to-end through inspect_ai's own eval() without errors: i.e. that
nothing in the plumbing this repo depends on (dataset loading, tool
wiring, the deliverable-extraction solver, the scorer, the
consolidate-deliverables hook) is broken."""

import shutil
import unittest
from pathlib import Path

from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.util import SandboxEnvironmentSpec
from inspect_evals.gdpval import util as gdpval_util
from inspect_evals.gdpval.gdpval import gdpval


class GdpvalIntegrationTests(unittest.TestCase):
    """Runs the real, installed inspect_evals/gdpval task through
    inspect_ai's own eval() -- see test_omniscience.py's module docstring
    for why an integration run through eval() is needed rather than
    exercising pieces in isolation."""

    def setUp(self):
        # The task's consolidate-deliverables hook always writes a
        # timestamped output folder under the installed package directory
        # (see gdpval's README: "An output folder with the deliverable
        # files will be created"). Track what's there before the run so
        # tearDown can remove only what this test added.
        self._upload_dir = Path(gdpval_util.__file__).parent / "gdpval_hf_upload"
        self._existing_upload_dirs = (
            set(self._upload_dir.iterdir()) if self._upload_dir.exists() else set()
        )

    def tearDown(self):
        if self._upload_dir.exists():
            for created in set(self._upload_dir.iterdir()) - self._existing_upload_dirs:
                shutil.rmtree(created, ignore_errors=True)

    def test_task_runs_end_to_end_with_a_mock_model(self):
        task = gdpval()
        # Swap the real Docker sandbox for a local one -- mockllm never
        # calls the bash/python tools the task provides, so nothing actually
        # executes inside it; this just avoids the multi-minute image build.
        task.sandbox = SandboxEnvironmentSpec("local")

        model = get_model(
            "mockllm/model",
            custom_outputs=[ModelOutput.from_content("mockllm", "the deliverable")],
        )
        logs = inspect_eval(
            task,
            model=model,
            limit=1,
            display="none",
            log_dir="/tmp/test_gdpval_integration_logs",
        )

        self.assertEqual(logs[0].status, "success")
        self.assertIsNone(logs[0].error)


if __name__ == "__main__":
    unittest.main()
