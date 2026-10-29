"""SWE-bench (Verified) scoring through EvalScope's sandbox service instead of the Docker SDK.

EvalScope's own swe_bench eval_instance talks to a local Docker daemon directly
(DockerClient.from_env(), containers.create without a pull), so it bypasses
the sandbox layer k8s_sandbox.install() swaps out, and it only finds the
instance image if the agent loop happened to pull it first. This is the same
evaluation -- same swebench test spec and eval script, same three patch-apply
fallbacks, same get_eval_report grading -- run in a sandbox created through
SandboxService, so it works unchanged on the docker and k8s engines and pulls
the image itself. install() replaces evalscope.benchmarks.swe_bench.utils.eval_instance
(the agentic adapter imports it at call time), with the same signature and
return shape: {completed, resolved, report}.
"""

from __future__ import annotations

import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

# Same order as evalscope.benchmarks.swe_bench.utils.GIT_APPLY_CMDS / the swebench harness.
GIT_APPLY_CMDS = (
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
)
UPLOAD_DIR = "/evals_upload"

# Task-level sandbox defaults (memory_limit, cpu_limit, ...), set by install().
_SANDBOX_DEFAULTS: Dict[str, Any] = {}


def eval_instance(
    instance: dict,
    pred: str,
    timeout: int = 1800,
    log_dir: str = "outputs",
    dockerhub_username: str = "swebench",
    force_arch: str = "",
) -> dict:
    from evalscope.api.sandbox import SandboxEngine, build_sandbox_config, get_sandbox_service, merge_sandbox_config_dicts
    from evalscope.benchmarks.swe_bench.utils import resolve_swebench_arch
    from evalscope.utils.asyncio_runtime import AsyncioLoopRunner
    from ms_enclave.sandbox.model import ExecutionStatus
    from swebench.harness.constants import DOCKER_WORKDIR, KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION, LOG_TEST_OUTPUT
    from swebench.harness.grading import get_eval_report
    from swebench.harness.test_spec.test_spec import make_test_spec

    instance_id = instance[KEY_INSTANCE_ID]
    report: Dict[str, Any] = {}
    completed = False
    try:
        arch = resolve_swebench_arch(instance_id, force_arch)
        test_spec = make_test_spec(instance, namespace=dockerhub_username or "swebench", arch=arch)
        prediction = {KEY_PREDICTION: pred, KEY_INSTANCE_ID: instance_id, KEY_MODEL: ""}
        instance_log_dir = Path(log_dir) / "swebench_log" / instance_id
        instance_log_dir.mkdir(parents=True, exist_ok=True)
        (instance_log_dir / "patch.diff").write_text(pred or "")
        (instance_log_dir / "eval.sh").write_text(test_spec.eval_script)

        cfg = merge_sandbox_config_dicts(
            {"tools_config": {"shell_executor": {}}},
            _SANDBOX_DEFAULTS,
            {
                "image": test_spec.instance_image_key,
                "working_dir": DOCKER_WORKDIR,
                "platform": "linux/arm64" if arch == "arm64" else "linux/amd64",
            },
        )

        async def _run() -> tuple[bool, str, bool]:
            service = get_sandbox_service()
            handle = await service.create_sandbox(SandboxEngine.DOCKER, build_sandbox_config(SandboxEngine.DOCKER, cfg))

            async def sh(script: str, seconds: int = 600):
                return await handle.execute_tool(
                    "shell_executor", {"command": ["bash", "-c", script], "timeout": seconds}
                )

            try:
                with tempfile.TemporaryDirectory() as upload:
                    Path(upload, "patch.diff").write_text(pred or "")
                    Path(upload, "eval.sh").write_text(test_spec.eval_script)
                    await handle.put_dir(upload, UPLOAD_DIR)
                applied = False
                for cmd in GIT_APPLY_CMDS:
                    result = await sh(f"cd {DOCKER_WORKDIR} && {cmd} {UPLOAD_DIR}/patch.diff")
                    if result.status == ExecutionStatus.SUCCESS:
                        applied = True
                        break
                if not applied:
                    return False, f"{result.output or ''}{result.error or ''}", False
                result = await sh(f"/bin/bash {UPLOAD_DIR}/eval.sh 2>&1", seconds=int(timeout))
                return True, f"{result.output or ''}{result.error or ''}", result.status == ExecutionStatus.TIMEOUT
            finally:
                await handle.close()

        applied, output, timed_out = AsyncioLoopRunner.run(_run(), timeout=timeout + 1800)
        if not applied:
            raise RuntimeError(f"{instance_id} Patch Apply Failed:\n{output}")
        test_output_path = instance_log_dir / LOG_TEST_OUTPUT
        test_output_path.write_text(output + (f"\n\nTimeout error: {timeout} seconds exceeded." if timed_out else ""))
        if timed_out:
            raise TimeoutError(f"{instance_id} Test timed out after {timeout} seconds.")
        report = get_eval_report(
            test_spec=test_spec, prediction=prediction, test_log_path=test_output_path, include_tests_status=True
        )
        completed = True
    except Exception as exc:
        report["error"] = f"Error in evaluating model for {instance_id}: {exc}\n{traceback.format_exc()}"
    return {
        "completed": completed,
        "resolved": report.get(instance_id, {}).get("resolved", False),
        "report": report,
    }


def install(sandbox_defaults: Optional[Dict[str, Any]] = None) -> None:
    """Route the SWE-bench agentic adapter's scoring through eval_instance above."""
    import evalscope.benchmarks.swe_bench.utils as upstream

    _SANDBOX_DEFAULTS.clear()
    _SANDBOX_DEFAULTS.update(sandbox_defaults or {})
    upstream.eval_instance = eval_instance


__all__ = ["eval_instance", "install"]
