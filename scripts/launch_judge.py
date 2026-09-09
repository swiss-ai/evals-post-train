#!/usr/bin/env python3
"""Launch and manage judge models on SLURM for LLM-as-judge evaluations.

Uses the swiss_ai_model_launch (SML) package to submit a vLLM serving job
on a SLURM node. The script polls until the model is healthy on the CSCS
serving platform, then prints machine-readable output for the caller.

Usage:
    # Launch a specific preset
    python3 scripts/launch_judge.py --preset qwen3.5-27b

    # Auto-detect from task list
    python3 scripts/launch_judge.py --detect-from-tasks configs/apertus/tasks_default.txt

    # Cancel a running judge
    python3 scripts/launch_judge.py --cancel-job-id 12345

    # Dry run
    python3 scripts/launch_judge.py --preset qwen3.5-27b --dry-run
"""

import argparse
import asyncio
import getpass
import grp
import json
import os
import ssl
import sys
import time
import urllib.request
from importlib.resources import files
from pathlib import Path

from swiss_ai_model_launch.launchers.launch_args import LaunchArgs as BaseLaunchArgs
from swiss_ai_model_launch.launchers.launcher import JobStatus
from swiss_ai_model_launch.launchers.slurm_launcher import SlurmLauncher

# overwrite LaunchArgs to include nodes and worker_port
class LaunchArgs(BaseLaunchArgs):
    nodes: int = 1
    worker_port: int = 8080

# ── Task-to-judge mapping ────────────────────────────────────────────

TASK_TO_JUDGE = {
    "alpaca_eval": "llama-3.3-70b",
    "multijail": "llama-3.3-70b",
    "aya_redteaming": "llama-3.3-70b",
    "hallulens": "cais-llama-harmbench",
    "arena_hard_v01": "qwen3.5-27b",
    "arena_hard_v2": "qwen3.5-27b",
    "harmbench": "cais-llama-harmbench",
    "realtoxicitypromptsllama": "llama-guard",
    "realtoxicitypromptsllama_small": "llama-guard",
    "polyglotoxicitypromptsllama_small": "llama-guard",
    "polyglotoxicitypromptsllama": "llama-guard",
}

# ── Judge presets ─────────────────────────────────────────────────────

MODEL_REGISTRY = Path("/capstor/store/cscs/swissai/infra01/hf_models/models")

JUDGE_PRESETS = {
    "qwen3.5-27b": {
        "served_model_name": "Qwen/Qwen3.5-27B",
        "framework": "vllm",
        "nodes": 1,
        "account": "infra01",
        "time": "04:00:00",
        "partition": "normal",
        "worker_port": 8080,
        "framework_args": (
            "--model Qwen/Qwen3.5-27B "
            "--host 0.0.0.0 --port 8080 "
            "--served-model-name Qwen/Qwen3.5-27B "
            "--tensor-parallel-size 4 --max-model-len 26000 "
        ),
    },
    "llama-3.3-70b": {
        "served_model_name": "meta-llama/Llama-3.3-70B-Instruct",
        "framework": "vllm",
        "nodes": 1,
        "time": "04:00:00",
        "account": "infra01",
        "worker_port": 8080,
        "framework_args": (
            "--model meta-llama/Llama-3.3-70B-Instruct "
            "--host 0.0.0.0 --port 8080 "
            "--served-model-name meta-llama/Llama-3.3-70B-Instruct "
            "--tensor-parallel-size 4 --max-model-len 35000"
        ),
    },
    "cais-llama-harmbench": {
        "served_model_name": "cais/HarmBench-Llama-2-13b-cls",
        "framework": "vllm",
        "nodes": 1,
        "time": "04:00:00",
        "account": "infra01",
        "partition": "normal",
        "worker_port": 8080,
        "framework_args": (
            "--model /capstor/store/cscs/swissai/infra01/hf_models/models/cais/HarmBench-Llama-2-13b-cls "
            "--host 0.0.0.0 --port 8080 "
            "--served-model-name cais/HarmBench-Llama-2-13b-cls "
            # This model's config.json caps max_position_embeddings at 2048 (unlike the
            # other judge presets); vLLM refuses to start above that, so keep it in bounds.
            "--tensor-parallel-size 4 --max-model-len 2048"
        ),
    },
    "llama-guard": {
        "served_model_name": "cais/Llama-Guard-13b",
        "framework": "vllm",
        "nodes": 1,
        "time": "04:00:00",
        "acount": "infra01",
        "partition": "normal",
        "framework_args": (
            f"--model {MODEL_REGISTRY / 'meta-llama/Llama-Guard-4-12B'} "
            "--host 0.0.0.0 "
            "--served-model-name meta-llama/Llama-Guard-4-12B "
            "--tensor-parallel-size 4 --max-model-len 35000"
        ),
    },
}

# ── Health check ──────────────────────────────────────────────────────

HEALTH_CHECK_URL = "https://api.swissai.svc.cscs.ch/v1/models"
TERMINAL_JOB_STATUSES = {
    JobStatus.COMPLETED,
    JobStatus.CANCELLED,
    JobStatus.FAILED,
    JobStatus.TIMEOUT,
}
UNKNOWN_STATUS_GRACE_POLLS = 3


def _launched_hosted_model_name(
    model_name: str,
    model_ids: list[str],
    username: str | None = None,
) -> str | None:
    """Find the launched model under the current CSCS user scope."""
    model_name = os.path.expandvars(model_name).strip().strip("/")
    username = (username or getpass.getuser()).strip().strip("/")
    is_scoped = len(model_name.split("/")) >= 3 or model_name.startswith(f"{username}/")
    candidates = [model_name] if is_scoped else [f"{username}/{model_name}"]
    return next((candidate for candidate in candidates if candidate in model_ids), None)


def check_judge_health(model_name: str, api_key: str) -> str | None:
    """Return the hosted model ID once the launched judge is listed."""
    req = urllib.request.Request(
        HEALTH_CHECK_URL,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
            data = json.loads(resp.read().decode())
            model_ids = [m.get("id", "") for m in data.get("data", [])]
            return _launched_hosted_model_name(model_name, model_ids)
    except Exception as exc:
        _log(f"  Health check failed: {type(exc).__name__}: {exc}")
        return None


# ── Core logic ────────────────────────────────────────────────────────


def _get_vllm_environment() -> str:
    """Resolve the SML built-in vllm.toml environment file path."""
    return str(files("swiss_ai_model_launch.assets.envs").joinpath("vllm.toml"))


def _build_launch_args(preset_name: str, overrides: dict) -> LaunchArgs:
    """Build LaunchArgs from a preset with optional overrides."""
    if preset_name not in JUDGE_PRESETS:
        raise ValueError(
            f"Unknown preset '{preset_name}'. Available: {list(JUDGE_PRESETS.keys())}"
        )

    preset = {**JUDGE_PRESETS[preset_name]}
    preset.update({k: v for k, v in overrides.items() if v is not None})

    username = getpass.getuser()
    account = preset.pop("account", None) or grp.getgrgid(os.getgid()).gr_name
    partition = preset.pop("partition", "normal")
    environment = preset.pop("environment", None) or _get_vllm_environment()

    return LaunchArgs(
        job_name=f"judge-{preset_name}-{username}",
        served_model_name=preset["served_model_name"],
        account=account,
        partition=partition,
        topology=Topology(
            replicas=1,
            nodes_per_replica=preset.get("nodes", 1),
        ),
        environment=environment,
        framework=preset["framework"],
        framework_args=preset["framework_args"],
        time=preset.get("time", "06:00:00"),
    )


def _detect_presets_from_tasks(tasks_file: str) -> set[str]:
    """Read a task list file and return the set of judge presets needed."""
    tasks_path = Path(tasks_file)
    if not tasks_path.is_file():
        # Might be a single task name (for 'single' mode)
        task_names = [tasks_file.strip()]
    else:
        task_names = [
            line.strip()
            for line in tasks_path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    presets = set()
    for task in task_names:
        for judge_task, preset in TASK_TO_JUDGE.items():
            if judge_task in task:
                presets.add(preset)
    return presets


async def launch_judge(
    args: LaunchArgs,
    api_key: str,
    health_timeout: int,
    health_interval: int,
    reservation: str | None = None,
) -> tuple[int, str]:
    """Launch a judge model and wait for it to become healthy."""
    username = getpass.getuser()
    account = args.account

    launcher = SlurmLauncher(
        system_name="alps",
        username=username,
        account=account,
        partition=args.partition,
        reservation=reservation,
    )

    _log(f"Submitting judge job: {args.job_name}")
    _log(f"  Model: {args.served_model_name}")
    _log(f"  Nodes: {args.total_nodes}, Time: {args.time}")
    if reservation:
        _log(f"  Reservation: {reservation}")

    job_id, served_name = await launcher.launch_with_args(args)
    _log(f"Job submitted: {job_id}")

    # Poll for health
    _log(f"Waiting for judge to become healthy (timeout: {health_timeout}s)...")
    start = time.time()
    seen_active = False
    consecutive_unknown = 0
    while time.time() - start < health_timeout:
        status = await launcher.get_job_status(job_id)

        if status in (JobStatus.PENDING, JobStatus.RUNNING):
            seen_active = True
            consecutive_unknown = 0
        elif status == JobStatus.UNKNOWN:
            # UNKNOWN is briefly possible before sacct sees a new job, but SML
            # also uses it for SLURM states it cannot parse. Allow a short
            # submission grace period, then surface the job logs.
            consecutive_unknown += 1

        terminal_status = status in TERMINAL_JOB_STATUSES
        unknown_after_active = status == JobStatus.UNKNOWN and seen_active
        unknown_after_grace = (
            status == JobStatus.UNKNOWN
            and consecutive_unknown >= UNKNOWN_STATUS_GRACE_POLLS
        )
        if terminal_status or unknown_after_active or unknown_after_grace:
            logs = await _collect_job_logs(
                launcher,
                job_id,
                replicas=args.topology.replicas,
            )
            raise RuntimeError(
                f"Judge job {job_id} terminated before becoming healthy "
                f"(SLURM status: {status.value}).{logs}"
            )

        if status == JobStatus.RUNNING:
            hosted_name = check_judge_health(served_name, api_key)
            if hosted_name:
                elapsed = int(time.time() - start)
                _log(f"Judge healthy after {elapsed}s as {hosted_name}")
                return job_id, hosted_name

        elapsed = int(time.time() - start)
        _log(f"  [{elapsed}s] SLURM status: {status.value}, waiting...")
        await asyncio.sleep(health_interval)

    # Timed out — cancel the orphaned job
    logs = await _collect_job_logs(
        launcher,
        job_id,
        replicas=args.topology.replicas,
    )
    _log(f"Health check timed out after {health_timeout}s. Cancelling job {job_id}.")
    await launcher.cancel_job(job_id)
    raise TimeoutError(
        f"Judge model '{served_name}' not healthy after {health_timeout}s.{logs}"
    )


def _tail(text: str, lines: int = 80) -> str:
    """Return a bounded log tail suitable for an error message."""
    if not text:
        return "(empty)"
    return "\n".join(text.splitlines()[-lines:])


async def _collect_job_logs(
    launcher: SlurmLauncher,
    job_id: int,
    replicas: int,
) -> str:
    """Collect master and per-replica logs after an early job termination."""
    sections = []
    master_out, master_err = await launcher.get_job_logs(job_id)
    sections.extend(
        [
            ("master log.out", master_out),
            ("master log.err", master_err),
        ]
    )
    for replica in range(replicas):
        replica_out = await launcher.read_job_file(job_id, f"replica_{replica}.out")
        replica_err = await launcher.read_job_file(job_id, f"replica_{replica}.err")
        sections.extend(
            [
                (f"replica_{replica}.out", replica_out or ""),
                (f"replica_{replica}.err", replica_err or ""),
            ]
        )

    rendered = "".join(
        f"\n\n--- {label} (tail) ---\n{_tail(content)}"
        for label, content in sections
        if content
    )
    return rendered or "\n\nNo SML job logs were available."


async def cancel_job(job_id: int) -> None:
    """Cancel a judge SLURM job."""
    username = getpass.getuser()
    account = grp.getgrgid(os.getgid()).gr_name

    launcher = SlurmLauncher(
        system_name="alps",
        username=username,
        account=account,
        partition="normal",
    )
    await launcher.cancel_job(job_id)
    _log(f"Cancelled job {job_id}")


def _log(msg: str) -> None:
    """Print to stderr so stdout stays clean for machine-readable output."""
    print(msg, file=sys.stderr, flush=True)


# ── CLI ───────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch judge models on SLURM for LLM-as-judge evaluations."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preset", choices=list(JUDGE_PRESETS.keys()),
                       help="Launch a specific judge preset.")
    group.add_argument("--detect-from-tasks",
                       help="Path to task list file. Auto-detect needed judges.")
    group.add_argument("--cancel-job-id", type=int,
                       help="Cancel a running judge job and exit.")

    parser.add_argument("--served-model-name", help="Override served model name.")
    parser.add_argument("--framework", help="Override serving framework.")
    parser.add_argument("--framework-args", help="Override framework arguments.")
    parser.add_argument("--nodes", type=int, help="Override number of SLURM nodes.")
    parser.add_argument("--time", help="Override SLURM time limit (HH:MM:SS).")
