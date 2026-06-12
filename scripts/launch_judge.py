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
from swiss_ai_model_launch.launchers.topology import Topology

# overwrite LaunchArgs to include nodes and worker_port
class LaunchArgs(BaseLaunchArgs):
    nodes: int = 1
    worker_port: int = 8080

# ── Task-to-judge mapping ────────────────────────────────────────────

TASK_TO_JUDGE = {
    "alpaca_eval": "llama-3.3-70b",
    "multijail": "llama-3.3-70b",
    "aya_redteaming": "llama-3.3-70b",
    "arena_hard_v01": "qwen3.5-27b",
    "arena_hard_v2": "qwen3.5-27b",
}

# ── Judge presets ─────────────────────────────────────────────────────

MODEL_REGISTRY = Path("/capstor/store/cscs/swissai/infra01/hf_models/models")

JUDGE_PRESETS = {
    "qwen3.5-27b": {
        "served_model_name": "Qwen/Qwen3.5-27B",
        "framework": "vllm",
        "time": "04:00:00",
        "partition": "normal",
        "framework_args": (
            "--model Qwen/Qwen3.5-27B "
            "--host 0.0.0.0 "
            "--served-model-name Qwen/Qwen3.5-27B "
            "--tensor-parallel-size 4 --max-model-len 26000 "
        ),
    },
    "llama-3.3-70b": {
        "served_model_name": "meta-llama/Llama-3.3-70B-Instruct",
        "framework": "vllm",
        "time": "04:00:00",
        "framework_args": (
            "--model meta-llama/Llama-3.3-70B-Instruct "
            "--host 0.0.0.0 "
            "--served-model-name meta-llama/Llama-3.3-70B-Instruct "
            "--tensor-parallel-size 4 --max-model-len 35000"
        ),
    },
}

# ── Health check ──────────────────────────────────────────────────────

HEALTH_CHECK_URL = "https://api.swissai.svc.cscs.ch/v1/models"


def check_judge_health(model_name: str, api_key: str) -> bool:
    """Check if model_name is listed on the CSCS serving platform."""
    req = urllib.request.Request(
        HEALTH_CHECK_URL,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
            data = json.loads(resp.read().decode())
            model_ids = [m.get("id", "") for m in data.get("data", [])]
            return model_name in model_ids
    except Exception:
        return False


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

    topology = Topology(
        replicas=preset.get("replicas", 1),
        nodes_per_replica=preset.get("nodes_per_replica", 1),
    )

    return LaunchArgs(
        job_name=f"judge-{preset_name}-{username}",
        served_model_name=preset["served_model_name"],
        account=account,
        partition=partition,
        environment=environment,
        framework=preset["framework"],
        framework_args=preset["framework_args"],
        time=preset.get("time", "06:00:00"),
        topology=topology,
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


async def launch_judge(args: LaunchArgs, api_key: str, health_timeout: int,
                       health_interval: int) -> tuple[int, str]:
    """Launch a judge model and wait for it to become healthy."""
    username = getpass.getuser()
    account = args.account

    launcher = SlurmLauncher(
        system_name="alps",
        username=username,
        account=account,
        partition=args.partition,
    )

    _log(f"Submitting judge job: {args.job_name}")
    _log(f"  Model: {args.served_model_name}")
    _log(f"  Nodes: {args.total_nodes}, Time: {args.time}")

    job_id, served_name = await launcher.launch_with_args(args)
    _log(f"Job submitted: {job_id}")

    # Poll for health
    _log(f"Waiting for judge to become healthy (timeout: {health_timeout}s)...")
    start = time.time()
    while time.time() - start < health_timeout:
        status = await launcher.get_job_status(job_id)

        if status in (JobStatus.TIMEOUT, JobStatus.UNKNOWN):
            # Check if it's truly terminal (UNKNOWN could be transient)
            if status == JobStatus.TIMEOUT:
                raise RuntimeError(
                    f"Judge job {job_id} timed out (SLURM status: {status.value})"
                )

        if status == JobStatus.RUNNING:
            if check_judge_health(served_name, api_key):
                elapsed = int(time.time() - start)
                _log(f"Judge healthy after {elapsed}s")
                return job_id, served_name

        elapsed = int(time.time() - start)
        _log(f"  [{elapsed}s] SLURM status: {status.value}, waiting...")
        await asyncio.sleep(health_interval)

    # Timed out — cancel the orphaned job
    _log(f"Health check timed out after {health_timeout}s. Cancelling job {job_id}.")
    await launcher.cancel_job(job_id)
    raise TimeoutError(
        f"Judge model '{served_name}' not healthy after {health_timeout}s"
    )


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
    parser.add_argument("--replicas", type=int, help="Override number of serving replicas.")
    parser.add_argument("--nodes-per-replica", type=int, help="Override nodes per replica.")
    parser.add_argument("--time", help="Override SLURM time limit (HH:MM:SS).")
    parser.add_argument("--environment", help="Override environment TOML path.")
    parser.add_argument("--account", help="SLURM account (default: user's group).")
    parser.add_argument("--partition", help="SLURM partition (default: from preset, or 'normal').")
    parser.add_argument("--health-timeout", type=int, default=900,
                        help="Max seconds to wait for judge health (default: 900).")
    parser.add_argument("--health-interval", type=int, default=15,
                        help="Seconds between health checks (default: 15).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config without submitting.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Cancel mode
    if args.cancel_job_id is not None:
        asyncio.run(cancel_job(args.cancel_job_id))
        return

    # in sbatch, we can run export CSCS_SERVING_API="${CSCS_SERVING_API:-$(tr -d '\r\n' < ./scripts/cscs_serving_api_key.txt)} - also try to read from file

    api_key = os.environ.get("CSCS_SERVING_API")
    if not api_key:
        key_path = Path(__file__).parent.joinpath("cscs_serving_api_key.txt")
        if key_path.is_file():
            api_key = key_path.read_text().strip()
    if not api_key and not args.dry_run:
        _log("ERROR: CSCS_SERVING_API environment variable not set.")
        sys.exit(1)

    # Determine which presets to launch
    if args.preset:
        presets_to_launch = {args.preset}
    else:
        presets_to_launch = _detect_presets_from_tasks(args.detect_from_tasks)

    if not presets_to_launch:
        _log("No judge-dependent tasks found. Nothing to launch.")
        return

    _log(f"Judge presets to launch: {sorted(presets_to_launch)}")

    overrides = {
        "served_model_name": args.served_model_name,
        "framework": args.framework,
        "framework_args": args.framework_args,
        "replicas": args.replicas,
        "nodes_per_replica": args.nodes_per_replica,
        "time": args.time,
        "environment": args.environment,
        "account": args.account,
        "partition": args.partition,
    }

    for preset_name in sorted(presets_to_launch):
        launch_args = _build_launch_args(preset_name, overrides)

        if args.dry_run:
            _log(f"\n[DRY RUN] Would launch preset '{preset_name}':")
            _log(f"  job_name:         {launch_args.job_name}")
            _log(f"  served_model_name: {launch_args.served_model_name}")
            _log(f"  framework:        {launch_args.framework}")
            _log(f"  framework_args:   {launch_args.framework_args}")
            _log(f"  replicas:         {launch_args.topology.replicas}")
            _log(f"  nodes_per_replica: {launch_args.topology.nodes_per_replica}")
            _log(f"  total_nodes:      {launch_args.total_nodes}")
            _log(f"  time:             {launch_args.time}")
            _log(f"  environment:      {launch_args.environment}")
            _log(f"  account:          {launch_args.account}")
            _log(f"  partition:        {launch_args.partition}")
            continue

        try:
            job_id, model_name = asyncio.run(
                launch_judge(
                    launch_args, api_key,
                    args.health_timeout, args.health_interval,
                )
            )
        except (RuntimeError, TimeoutError) as e:
            _log(f"ERROR: {e}")
            sys.exit(1)

        # Machine-readable output on stdout
        print(f"JUDGE_JOB_ID={job_id}")
        print(f"JUDGE_MODEL_NAME={model_name}")


if __name__ == "__main__":
    main()
