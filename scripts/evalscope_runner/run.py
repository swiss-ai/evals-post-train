"""Run SWE-bench (Verified) and/or SWE-bench Pro through EvalScope (see run_evalscope_eval.sh).

Builds an EvalScope TaskConfig from the flags, routes SWE-bench's scoring through
the sandbox service (swe_bench_scoring.install()) and, with --sandbox k8s, every
sandbox to a Kubernetes pod (k8s_sandbox.install()), runs it, and writes
<work-dir>/summary.json: the scores plus everything needed to reproduce the run
(evalscope version, dataset revisions, SWE-bench_Pro-os commit, sandbox engine).

--gold skips the agent and the model: each sample's reference patch is the
prediction (install_gold()), so a run checks the scoring pipeline itself --
every sample should come out resolved.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

# Our benchmark names -> the EvalScope benchmark, its pinned Hugging Face dataset revision
# (--dataset-revision overrides) and the extra_params we always set.
BENCHMARKS: Dict[str, Dict[str, Any]] = {
    # SWE-bench Verified (500 tasks) as a bash agent loop, the same protocol as swe_bench_pro.
    # Images are the prebuilt swebench/sweb.eval.x86_64.* ones from DockerHub: EvalScope's
    # default builds them locally, which needs a Docker daemon (and arm64 builds differ).
    "swe_bench": {
        "evalscope": "swe_bench_verified_agentic",
        "revision": "c104f840cc67f8b6eec6f759ebc8b2693d585d4a",  # princeton-nlp/SWE-bench_Verified, 2025-02-18
        "extra_params": {"build_docker_images": False, "force_arch": "x86_64"},
    },
    # EvalScope implements SWE-bench Pro *v1* (731 tasks, jefzda/sweap-images, the
    # SWE-bench_Pro-os v1 run scripts it pins at ca10a60). Since 2026-09-22 the dataset's main
    # branch is V2 (642 Harbor tasks, other images and test setup): v1 scoring against it fails
    # even gold patches (its before_repo_set_cmd applies a /tests/test_patch.patch v1 images do
    # not have). 7ab5114 is the last pre-V2 commit (2026-02-23).
    "swe_bench_pro": {
        "evalscope": "swe_bench_pro",
        "revision": "7ab5114912baf22bb098818e604c02fe7ad2c11f",
        "extra_params": {},
    },
}

# EvalScope's own default for API models (DEFAULT_TEXT_GEN_SERVICE_CONFIG), which it only
# applies when generation_config is empty: --generation-config is merged over it instead.
DEFAULT_GENERATION_CONFIG: Dict[str, Any] = {"temperature": 0.0}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", required=True, help=f"Comma-separated, from: {', '.join(BENCHMARKS)}.")
    p.add_argument("--model", default=None, help="Served model name (not with --gold).")
    p.add_argument("--api-base-url", default=None, help="OpenAI-compatible /v1 root (not with --gold).")
    p.add_argument("--gold", action="store_true", help="Score the reference patches instead of running a model.")
    p.add_argument("--work-dir", required=True)
    p.add_argument("--sandbox", choices=("docker", "k8s"), default="docker")
    p.add_argument("--sandbox-config", default="{}", help="JSON merged into every sandbox's config.")
    p.add_argument("--limit", type=float, default=None)
    p.add_argument("--max-concurrency", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--strategy", default=None)
    p.add_argument("--command-timeout", type=float, default=None)
    p.add_argument("--generation-config", default="{}", help="JSON merged over {\"temperature\": 0}.")
    p.add_argument("--dataset-revision", action="append", default=[], help="<benchmark>=<revision>.")
    p.add_argument("--extra-param", action="append", default=[], help="key=value (JSON value or string).")
    p.add_argument("--api-timeout", type=float, default=None)
    args = p.parse_args(argv)
    if not args.gold and not (args.model and args.api_base_url):
        p.error("--model and --api-base-url are required (unless --gold)")
    return args


def parse_pairs(pairs: List[str], flag: str, as_json: bool) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"{flag} needs key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        try:
            out[key] = json.loads(value) if as_json else value
        except json.JSONDecodeError:
            out[key] = value
    return out


def selected_benchmarks(value: str) -> List[str]:
    names = [b.strip() for b in value.split(",") if b.strip()]
    unknown = [n for n in names if n not in BENCHMARKS]
    if unknown or not names:
        raise SystemExit(f"--benchmark must be one or more of {', '.join(BENCHMARKS)}; got {value!r}")
    return names


def resolved_revisions(names: List[str], overrides: List[str]) -> Dict[str, str]:
    given = parse_pairs(overrides, "--dataset-revision", as_json=False)
    stray = sorted(set(given) - set(names))
    if stray:
        raise SystemExit(f"--dataset-revision for a benchmark that is not selected: {', '.join(stray)}")
    return {n: given.get(n, BENCHMARKS[n]["revision"]) for n in names}


def build_task_config(args: argparse.Namespace, names: List[str], revisions: Dict[str, str]) -> Dict[str, Any]:
    extra_params = parse_pairs(args.extra_param, "--extra-param", as_json=True)
    dataset_args = {
        BENCHMARKS[n]["evalscope"]: {
            "hf_dataset_revision": revisions[n],
            "extra_params": {**BENCHMARKS[n]["extra_params"], **extra_params},
        }
        for n in names
    }

    agent: Dict[str, Any] = {}
    if args.strategy:
        agent["strategy"] = args.strategy
    if args.max_steps:
        agent["max_steps"] = args.max_steps
    if args.command_timeout:
        agent["command_timeout"] = args.command_timeout

    if args.gold:
        # install_gold() never calls the model; mock_llm keeps EvalScope from needing an endpoint.
        model_cfg: Dict[str, Any] = {"model": "gold", "eval_type": "mock_llm"}
    else:
        model_cfg = {
            "model": args.model,
            "api_url": args.api_base_url,
            "api_key": os.environ.get("EVALSCOPE_API_KEY") or "EMPTY",
            "eval_type": "openai_api",
        }
    cfg: Dict[str, Any] = {
        **model_cfg,
        "datasets": [BENCHMARKS[n]["evalscope"] for n in names],
        "dataset_args": dataset_args,
        "dataset_hub": "huggingface",
        "eval_batch_size": args.max_concurrency,
        "generation_config": {**DEFAULT_GENERATION_CONFIG, **json.loads(args.generation_config)},
        "work_dir": args.work_dir,
        "no_timestamp": True,
        # Every sandbox (agent loop and scorer) reads this; the engine stays "docker"
        # either way -- with --sandbox k8s, the Docker manager itself is swapped out.
        "sandbox": {"enabled": True, "engine": "docker", "default_config": json.loads(args.sandbox_config)},
    }
    if agent:
        # Only explicitly given fields: EvalScope keeps the benchmark's own
        # strategy/max_steps for anything left unset.
        cfg["agent_config"] = {"mode": "native", **agent}
    if args.limit is not None:
        cfg["limit"] = int(args.limit) if args.limit >= 1 else args.limit
    if args.api_timeout:
        cfg["timeout"] = args.api_timeout
    return cfg


def collect_reports(work_dir: str) -> List[Dict[str, Any]]:
    reports = []
    for path in sorted(glob.glob(os.path.join(work_dir, "reports", "**", "*.json"), recursive=True)):
        try:
            with open(path) as f:
                report = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        reports.append(
            {
                "path": os.path.relpath(path, work_dir),
                "dataset_name": report.get("dataset_name"),
                "model_name": report.get("model_name"),
                "score": report.get("score"),
                "metrics": report.get("metrics"),
            }
        )
    return reports


def swe_bench_pro_os_commit() -> Optional[str]:
    try:
        from evalscope.benchmarks.swe_bench_pro.utils import PINNED_COMMIT
    except Exception:
        return None
    return PINNED_COMMIT


def install_gold() -> None:
    """Make the SWE-bench agentic adapters predict each sample's reference patch, with no agent loop."""
    from evalscope.api.model import ModelOutput
    from evalscope.benchmarks.swe_bench.swe_bench_agentic_adapter import _SWEBenchAgenticAdapterBase
    from evalscope.benchmarks.swe_bench_pro.swe_bench_pro_agentic_adapter import SWEBenchProAgenticAdapter

    def _on_inference(self, model: Any, sample: Any) -> Any:
        return ModelOutput.from_content(model="gold", content=sample.metadata["patch"])

    for adapter in (_SWEBenchAgenticAdapterBase, SWEBenchProAgenticAdapter):
        adapter._on_inference = _on_inference


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    names = selected_benchmarks(args.benchmark)
    revisions = resolved_revisions(names, args.dataset_revision)
    os.makedirs(args.work_dir, exist_ok=True)

    from scripts.evalscope_runner import swe_bench_scoring

    swe_bench_scoring.install(json.loads(args.sandbox_config))
    if args.sandbox == "k8s":
        from scripts.evalscope_runner import k8s_sandbox

        k8s_sandbox.install()
    if args.gold:
        install_gold()

    import evalscope
    from evalscope.run import run_task

    cfg = build_task_config(args, names, revisions)
    printable = {k: v for k, v in cfg.items() if k != "api_key"}
    print("EvalScope task config:", json.dumps(printable, indent=2), flush=True)

    run_task(cfg)

    summary = {
        "evalscope_version": evalscope.__version__,
        "benchmarks": {n: BENCHMARKS[n]["evalscope"] for n in names},
        "model": "gold (reference patches)" if args.gold else args.model,
        "api_base_url": args.api_base_url,
        "sandbox": args.sandbox,
        "dataset_revisions": revisions,
        "limit": cfg.get("limit"),
        "reports": collect_reports(args.work_dir),
    }
    if "swe_bench_pro" in names:
        summary["swe_bench_pro_os_commit"] = swe_bench_pro_os_commit()
    with open(os.path.join(args.work_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    if not summary["reports"]:
        print(f"No EvalScope reports found under {args.work_dir}/reports", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
