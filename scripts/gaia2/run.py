"""Run Omnilingual GAIA2 through gaia2-runner on OpenSandbox (see run_gaia2_eval.sh).

facebook/omnilingual-gaia2 is GAIA2 (Meta's agent benchmark of simulated phone apps) translated
into ten languages, with four subsets each. gaia2-runner runs it: one container per scenario
holding the app simulator, the agent (OpenClaw) and an in-container LLM judge, which compares
the agent's actions and the apps' final state with the reference. Here each container is a Kata
microVM on an OpenSandbox server (scripts/gaia2/opensandbox_launcher.py), so the agent calls the
model from inside the sandbox: the model must be reachable from there (a public endpoint such as
the SwissAI gateway, not a server inside a Clariden job), and so must the judge.

The dataset is downloaded at a pinned revision (DATASET_REVISION) and materialized as one
scenario JSON per file, the layout gaia2-runner reads ([target].dataset_root); gaia2-runner's own
download would take the dataset's latest revision. --limit n runs a random n of each language and
subset (seed 42, the same on every run), through gaia2-runner's subset manifest.

One gaia2-runner run-config per language and subset, under
<work-dir>/runs/<language>/<subset>/ (config.toml, and gaia2-runner's results/ with one directory
per scenario and attempt). <work-dir>/summary.json, rewritten after each, has per cell, language,
subset and overall:

- success_rate: passed / attempts, gaia2-runner's own pass rate (avg@k for k attempts); an
  attempt without a verdict (an error) counts as not passed, as there;
- pass_at_k (with --pass-at k > 1): scenarios passed in any of the k attempts / scenarios;
- n_scenarios, n_attempts, n_passed, n_failed, n_errors and the errored attempts (errors).

An error is an attempt the judge gave no verdict on: the sandbox failed, the adapter was
unreachable, or the agent never ended its turn. The run exits non-zero when there are any
(those scores are lower bounds). --gold runs Meta's oracle image (the reference actions
replayed, no model): every scenario should pass, which checks the sandboxes, the apps and the
judge.

The scores use the judge given (--judge-model, default GLM-5.3 on the SwissAI gateway) with the
omnilingual judge prompts; Meta's published numbers use gpt-oss-120b (reasoning effort low), so
the two are not directly comparable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import signal
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
# facebookresearch/meta-agents-research-environments: requirements/gaia2-runtime.txt and the
# images (.github/workflows/gaia2-images.yml) are at this commit.
ARE_COMMIT = "a9fd2edccb4bc7e22fbf45c8482a5fef30a2a5a4"
IMAGE_REPO = "ghcr.io/swiss-ai"
AGENT_IMAGE = f"{IMAGE_REPO}/gaia2-oc:{ARE_COMMIT}"
ORACLE_IMAGE = f"{IMAGE_REPO}/gaia2-oracle:{ARE_COMMIT}"
DATASET = "facebook/omnilingual-gaia2"
# Its main branch on 2026-11-19 (last changed 2026-08-27).
DATASET_REVISION = "c1fe9241cfa687594cfb149a9781ed3072d11038"
LIMIT_SEED = 42
LANGUAGES = (
    "cmn_Hans",
    "deu_Latn",
    "fra_Latn",
    "hin_Deva",
    "ind_Latn",
    "ita_Latn",
    "jpn_Jpan",
    "por_Latn",
    "spa_Latn",
    "tur_Latn",
)
SUBSETS = ("execution", "search", "ambiguity", "adaptability")
GATEWAY = "https://api.swissai.svc.cscs.ch/v1"
DEFAULT_JUDGE_MODEL = "CSCS-Inference/zai-org/GLM-5.3"
# Gaia2's multilingual judge prompts: the stock (English) ones score correct non-English
# answers as misses.
JUDGE_PROMPT_VERSION = "omnilingual-gaia2"
OPENSANDBOX_KATA_DOMAIN = "opensandbox-kata.breithorn.svc.cscs.ch"  # = opensandbox_sandbox.DEFAULT_DOMAIN
# The keys reach gaia2-runner by name ([agent]/[judge].api_key_env), never in config.toml.
MODEL_KEY_ENV = "EVAL_MODEL_API_KEY"
JUDGE_KEY_ENV = "EVAL_JUDGE_API_KEY"


def _names(value: str, allowed: Iterable[str], what: str) -> List[str]:
    allowed = tuple(allowed)
    if value.strip().lower() == "all":
        return list(allowed)
    names = [v.strip() for v in value.split(",") if v.strip()]
    unknown = [n for n in names if n not in allowed]
    if unknown or not names:
        raise argparse.ArgumentTypeError(f"unknown {what} {', '.join(unknown) or value!r}; one or more of {', '.join(allowed)} or all")
    return list(dict.fromkeys(names))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    who = p.add_mutually_exclusive_group(required=True)
    who.add_argument("--model", help="Served model name at --api-base-url (OpenClaw needs native tool calls).")
    who.add_argument("--gold", action="store_true", help="The oracle image (reference actions, no model).")
    p.add_argument("--api-base-url", help="OpenAI-compatible /v1 root the sandbox can reach.")
    p.add_argument("--work-dir", required=True)
    p.add_argument("--language", default="all", type=lambda v: _names(v, LANGUAGES, "language"))
    p.add_argument("--subset", default="all", type=lambda v: _names(v, SUBSETS, "subset"))
    p.add_argument("--limit", type=int, help="A random n scenarios of each language and subset (seed 42).")
    p.add_argument("--pass-at", type=int, default=1, help="Attempts per scenario (Meta's headline is 3).")
    p.add_argument("--concurrency", type=int, default=4, help="Sandboxes alive at once.")
    p.add_argument("--thinking", choices=("off", "low", "medium", "high"), default="low", help="OpenClaw's thinking level.")
    p.add_argument("--timeout", type=int, default=1200, help="Seconds per scenario (at least its duration + 60).")
    p.add_argument("--idle-timeout", type=float, default=600.0, help="Seconds the daemon waits for agent activity.")
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--judge-base-url", default=GATEWAY)
    p.add_argument("--judge-extra-body", type=json.loads, help="JSON forwarded to the judge's requests.")
    p.add_argument("--agent-image", default=AGENT_IMAGE)
    p.add_argument("--oracle-image", default=ORACLE_IMAGE)
    p.add_argument(
        "--sandbox-network",
        choices=("allowlist", "public"),
        default="allowlist",
        help="allowlist: the sandbox reaches only the model's and the judge's hosts; public: everything.",
    )
    args = p.parse_args(argv)
    if args.model and not args.api_base_url:
        p.error("--model needs --api-base-url")
    if args.gold and args.api_base_url:
        p.error("--gold calls no model: drop --api-base-url")
    for name in ("limit", "pass_at", "concurrency", "timeout"):
        value = getattr(args, name)
        if value is not None and value < 1:
            p.error(f"--{name.replace('_', '-')} must be >= 1")
    return args


# --- config.toml ------------------------------------------------------------------------------


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)  # a TOML basic string: JSON's escapes are TOML's
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{json.dumps(str(k))} = {_toml_value(v)}" for k, v in value.items()) + " }"
    raise TypeError(f"no TOML for {value!r}")


def to_toml(tables: Dict[str, Dict[str, Any]]) -> str:
    lines: List[str] = []
    for name, table in tables.items():
        lines.append(f"[{name}]")
        lines += [f"{key} = {_toml_value(value)}" for key, value in table.items() if value is not None]
        lines.append("")
    return "\n".join(lines)


def runner_config(
    args: argparse.Namespace, dataset_root: Path, subset: str, output_dir: Path, subset_manifest: Optional[Path] = None
) -> Dict[str, Dict[str, Any]]:
    """gaia2-runner's run-config TOML (gaia2_runner/config.py) for one language (dataset_root,
    see materialize) and subset."""
    if args.gold:
        agent: Dict[str, Any] = {"image": args.oracle_image, "runtime": "opensandbox"}
    else:
        agent = {
            "image": args.agent_image,
            "runtime": "opensandbox",
            "provider": "openai-compat",
            "model": args.model,
            "base_url": args.api_base_url,
            "api_key_env": MODEL_KEY_ENV,
            "thinking": args.thinking,
        }
    return {
        "target": {
            "dataset_root": str(dataset_root),
            "splits": [subset],
            "subset_manifest": str(subset_manifest) if subset_manifest else None,
        },
        "agent": agent,
        "judge": {
            "provider": "openai-compat",
            "model": args.judge_model,
            "base_url": args.judge_base_url,
            "api_key_env": JUDGE_KEY_ENV,
            "prompt_version": JUDGE_PROMPT_VERSION,
            "extra_body": args.judge_extra_body,
        },
        "run": {
            "timeout": args.timeout,
            "health_timeout": 300,
            "concurrency": args.concurrency,
            "idle_timeout": args.idle_timeout,
            "pass_at": args.pass_at,
            "output_dir": str(output_dir),
            "log_level": "INFO",
        },
    }


def dataset_cache() -> Path:
    root = os.environ.get("GAIA2_HF_CACHE") or Path.home() / ".cache" / "gaia2" / "hf_datasets"
    return Path(root) / f"omnilingual-gaia2@{DATASET_REVISION}"


def materialize(language: str, subset: str, cache: Path) -> Path:
    """<cache>/<language>/<subset>/<scenario id>.json from the dataset at DATASET_REVISION, as
    gaia2-runner's own download lays it out; returns <cache>/<language> (the dataset root)."""
    split_dir = cache / language / subset
    if (split_dir / ".complete").exists():
        return split_dir.parent
    from datasets import load_dataset

    rows = load_dataset(DATASET, f"{language}_{subset}", split="test", revision=DATASET_REVISION)
    partial = split_dir.with_name(f".{subset}.partial-{uuid.uuid4().hex[:8]}")
    partial.mkdir(parents=True)
    for row in rows:
        (partial / f"{row['scenario_id']}.json").write_text(row["scenario"])
    (partial / ".complete").write_text(f"{len(rows)}\n")
    try:
        partial.rename(split_dir)
    except OSError:  # another run materialized it first
        if not (split_dir / ".complete").exists():
            raise
    return split_dir.parent


def limit_manifest(dataset_root: Path, subset: str, limit: int, path: Path) -> Path:
    """gaia2-runner's subset manifest with a random `limit` of the subset's scenarios (seeded)."""
    ids = sorted(p.stem for p in (dataset_root / subset).glob("*.json"))
    chosen = sorted(random.Random(LIMIT_SEED).sample(ids, min(limit, len(ids))))
    path.write_text(json.dumps({"splits": {subset: chosen}}, indent=2) + "\n")
    return path


def allowed_hosts(args: argparse.Namespace) -> List[str]:
    urls = [args.judge_base_url] + ([] if args.gold else [args.api_base_url])
    return list(dict.fromkeys(urlparse(u).hostname for u in urls if urlparse(u).hostname))


# --- summary ----------------------------------------------------------------------------------


def _attempt(path: Path, results_dir: Path) -> int:
    for part in path.relative_to(results_dir).parts:
        if part.startswith("run_") and part[4:].isdigit():
            return int(part[4:])
    return 1


Outcomes = Dict[str, Dict[int, Optional[bool]]]


def cell_outcomes(results_dir: Path) -> "tuple[Outcomes, List[Dict[str, Any]]]":
    """Per scenario and attempt, from gaia2-runner's result.json files (<results>/[run_<n>/]
    <subset>/<scenario id>/result.json): success true is passed, false failed, anything else
    (no verdict) an error, listed with its message."""
    outcomes: Outcomes = {}
    errors: List[Dict[str, Any]] = []
    for path in sorted(results_dir.glob("**/result.json")):
        try:
            result = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        scenario = str(result.get("scenario_id") or path.parent.name)
        attempt = _attempt(path, results_dir)
        success = result.get("success")
        outcomes.setdefault(scenario, {})[attempt] = success if isinstance(success, bool) else None
        if not isinstance(success, bool):
            errors.append({"scenario": scenario, "attempt": attempt, "error": str(result.get("error") or "no verdict")[:500]})
    return outcomes, errors


def aggregate(outcomes: Outcomes, errors: List[Dict[str, Any]]) -> Dict[str, Any]:
    attempts = [v for per in outcomes.values() for v in per.values()]
    n_passed = sum(1 for v in attempts if v is True)
    n_failed = sum(1 for v in attempts if v is False)
    summary: Dict[str, Any] = {
        "success_rate": n_passed / len(attempts) if attempts else None,
        "n_scenarios": len(outcomes),
        "n_attempts": len(attempts),
        "n_passed": n_passed,
        "n_failed": n_failed,
        "n_errors": len(attempts) - n_passed - n_failed,
    }
    k = max((len(per) for per in outcomes.values()), default=0)
    if k > 1:
        any_pass = sum(1 for per in outcomes.values() if any(v is True for v in per.values()))
        summary["pass_at_k"] = {str(k): any_pass / len(outcomes)}
    summary["errors"] = errors
    return summary


def summarize(work_dir: Path, languages: List[str], subsets: List[str]) -> Dict[str, Any]:
    """Per cell (language/subset), per language, per subset and overall, pooled over the
    attempts: a language's success rate is over all its scenarios."""
    found = {}
    for language in languages:
        for subset in subsets:
            results = work_dir / "runs" / language / subset / "results"
            if results.is_dir():
                found[(language, subset)] = cell_outcomes(results)

    def pooled(keys: List[Any]) -> Dict[str, Any]:
        outcomes: Outcomes = {}
        errors: List[Dict[str, Any]] = []
        for key in keys:
            cell, cell_errors = found[key]
            outcomes.update({f"{key[0]}/{key[1]}/{scenario}": per for scenario, per in cell.items()})
            errors += [dict(e, cell=f"{key[0]}/{key[1]}") for e in cell_errors]
        return aggregate(outcomes, errors)

    def group(index: int, names: List[str]) -> Dict[str, Any]:
        keys = {name: [key for key in found if key[index] == name] for name in names}
        return {name: pooled(ks) for name, ks in keys.items() if ks}

    return {
        "cells": {f"{l}/{s}": aggregate(*found[(l, s)]) for (l, s) in found},
        "languages": group(0, languages),
        "subsets": group(1, subsets),
        "overall": pooled(list(found)),
    }


# --- running ----------------------------------------------------------------------------------


def runner_env(args: argparse.Namespace) -> Dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), env.get("PYTHONPATH")]))
    env.setdefault("OPENSANDBOX_DOMAIN", OPENSANDBOX_KATA_DOMAIN)
    # Unique to this run, so the cleanup after it touches no other run's sandboxes; a label the
    # caller set (one per run of theirs) is kept.
    env.setdefault("OPENSANDBOX_RUN_LABEL", f"gaia2-{uuid.uuid4().hex[:12]}")
    model_key = os.environ.get(MODEL_KEY_ENV, "")
    env[MODEL_KEY_ENV] = model_key
    env[JUDGE_KEY_ENV] = os.environ.get(JUDGE_KEY_ENV) or model_key
    if args.sandbox_network == "allowlist":
        env["GAIA2_ALLOWED_HOSTS"] = ",".join(allowed_hosts(args))
    else:
        env.pop("GAIA2_ALLOWED_HOSTS", None)
    # gaia2-runner reads its own .env next to its sources; nothing of ours is there.
    return env


def delete_leftover_sandboxes(env: Dict[str, str]) -> None:
    """A run that is stopped can leave sandboxes behind. Never fails the run."""
    label = env["OPENSANDBOX_RUN_LABEL"]
    try:
        from scripts.common.opensandbox_sandbox import delete_run_sandboxes

        deleted = asyncio.run(
            asyncio.wait_for(delete_run_sandboxes(label, env["OPENSANDBOX_DOMAIN"], env["OPENSANDBOX_API_KEY"]), 120)
        )
    except Exception as exc:
        print(f"warning: could not delete sandboxes with sandbox-run={label}: {exc!r}", file=sys.stderr)
        return
    if deleted:
        print(f"Deleted {deleted} leftover sandboxes (sandbox-run={label})", flush=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def gaia2_version() -> Optional[str]:
    try:
        from importlib.metadata import version

        return version("gaia2-runner")
    except Exception:
        return None


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if not os.environ.get("OPENSANDBOX_API_KEY"):
        print("gaia2 runs every scenario on OpenSandbox: OPENSANDBOX_API_KEY is not set", file=sys.stderr)
        return 2
    work_dir = Path(args.work_dir).resolve()
    env = runner_env(args)
    record: Dict[str, Any] = {
        "gaia2_commit": ARE_COMMIT,
        "gaia2_runner_version": gaia2_version(),
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "model": "oracle (reference actions)" if args.gold else args.model,
        "agent": "oracle" if args.gold else "openclaw",
        "image": args.oracle_image if args.gold else args.agent_image,
        "api_base_url": None if args.gold else args.api_base_url,
        "thinking": None if args.gold else args.thinking,
        "judge_model": args.judge_model,
        "judge_base_url": args.judge_base_url,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "languages": args.language,
        "subsets": args.subset,
        "limit": args.limit,
        "pass_at": args.pass_at,
        "sandbox_endpoint": env["OPENSANDBOX_DOMAIN"],
        "sandbox_network": args.sandbox_network,
        "started_at": _now(),
    }
    stopped = False
    proc: Optional[subprocess.Popen] = None
    failed_cells: Dict[str, int] = {}

    def stop(signum: int, frame: Any) -> None:
        nonlocal stopped
        stopped = True
        if proc is not None and proc.poll() is None:
            proc.terminate()

    signal.signal(signal.SIGTERM, stop)

    def write_summary() -> Dict[str, Any]:
        summary = {**record, "finished_at": _now(), **summarize(work_dir, args.language, args.subset)}
        if failed_cells:
            summary["runner_exit_codes"] = dict(failed_cells)
        if stopped:
            summary["interrupted"] = True
        (work_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary

    for language in args.language:
        for subset in args.subset:
            if stopped:
                break
            cell_dir = work_dir / "runs" / language / subset
            cell_dir.mkdir(parents=True, exist_ok=True)
            dataset_root = materialize(language, subset, dataset_cache())
            manifest = limit_manifest(dataset_root, subset, args.limit, cell_dir / "subset.json") if args.limit else None
            config = cell_dir / "config.toml"
            config.write_text(to_toml(runner_config(args, dataset_root, subset, cell_dir / "results", manifest)))
            cmd = [sys.executable, "-m", "scripts.gaia2.opensandbox_launcher", "run-config", "--config", str(config)]
            print(f"=== {language}/{subset}: {' '.join(cmd)}", flush=True)
            proc = subprocess.Popen(cmd, env=env, cwd=REPO_ROOT)
            if stopped:
                proc.terminate()
            code = proc.wait()
            if code != 0 and not stopped:
                failed_cells[f"{language}/{subset}"] = code
            # After each cell, so a stopped run still has the finished cells' scores.
            write_summary()
    delete_leftover_sandboxes(env)
    summary = write_summary()
    overall = {k: v for k, v in summary["overall"].items() if k != "errors"}
    print(json.dumps({"overall": overall}, indent=2))
    if stopped:
        print(f"Stopped: {work_dir}/summary.json has the scenarios that finished.", file=sys.stderr)
        return 128 + signal.SIGTERM
    if summary["overall"]["n_errors"] or failed_cells or not summary["overall"]["n_attempts"]:
        print(
            f"Some attempts got no verdict, a cell's runner failed, or nothing ran (see errors and "
            f"runner_exit_codes in {work_dir}/summary.json): those scores are incomplete.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
