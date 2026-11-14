import ast
import json
import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

# Launcher inputs a caller's shell may export; the dry runs start without them.
_LAUNCH_INPUTS = {
    "TOKENIZER", "MAX_LENGTH", "BOS", "CHAT_FORMAT", "PREFILL_MODE", "EVAL_PREBUILT_ENV",
    "LM_EVAL_BACKEND", "LM_EVAL_HARNESS_BRANCH", "LM_EVAL_HARNESS_REPO", "EVAL_ENV_CACHE_ROOT", "API_CHAT_ENDPOINT", "ENABLE_THINKING",
}
ENV = {key: value for key, value in os.environ.items() if key not in _LAUNCH_INPUTS}


def _dry_run(*arguments: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        completed = subprocess.run(
            [
                "bash",
                "scripts/lm_eval_harness/launch_evaluations.sh",
                *arguments,
                "--logs-root",
                tmp,
                "--debug",
            ],
            cwd=REPO_ROOT,
            env=ENV | {"WANDB_PROJECT": "launcher-test"},
            check=True,
            text=True,
            capture_output=True,
        )
    return completed.stdout + completed.stderr


def _prebuilt_dry_run(*arguments: str, **extra_env: str) -> tuple[subprocess.CompletedProcess, str]:
    """A dry run with EVAL_PREBUILT_ENV=1; returns the run and its manifest."""
    with tempfile.TemporaryDirectory() as tmp:
        completed = subprocess.run(
            [
                "bash",
                "scripts/lm_eval_harness/launch_evaluations.sh",
                *arguments,
                "--logs-root",
                tmp,
                "--debug",
            ],
            cwd=REPO_ROOT,
            env=ENV
            | {"WANDB_PROJECT": "launcher-test", "EVAL_PREBUILT_ENV": "1"}
            | extra_env,
            text=True,
            capture_output=True,
        )
        manifests = list(Path(tmp).rglob("environment.sh"))
        manifest = manifests[0].read_text() if manifests else ""
    return completed, manifest


def _launch(*arguments: str, **env: str) -> subprocess.CompletedProcess:
    """A dry run that may fail; the caller passes --logs-root."""
    return subprocess.run(
        ["bash", "scripts/lm_eval_harness/launch_evaluations.sh", *arguments, "--debug"],
        cwd=REPO_ROOT,
        env=ENV | {"WANDB_PROJECT": "launcher-test"} | env,
        text=True,
        capture_output=True,
    )


def _harness_pin() -> str:
    pins = (REPO_ROOT / "requirements/lm-eval-harness.txt").read_text().split()
    return pins[pins.index("swiss-ai/lm-evaluation-harness") + 1]


def _run_config(logs_root: str) -> dict:
    configs = list(Path(logs_root).rglob("run_config.json"))
    assert len(configs) == 1, configs
    return json.loads(configs[0].read_text())


SBATCH = (REPO_ROOT / "scripts/lm_eval_harness/run_lm_eval.sbatch").read_text()


def _function(name: str) -> str:
    start = SBATCH.index(f"{name}() {{")
    return SBATCH[start: SBATCH.index("\n}\n", start) + 3]


def _chat_format(**env: str) -> subprocess.CompletedProcess:
    """run_lm_eval.sbatch's CHAT_FORMAT checks for an apertus2 openai run, run on their own."""
    start = SBATCH.index("CHAT_FORMAT=${CHAT_FORMAT:-template}")
    script = "set -e\n" + _function("die") + SBATCH[start: SBATCH.index("esac\n", start) + 5]
    base = {
        "PATH": os.environ["PATH"], "CHAT_FORMAT": "apertus2", "LM_EVAL_BACKEND": "openai",
        "APPLY_CHAT_TEMPLATE": "true", "MAX_LENGTH": "65536", "ENABLE_THINKING_EXPLICIT": "0",
        "AUTODETECT_THINK_TOKENS": "False", "TOKENIZER": "/ckpt",
    }
    return subprocess.run(["bash", "-c", script], env=base | env, capture_output=True, text=True)


def _gen_kwargs(**env: str) -> str:
    """run_lm_eval.sbatch's --gen_kwargs for this environment (its GEN_KWARGS block)."""
    block = SBATCH[SBATCH.index('GEN_KWARGS=""'): SBATCH.index('MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-""}')]
    script = "set -e\n" + block + '\nprintf "%s" "$GEN_KWARGS" >&2\n'
    base = {"PATH": os.environ["PATH"], "ENABLE_THINKING": "False", "REASONING_EFFORT": "", "MAX_NEW_TOKENS": "32768"}
    done = subprocess.run(["bash", "-c", script], env=base | env, capture_output=True, text=True, check=True)
    return done.stderr


def _openai_model(**env: str) -> subprocess.CompletedProcess:
    """run_lm_eval.sbatch's openai backend branch and context mapping, run on their own; the
    last stdout line is the harness model and its --model_args."""
    branch = SBATCH[SBATCH.index('elif [[ $LM_EVAL_BACKEND == "openai" ]]; then'):
                    SBATCH.index('else\n    die "Unsupported LM_EVAL_BACKEND')]
    mapping = SBATCH[SBATCH.index('if [[ -n "$BACKEND_MAX_LENGTH" ]]; then'):
                     SBATCH.index('echo "Using lm-evaluation-harness:')]
    script = ("set -e\n" + _function("die") + _function("_norm_bool") + "if false; then :\n" + branch + "fi\n"
              + mapping + 'printf "\\n%s %s%s" "$API_LM_EVAL_MODEL" "$COMMON_MODEL_ARGS" "$CHAT_TEMPLATE_ARGS"\n')
    base = {
        "PATH": os.environ["PATH"], "LM_EVAL_BACKEND": "openai", "CHAT_FORMAT": "template",
        "APPLY_CHAT_TEMPLATE": "true", "API_BASE_URL": "http://host:8000", "API_MODEL_NAME": "m",
        "API_NUM_CONCURRENT": "8", "API_MAX_RETRIES": "3", "API_REQUEST_TIMEOUT": "3600",
        "TOKENIZER": "/ckpt", "BS": "1", "BOS": "false", "THINK_ARGS": "", "ENABLE_THINKING": "False",
        "ENABLE_THINKING_EXPLICIT": "0", "THINKING_DEFAULT_MODEL_ARG": ",max_gen_toks=32768",
        "MAX_LENGTH": "65536", "BACKEND_MAX_LENGTH": "65536", "PREFILL_MODE": "after_think",
        "REASONING_EFFORT": "medium", "CHAT_TEMPLATE_ARGS": " 'chat_template_args={...}'",
    }
    return subprocess.run(["bash", "-c", script], env=base | env, capture_output=True, text=True)


class LaunchFlowTests(unittest.TestCase):
    def test_omitted_mode_defaults_to_posttrain(self) -> None:
        output = _dry_run("--model", "test/model")
        self.assertIn("Mode:   posttrain", output)

    def test_openai_backend_selects_cpu_only_sbatch_wrapper(self) -> None:
        output = _dry_run(
            "single",
            "--task",
            "hellaswag",
            "--backend",
            "openai",
            "--api-base-url",
            "http://localhost:8000",
            "--api-model-name",
            "test-model",
        )
        self.assertIn("scripts/lm_eval_harness/run_lm_eval_api.sbatch test-model test-model", output)
        self.assertNotIn("scripts/lm_eval_harness/run_lm_eval.sbatch test-model test-model", output)
        wrapper = (REPO_ROOT / "scripts/lm_eval_harness/run_lm_eval_api.sbatch").read_text()
        self.assertNotIn("#SBATCH --gres", wrapper)
        self.assertNotIn("#SBATCH --exclusive", wrapper)

    def test_trust_remote_code_flag_is_accepted(self) -> None:
        output = _dry_run(
            "single",
            "--task",
            "hellaswag",
            "--backend",
            "openai",
            "--api-base-url",
            "http://localhost:8000",
            "--api-model-name",
            "test-model",
            "--trust-remote-code",
        )
        self.assertNotIn("Unknown option", output)

    def test_trust_remote_code_reaches_the_model_args(self) -> None:
        # Runs run_lm_eval.sbatch's own block, so the test follows the script.
        sbatch = (REPO_ROOT / "scripts/lm_eval_harness/run_lm_eval.sbatch").read_text()
        start = sbatch.index("# Opt-in, for models whose tokenizer is custom code")
        block = sbatch[start: sbatch.index("HARNESS_EVAL_DIR=", start)]

        def run(backend: str, flag: str) -> subprocess.CompletedProcess:
            script = (
                'die() { echo "DIE: $*"; exit 1; }\n'
                f'LM_EVAL_BACKEND={backend}; TRUST_REMOTE_CODE={flag}; COMMON_MODEL_ARGS="model=m"\n'
                f'{block}\necho "ARGS=$COMMON_MODEL_ARGS"\n'
            )
            return subprocess.run(["bash", "-c", script], text=True, capture_output=True)

        for backend in ("hf", "vllm", "openai"):
            self.assertIn("ARGS=model=m,trust_remote_code=True", run(backend, "true").stdout)
            self.assertIn("ARGS=model=m\n", run(backend, "false").stdout)
        refused = run("sglang", "true")
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("only supported with the hf, vllm and openai backends", refused.stdout)

    def test_bos_is_resolved_for_each_command_by_whether_it_applies_the_chat_template(self) -> None:
        sbatch = (REPO_ROOT / "scripts/lm_eval_harness/run_lm_eval.sbatch").read_text()
        block = sbatch[sbatch.index("# bos per pass begin"): sbatch.index("# bos per pass end")]

        def pass_bos(requested: str, template: str, config: dict[str, object]) -> str:
            with tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / "tokenizer_config.json").write_text(json.dumps(config))
                script = (
                    f"set -euo pipefail\nBOS_REQUESTED={requested}; TOKENIZER={tmp}\n{block}\n"
                    f'resolve_pass_bos {template}\necho "[$PASS_BOS]"\n'
                )
                out = subprocess.run(["bash", "-c", script], cwd=REPO_ROOT, text=True, capture_output=True)
            return out.stdout.strip().splitlines()[-1]

        not_added = {"bos_token": "<s>", "add_bos_token": False}  # e.g. the Apertus 2 instruct tokenizer
        added = {"bos_token": "<s>", "add_bos_token": True}  # a tokenizer that adds it itself
        # BOS=auto: the config rule applies to a command without the chat template only
        self.assertEqual(pass_bos("auto", "false", not_added), "[true]")
        self.assertEqual(pass_bos("auto", "true", not_added), "[false]")
        self.assertEqual(pass_bos("auto", "false", added), "[false]")
        self.assertEqual(pass_bos("auto", "true", added), "[false]")
        # --bos / --no-bos decide every command
        for template in ("true", "false"):
            self.assertEqual(pass_bos("true", template, added), "[true]")
            self.assertEqual(pass_bos("false", template, not_added), "[false]")

    def test_a_failing_bos_helper_is_reported_and_does_not_kill_the_job(self) -> None:
        sbatch = (REPO_ROOT / "scripts/lm_eval_harness/run_lm_eval.sbatch").read_text()
        block = sbatch[sbatch.index("# bos per pass begin"): sbatch.index("# bos per pass end")]
        with tempfile.TemporaryDirectory() as tmp:
            stub = Path(tmp) / "python3"  # the node's python3 too old for the helper
            stub.write_text("#!/bin/sh\necho 'SyntaxError: future feature annotations is not defined' >&2\nexit 1\n")
            stub.chmod(0o755)
            script = (
                f"set -euo pipefail\nPATH={tmp}:$PATH; BOS_REQUESTED=auto; TOKENIZER=t\n{block}\n"
                'resolve_pass_bos false\necho "[$PASS_BOS]"\n'
            )
            out = subprocess.run(["bash", "-c", script], text=True, capture_output=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("[false]", out.stdout)
        self.assertIn("BOS=auto could not be resolved", out.stderr)
        self.assertIn("SyntaxError", out.stderr)

    def test_bos_by_client_applies_to_commands_with_a_chat_template(self) -> None:
        sbatch = (REPO_ROOT / "scripts/lm_eval_harness/run_lm_eval.sbatch").read_text()
        block = sbatch[sbatch.index("# bos per pass begin"): sbatch.index("# bos per pass end")]

        def by_client(requested: str, template: str) -> tuple[str, str]:
            script = (
                f"set -euo pipefail\nBOS_BY_CLIENT={requested}\n{block}\n"
                f'resolve_pass_bos_by_client {template}\necho "$PASS_BOS_BY_CLIENT"\n'
            )
            out = subprocess.run(["bash", "-c", script], text=True, capture_output=True).stdout.strip().splitlines()
            return out[-1], "\n".join(out[:-1])

        self.assertEqual(by_client("true", "true"), ("true", ""))
        self.assertEqual(by_client("false", "true"), ("false", ""))
        self.assertEqual(by_client("false", "false"), ("false", ""))
        ignored, note = by_client("true", "false")
        self.assertEqual(ignored, "false")
        self.assertIn("ignored for this one (no chat template)", note)

    def test_bos_probe_runs_on_the_completions_endpoint_with_its_flags(self) -> None:
        sbatch = (REPO_ROOT / "scripts/lm_eval_harness/run_lm_eval.sbatch").read_text()
        block = sbatch[sbatch.index("# bos per pass begin"): sbatch.index("# bos per pass end")]

        def probe_line(
            template: str, backend: str = "openai", model: str = "local-completions", check: str = "", by_client: str = "false"
        ) -> str:
            script = (
                f"set -euo pipefail\nLM_EVAL_BACKEND={backend}; API_LM_EVAL_MODEL={model}; BOS_CHECK={check}; PASS_BOS=true; BOS_REQUESTED=auto\n"
                f"PASS_BOS_BY_CLIENT={by_client}\n"
                "API_BASE_URL=http://x/v1/completions; API_MODEL_NAME=m; TOKENIZER=t; TIMED_CMD=start\n"
                f"{block}\nappend_bos_probe {template}\necho \"$TIMED_CMD\"\n"
            )
            out = subprocess.run(["bash", "-c", script], text=True, capture_output=True).stdout
            return out if "bos_probe" in out else ""

        raw = probe_line("false")
        self.assertIn("--bos 'true'", raw)
        self.assertNotIn("--chat-template", raw)
        self.assertNotIn("--bos-by-client", raw)
        chat = probe_line("true", by_client="true")
        self.assertIn("--chat-template", chat)
        self.assertIn("--bos-by-client", chat)
        self.assertEqual(probe_line("false", check="false"), "")
        self.assertEqual(probe_line("false", backend="vllm"), "")
        self.assertEqual(probe_line("false", model="local-chat-completions"), "")


    def test_debug_is_forwarded_to_deferred_judge_launch(self) -> None:
        orchestrator = (REPO_ROOT / "scripts/lm_eval_harness/evaluation_orchestrator.sh").read_text()
        self.assertIn("launch_args+=(--dry-run)", orchestrator)
        self.assertIn('_eval_launch_judge "$state_dir/missing_0.txt"', orchestrator)

    def test_prebuilt_environment_skips_preparation_and_pins_the_harness(self) -> None:
        pins = dict(
            line.split()
            for line in (REPO_ROOT / "requirements/lm-eval-harness.txt").read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        for policy in ("resume", "fail-fast"):
            with self.subTest(policy=policy):
                completed, manifest = _prebuilt_dry_run(
                    "single", "--task", "hellaswag", "--model", "test/model",
                    "--failure-policy", policy,
                )
                output = completed.stdout + completed.stderr
                self.assertEqual(completed.returncode, 0, output)
                self.assertNotIn("prepare_eval_env.sbatch", output)
                self.assertIn("(no preparation job)", output)
                self.assertIn("EVAL_ENV_KIND=prebuilt", manifest)
                self.assertIn(
                    "EVAL_RESOLVED_HARNESS_COMMIT="
                    + pins["swiss-ai/lm-evaluation-harness"],
                    manifest,
                )
                self.assertIn(
                    "EVAL_HARNESS_OVERLAY=/opt/lm-eval-harness/swiss-ai/lm-evaluation-harness",
                    manifest,
                )

    def test_prebuilt_apertus2_run_names_the_wheel_of_the_pinned_apertus_common(self) -> None:
        pin = (REPO_ROOT / "requirements/apertus2.txt").read_text().rsplit("@", 1)[1].strip()
        api = ["single", "--task", "gsm8k", "--backend", "openai", "--api-base-url", "http://localhost:8000",
               "--api-model-name", "test-model"]
        with tempfile.TemporaryDirectory() as cache:
            completed, manifest = _prebuilt_dry_run(
                *api, "--chat-format", "apertus2", "--tokenizer", "/ckpt", "--max-length", "65536",
                EVAL_ENV_CACHE_ROOT=cache,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            # The image has no apertus-common: the launcher names the wheel it builds for the pin
            # (a dry run builds nothing), and the job installs that offline.
            self.assertIn(f"EVAL_APERTUS_COMMON_WHEELS={cache}/apertus-common/{pin}", manifest)
            self.assertFalse(list(Path(cache).rglob("*.whl")))
            # Other runs don't need it.
            completed, manifest = _prebuilt_dry_run(*api, EVAL_ENV_CACHE_ROOT=cache)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertNotIn("EVAL_APERTUS_COMMON_WHEELS", manifest)
        self.assertIn("'$EVAL_APERTUS_COMMON_WHEELS'/*.whl", SBATCH)
        self.assertIn("--no-index", SBATCH)

    def test_prebuilt_container_config_names_the_edf_of_the_jobs(self) -> None:
        api = ["single", "--task", "gsm8k", "--backend", "openai", "--api-base-url", "http://localhost:8000",
               "--api-model-name", "test-model"]
        completed, _ = _prebuilt_dry_run(*api, EVAL_PREBUILT_CONTAINER_CONFIG="/nonexistent/prebuilt.toml")
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("EVAL_PREBUILT_CONTAINER_CONFIG is not a readable file", completed.stderr)
        with tempfile.NamedTemporaryFile("w", suffix=".toml") as edf:
            edf.write('image = "ghcr.io#example/prebuilt:tag"\n')
            edf.flush()
            completed, _ = _prebuilt_dry_run(*api, EVAL_PREBUILT_CONTAINER_CONFIG=edf.name)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        # Every job that needs no model server takes its container from that variable (default
        # env.toml): none may name env.toml itself.
        for script in ("run_lm_eval.sbatch", "aggregate_chunks.sbatch"):
            text = (REPO_ROOT / "scripts/lm_eval_harness" / script).read_text()
            self.assertNotIn("--environment=./containers/env.toml", text, script)
            self.assertIn("EVAL_PREBUILT_CONTAINER_CONFIG", text, script)

    def test_prebuilt_job_puts_the_image_venv_on_path_itself(self) -> None:
        # enroot/pyxis keep the base image's PATH, so the image's ENV PATH is not enough: python
        # would be the image's own, without lm-eval's packages (nor Python 3.13 for apertus-common).
        self.assertIn('export PATH=\\"\\${LM_EVAL_VENV:-/opt/lm-eval-venv}/bin:\\$PATH\\"', SBATCH)
        self.assertIn("LM_EVAL_VENV=/opt/lm-eval-venv", (REPO_ROOT / "containers/Dockerfile.eval").read_text())

    def test_prebuilt_environment_overlays_another_harness_commit(self) -> None:
        commit = "32b8b593890ec9a983c9aadf81a6a08d840beb63"
        args = ("single", "--task", "hellaswag", "--model", "test/model", "--harness-branch")
        with tempfile.TemporaryDirectory() as cache:
            overlay = Path(cache, "prebuilt-harness/swiss-ai/lm-evaluation-harness", commit)
            completed, manifest = _prebuilt_dry_run(*args, commit.upper(), EVAL_ENV_CACHE_ROOT=cache)
            output = completed.stdout + completed.stderr
            self.assertEqual(completed.returncode, 0, output)
            self.assertNotIn("prepare_eval_env.sbatch", output)
            self.assertIn(f"[DRY RUN] pip install --no-deps --target {overlay}", output)
            self.assertIn(f"EVAL_HARNESS_OVERLAY={overlay}", manifest)
            self.assertIn(f"EVAL_RESOLVED_HARNESS_COMMIT={commit}", manifest)
            # Installed once, the overlay is reused.
            overlay.mkdir(parents=True)
            (overlay / ".complete").touch()
            completed, manifest = _prebuilt_dry_run(*args, commit, EVAL_ENV_CACHE_ROOT=cache)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertNotIn("pip install", completed.stdout + completed.stderr)
            self.assertIn(f"EVAL_HARNESS_OVERLAY={overlay}", manifest)

    def test_prebuilt_environment_takes_a_harness_fork(self) -> None:
        commit = "c2efb6bd" + "0" * 32
        with tempfile.TemporaryDirectory() as cache:
            completed, manifest = _prebuilt_dry_run(
                "single", "--task", "hellaswag", "--model", "test/model",
                "--harness-repo", "someone/lm-evaluation-harness", "--harness-branch", commit,
                EVAL_ENV_CACHE_ROOT=cache,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("EVAL_RESOLVED_HARNESS_REPO=someone/lm-evaluation-harness", manifest)
        self.assertIn(f"/prebuilt-harness/someone/lm-evaluation-harness/{commit}", manifest)

    def test_bad_harness_overrides_are_refused(self) -> None:
        for repo, error in (
            # A fork has no pin: it needs a ref.
            ("someone/lm-evaluation-harness", "name one with --harness-branch"),
            ("https://github.com/x/y", "--harness-repo must be <owner>/<repo>"),
        ):
            with self.subTest(repo=repo):
                completed, _ = _prebuilt_dry_run(
                    "single", "--task", "hellaswag", "--model", "test/model", "--harness-repo", repo,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(error, completed.stderr)

    def test_max_length_reaches_the_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            completed = _launch(
                "single", "--task", "uncheatable_eval", "--model", "test/model",
                "--max-length", "65536", "--logs-root", tmp,
            )
            output = completed.stdout + completed.stderr
            self.assertEqual(completed.returncode, 0, output)
            self.assertIn("Max length: 65536", output)
            self.assertEqual(_run_config(tmp)["configuration"]["max_length"], "65536")

    def test_max_length_must_be_a_positive_integer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            completed = _launch(
                "single", "--task", "uncheatable_eval", "--model", "test/model",
                "--max-length", "32k", "--logs-root", tmp,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("--max-length must be a positive integer", completed.stderr)

    def test_api_retry_and_concurrency_flags_reach_the_openai_run(self) -> None:
        output = _dry_run(
            "single", "--task", "hellaswag", "--backend", "openai", "--api-base-url", "http://localhost:8000",
            "--api-model-name", "test-model", "--api-max-retries", "20", "--api-num-concurrent", "4",
        )
        self.assertIn("API max retries: 20, concurrent requests: 4", output)

    def test_api_retry_and_concurrency_default_to_the_sbatchs_own(self) -> None:
        output = _dry_run(
            "single", "--task", "hellaswag", "--backend", "openai", "--api-base-url", "http://localhost:8000",
            "--api-model-name", "test-model",
        )
        self.assertIn("API max retries: 3 (default), concurrent requests: 8 (default)", output)

    def test_api_retry_and_concurrency_flags_need_the_openai_backend(self) -> None:
        for flag in ("--api-max-retries", "--api-num-concurrent"):
            with tempfile.TemporaryDirectory() as tmp:
                completed = _launch("single", "--task", "hellaswag", "--model", "test/model", flag, "5", "--logs-root", tmp)
            self.assertNotEqual(completed.returncode, 0, flag)
            self.assertIn(f"{flag} only applies with --backend openai", completed.stderr + completed.stdout)

    def test_api_retry_and_concurrency_must_be_positive_integers(self) -> None:
        for flag, value in (("--api-max-retries", "many"), ("--api-num-concurrent", "0")):
            with tempfile.TemporaryDirectory() as tmp:
                completed = _launch(
                    "single", "--task", "hellaswag", "--backend", "openai", "--api-base-url", "http://localhost:8000",
                    "--api-model-name", "test-model", flag, value, "--logs-root", tmp,
                )
            self.assertNotEqual(completed.returncode, 0, flag)
            self.assertIn(f"{flag} must be a positive integer", completed.stderr + completed.stdout)

    def test_reasoning_effort_is_a_named_part_of_the_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            completed = _launch(
                "single", "--task", "aime25", "--model", "openai/gpt-oss-120b",
                "--reasoning-effort", "high", "--logs-root", tmp,
            )
            output = completed.stdout + completed.stderr
            self.assertEqual(completed.returncode, 0, output)
            self.assertIn("Reasoning effort: high", output)
            self.assertIn("Name:   gpt-oss-120b-effort-high", output)
            self.assertIn("Chat:   true", output)
            config = _run_config(tmp)
        self.assertEqual(config["configuration"]["reasoning_effort"], "high")

    def test_no_reasoning_effort_keeps_the_run_config_unchanged(self) -> None:
        # run_config_matches() compares exactly: a new always-present key would make
        # every earlier result look foreign and rerun it.
        with tempfile.TemporaryDirectory() as tmp:
            completed = _launch(
                "single", "--task", "aime25", "--model", "test/model", "--logs-root", tmp,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            config = _run_config(tmp)
        self.assertNotIn("reasoning_effort", config["configuration"])

    def test_reasoning_effort_is_refused_where_it_cannot_reach_the_template(self) -> None:
        cases = {
            "megatron backend": (
                ["--backend", "megatron_lm", "--reasoning-effort", "high"],
                "not supported with the megatron_lm backend",
            ),
            "no chat template": (
                ["--no-chat-template", "--reasoning-effort", "high"],
                "drop --no-chat-template",
            ),
            "not a level": (["--reasoning-effort", "very high"], "expects a level"),
        }
        for name, (arguments, message) in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                completed = _launch(
                    "single", "--task", "aime25", "--model", "test/model",
                    *arguments, "--logs-root", tmp,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(message, completed.stdout + completed.stderr)

    def test_openai_takes_the_chat_template_switches(self) -> None:
        # The harness renders the template client-side (local-completions), so they reach it.
        api = ["--backend", "openai", "--api-base-url", "http://localhost:8000",
               "--api-model-name", "test-model"]
        cases = {
            "effort": (["--reasoning-effort", "high"], {"reasoning_effort": "high"}),
            "thinking on": (["--enable-thinking"], {"enable_thinking": "true"}),
            "thinking off": (["--no-enable-thinking"], {"enable_thinking": "false"}),
            "both": (["--enable-thinking", "--reasoning-effort", "low"],
                     {"enable_thinking": "true", "reasoning_effort": "low"}),
        }
        for name, (arguments, expected) in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                completed = _launch(
                    "single", "--task", "aime25", *api, *arguments, "--logs-root", tmp,
                )
                output = completed.stdout + completed.stderr
                self.assertEqual(completed.returncode, 0, output)
                self.assertIn("Chat:   true", output)
                self.assertNotIn("no reasoning close token", output)
                config = _run_config(tmp)["configuration"]
                for key, value in expected.items():
                    self.assertEqual(config[key], value, key)

    def test_openai_api_endpoint_is_chosen_and_recorded(self) -> None:
        api = ["--backend", "openai", "--api-base-url", "http://localhost:8000",
               "--api-model-name", "test-model"]
        cases = {
            "default": ([], "false", "API endpoint: completions"),
            "completions": (["--api-endpoint", "completions"], "false", "API endpoint: completions"),
            "chat": (["--api-endpoint", "chat-completions"], "true", "API endpoint: chat-completions"),
            "chat with thinking": (
                ["--api-endpoint", "chat-completions", "--enable-thinking",
                 "--think-end-token", "</think>"],
                "true", "API endpoint: chat-completions",
            ),
        }
        for name, (arguments, recorded, summary) in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                completed = _launch(
                    "single", "--task", "gsm8k", *api, *arguments, "--logs-root", tmp,
                )
                output = completed.stdout + completed.stderr
                self.assertEqual(completed.returncode, 0, output)
                self.assertIn(summary, output)
                config = _run_config(tmp)["configuration"]
                self.assertEqual(config["api_chat_endpoint"], recorded)

    def test_openai_api_endpoint_refusals(self) -> None:
        api = ["--backend", "openai", "--api-base-url", "http://localhost:8000",
               "--api-model-name", "test-model"]
        cases = {
            "unknown value": (
                [*api, "--api-endpoint", "responses"],
                "--api-endpoint must be 'completions' or 'chat-completions'",
            ),
            "not openai": (
                ["--model", "test/model", "--api-endpoint", "chat-completions"],
                "--api-endpoint only applies with --backend openai",
            ),
            "chat without template": (
                [*api, "--api-endpoint", "chat-completions", "--no-chat-template"],
                "needs the chat template on every task",
            ),
            "chat by type": (
                [*api, "--api-endpoint", "chat-completions", "--chat-template-by-type"],
                "needs the chat template on every task",
            ),
        }
        for name, (arguments, message) in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                completed = _launch("single", "--task", "gsm8k", *arguments, "--logs-root", tmp)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(message, completed.stdout + completed.stderr)

    def test_evaluate_passes_thinking_to_openai_chat_completions(self) -> None:
        # The server renders the template there; the harness sends the template args as
        # chat_template_kwargs and puts the server's reasoning field back between the tokens.
        sbatch = (REPO_ROOT / "scripts/lm_eval_harness/run_lm_eval.sbatch").read_text()
        completions = sbatch[sbatch.index('if [[ $API_LM_EVAL_MODEL == "local-completions" ]]; then'):]
        chat = completions[completions.index("    else"): completions.index("    fi")]
        self.assertIn("${THINKING_DEFAULT_MODEL_ARG}${THINK_ARGS}", chat)
        self.assertIn(',enable_thinking=${ENABLE_THINKING}"', chat)
        self.assertNotIn("need the completions endpoint", sbatch)

    def test_openai_takes_thinking_and_its_metrics(self) -> None:
        # The harness strips and measures the reasoning on API models (lm-evaluation-harness #111).
        api = ["--backend", "openai", "--api-base-url", "http://localhost:8000",
               "--api-model-name", "test-model"]
        cases = {
            "umbrella with tokens": ["--thinking", "--think-start-token", "<think>",
                                     "--think-end-token", "</think>"],
            "umbrella, autodetect": ["--thinking"],
            "close token": ["--enable-thinking", "--think-end-token", "</think>"],
            "gpt-oss recipe": ["--thinking", "--reasoning-effort", "high",
                               "--think-start-token", "<|channel|>analysis<|message|>",
                               "--think-end-token", "<|channel|>final<|message|>"],
            "lengths": ["--thinking", "--log-length-metrics"],
        }
        for name, arguments in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                completed = _launch(
                    "single", "--task", "aime25", *api, *arguments, "--logs-root", tmp,
                )
                output = completed.stdout + completed.stderr
                self.assertEqual(completed.returncode, 0, output)
                self.assertNotIn("not supported with the openai backend", output)
                config = _run_config(tmp)["configuration"]
                self.assertEqual(config["enable_thinking"], "true")

    def test_openai_thinking_metrics_still_need_a_close_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            completed = _launch(
                "single", "--task", "aime25", "--backend", "openai",
                "--api-base-url", "http://localhost:8000", "--api-model-name", "test-model",
                "--enable-thinking", "--track-thinking-metrics", "true", "--logs-root", tmp,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("no reasoning close token", completed.stdout + completed.stderr)

    def test_evaluate_passes_think_args_to_openai_and_raises_the_effort_budget(self) -> None:
        sbatch = (REPO_ROOT / "scripts/lm_eval_harness/run_lm_eval.sbatch").read_text()
        completions = sbatch[sbatch.index('if [[ $API_LM_EVAL_MODEL == "local-completions" ]]; then'):]
        completions = completions[: completions.index("    else")]
        self.assertIn("${THINKING_DEFAULT_MODEL_ARG}${THINK_ARGS}", completions)
        self.assertNotIn("unsupported with LM_EVAL_BACKEND=openai", sbatch)
        self.assertIn('if [[ "$ENABLE_THINKING" == "True" || -n "$REASONING_EFFORT" ]]; then', sbatch)
        self.assertIn(
            'if [[ "$ENABLE_THINKING" == "True" || -n "$REASONING_EFFORT" ]] && (( ! MAX_NEW_TOKENS_EXPLICIT )); then',
            sbatch,
        )

    def test_thinking_budget_replaces_task_local_max_gen_toks(self) -> None:
        # The --gen_kwargs value overrides every task's YAML max_gen_toks (mmlu_pro 2048,
        # ifeval 1280); 32768 is AIME's own budget, so no task is lowered.
        self.assertIn("    MAX_NEW_TOKENS=32768\n", SBATCH)
        self.assertEqual(
            _gen_kwargs(ENABLE_THINKING="True"),
            "do_sample=true,temperature=0.6,top_p=0.95,max_gen_toks=32768",
        )
        self.assertEqual(_gen_kwargs(REASONING_EFFORT="low", MAX_NEW_TOKENS="8192"), "max_gen_toks=8192")
        self.assertEqual(_gen_kwargs(), "")

    def test_a_run_name_with_whitespace_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            completed = _launch(
                "single", "--task", "hellaswag", "--model", "test/model",
                "--name", "test im", "--logs-root", tmp,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("must not contain whitespace", completed.stdout + completed.stderr)

    def test_openai_template_switches_need_the_chat_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            completed = _launch(
                "single", "--task", "aime25", "--backend", "openai",
                "--api-base-url", "http://localhost:8000", "--api-model-name", "test-model",
                "--no-chat-template", "--enable-thinking", "--logs-root", tmp,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("drop --no-chat-template", completed.stdout + completed.stderr)

    def test_evaluate_passes_template_args_to_openai_completions(self) -> None:
        sbatch = (REPO_ROOT / "scripts/lm_eval_harness/run_lm_eval.sbatch").read_text()
        self.assertIn(
            "python -m lm_eval --model $API_LM_EVAL_MODEL --model_args '$COMMON_MODEL_ARGS'$CHAT_TEMPLATE_ARGS ",
            sbatch,
        )
        completions = sbatch[sbatch.index('if [[ $API_LM_EVAL_MODEL == "local-completions" ]]; then'):]
        completions = completions[: completions.index("    else")]
        self.assertIn(',enable_thinking=${ENABLE_THINKING}"', completions)
        self.assertIn("${THINKING_DEFAULT_MODEL_ARG}", completions)

    def test_launch_summary_shows_partition_and_qos(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            completed = _launch(
                "single", "--task", "hellaswag", "--model", "test/model",
                "--qos", "highprio", "--partition", "highprio", "--logs-root", tmp,
            )
        output = completed.stdout + completed.stderr
        self.assertEqual(completed.returncode, 0, output)
        self.assertIn("Slurm:  partition=highprio qos=highprio", output)

    def test_no_partition_is_hard_coded_on_an_sbatch_command_line(self) -> None:
        # A --partition on the command line beats SBATCH_PARTITION, so it must follow it.
        for script in ("scripts/lm_eval_harness/launch_evaluations.sh", "scripts/lm_eval_harness/evaluation_orchestrator.sh"):
            text = (REPO_ROOT / script).read_text()
            self.assertNotIn("--partition=normal", text, script)
        orchestrator = (REPO_ROOT / "scripts/lm_eval_harness/evaluation_orchestrator.sh").read_text()
        self.assertIn('launch_args+=(--partition "$SBATCH_PARTITION")', orchestrator)

    def test_reasoning_effort_is_accepted_on_sglang(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            completed = _launch(
                "single", "--task", "aime25", "--model", "openai/gpt-oss-120b",
                "--backend", "sglang", "--reasoning-effort", "high", "--logs-root", tmp,
            )
            output = completed.stdout + completed.stderr
            self.assertEqual(completed.returncode, 0, output)
            config = _run_config(tmp)
        self.assertEqual(config["configuration"]["reasoning_effort"], "high")

    def test_evaluate_passes_think_args_to_sglang(self) -> None:
        sbatch = (REPO_ROOT / "scripts/lm_eval_harness/run_lm_eval.sbatch").read_text()
        sglang_args = next(
            line for line in sbatch.splitlines()
            if "tokenizer_path=$TOKENIZER" in line and "COMMON_MODEL_ARGS=" in line
        )
        self.assertIn("${THINK_ARGS}", sglang_args)

    def test_evaluate_passes_reasoning_effort_as_its_own_model_args_token(self) -> None:
        # The harness refuses a first --model_args token containing '{', and argparse gives
        # `--model_args='...'` exactly one value: the dict must be a second, space-separated token.
        # The harness splits it with shlex (POSIX) and reads the value with ast.literal_eval
        # (lm_eval/_cli/utils.py key_val_to_dict), so its quotes must survive shlex.
        assignment = next(line.strip() for line in SBATCH.splitlines()
                          if line.strip().startswith('CHAT_TEMPLATE_ARGS=" '))
        script = assignment + '\neval "set -- \'a=1\'$CHAT_TEMPLATE_ARGS"\nprintf "%s" "$2"\n'
        token = subprocess.run(["bash", "-c", script], env={"PATH": os.environ["PATH"], "REASONING_EFFORT": "high"},
                               capture_output=True, text=True, check=True).stdout
        lexer = shlex.shlex(token, posix=True)
        lexer.whitespace, lexer.whitespace_split, lexer.commenters = ",", True, ""
        key, value = next(iter(lexer)).split("=", 1)
        self.assertEqual((key, ast.literal_eval(value)), ("chat_template_args", {"reasoning_effort": "high"}))
        for launcher in ("python -m lm_eval", "python -m accelerate.commands.launch --num_processes=4  -m lm_eval"):
            self.assertIn(
                f"{launcher} --model $LM_EVAL_BACKEND --model_args '$COMMON_MODEL_ARGS'$CHAT_TEMPLATE_ARGS ",
                SBATCH,
            )

    def test_chat_template_by_type_is_recorded_in_the_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            completed = _launch(
                "single", "--task", "gsm8k,arc_easy", "--backend", "openai",
                "--api-base-url", "http://localhost:8000", "--api-model-name", "test-model",
                "--chat-template-by-type", "--reasoning-effort", "high", "--logs-root", tmp,
            )
            output = completed.stdout + completed.stderr
            self.assertEqual(completed.returncode, 0, output)
            config = _run_config(tmp)
        # Template switches only touch the generative tasks, so they keep by-type.
        self.assertEqual(config["configuration"]["apply_chat_template"], "by-type")
        self.assertIn("Chat:   by-type", output)

    def test_evaluate_runs_each_kind_as_its_own_pass(self) -> None:
        sbatch = (REPO_ROOT / "scripts/lm_eval_harness/run_lm_eval.sbatch").read_text()
        self.assertIn("true|false|by-type) ;;", sbatch)
        self.assertIn("python -m scripts.lm_eval_harness.task_output_types --tasks '$TASKS'", sbatch)
        self.assertIn('make_cmd "$CHAT_TASKS" "$PASS_ROOT/chat" true', sbatch)
        self.assertIn('make_cmd "$RAW_TASKS" "$PASS_ROOT/raw" false', sbatch)
        self.assertIn(
            "python -m scripts.results.merge_split_results --split_dirs '$PASS_ROOT/chat' '$PASS_ROOT/raw' --output_dir '$HARNESS_EVAL_DIR'",
            sbatch,
        )
        # The chat-template flags are per pass, never in the shared arguments.
        common = sbatch[sbatch.index("COMMON_EVAL_ARGS=("):]
        common = common[: common.index(")")]
        self.assertNotIn("--apply_chat_template", common)
        self.assertNotIn("--tasks", common)

    def test_apertus2_is_a_recorded_thinking_run(self) -> None:
        apertus2 = ["--backend", "openai", "--api-base-url", "http://localhost:8000",
                    "--api-model-name", "test-model", "--chat-format", "apertus2",
                    "--tokenizer", "/ckpt/tokenizer", "--max-length", "65536"]
        with tempfile.TemporaryDirectory() as tmp:
            completed = _launch("single", "--task", "gsm8k", *apertus2, "--logs-root", tmp)
            output = completed.stdout + completed.stderr
            self.assertEqual(completed.returncode, 0, output)
            self.assertIn("Chat format: apertus2 (prefill=after_think)", output)
            self.assertIn("Name:   test-model-apertus2-after_think-effort-medium", output)
            # Built environments get the pinned harness, which has apertus2-completions.
            self.assertIn(f"(swiss-ai/lm-evaluation-harness@{_harness_pin()})", output)
            config = _run_config(tmp)["configuration"]
        library = (REPO_ROOT / "requirements/apertus2.txt").read_text().rsplit("@", 1)[1].strip()
        self.assertEqual(
            {key: config[key] for key in ("chat_format", "prefill_mode", "reasoning_effort",
                                          "apply_chat_template", "max_length", "tokenizer",
                                          "log_length_metrics", "apertus_common", "harness_ref")},
            {"chat_format": "apertus2", "prefill_mode": "after_think", "reasoning_effort": "medium",
             "apply_chat_template": "true", "max_length": "65536", "tokenizer": "/ckpt/tokenizer",
             "log_length_metrics": "true", "apertus_common": library, "harness_ref": _harness_pin()},
        )
        with tempfile.TemporaryDirectory() as tmp:
            completed = _launch("single", "--task", "gsm8k", *apertus2, "--logs-root", tmp,
                                LM_EVAL_HARNESS_BRANCH="my-branch")
            self.assertIn("(swiss-ai/lm-evaluation-harness@my-branch)", completed.stdout)
        # --chat-format template wins over an ambient CHAT_FORMAT, in the job too.
        with tempfile.TemporaryDirectory() as tmp:
            completed = _launch("single", "--task", "gsm8k", *apertus2[:6], "--chat-format", "template",
                                "--logs-root", tmp, CHAT_FORMAT="apertus2", PREFILL_MODE="empty_think")
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertNotIn("chat_format", _run_config(tmp)["configuration"])
            self.assertNotIn("apertus2", completed.stdout)

    def test_apertus2_refusals(self) -> None:
        api = ["--backend", "openai", "--api-base-url", "http://localhost:8000",
               "--api-model-name", "test-model"]
        apertus2 = [*api, "--chat-format", "apertus2", "--tokenizer", "/ckpt", "--max-length", "65536"]
        single = ["single", "--task", "gsm8k"]
        cases = {
            "not openai": (["--model", "m", "--chat-format", "apertus2", "--tokenizer", "/ckpt",
                            "--max-length", "65536"], "needs --backend openai"),
            "preset suite": (["posttrain", *apertus2], "runs generative tasks only"),
            "no tokenizer": ([*api, "--chat-format", "apertus2", "--max-length", "65536"],
                             "needs --backend openai, --tokenizer"),
            "no max length": ([*api, "--chat-format", "apertus2", "--tokenizer", "/ckpt"],
                              "--max-length <the server's --max-model-len>"),
            "template off": ([*apertus2, "--no-chat-template"], "renders every task as chat"),
            "by type": ([*apertus2, "--chat-template-by-type"], "renders every task as chat"),
            "chat endpoint": ([*apertus2, "--api-endpoint", "chat-completions"], "renders every task as chat"),
            "ambient chat endpoint": (apertus2, "renders every task as chat", {"API_CHAT_ENDPOINT": "true"}),
            "thinking": ([*apertus2, "--thinking"], "always reasons"),
            "ambient thinking switch": (apertus2, "always reasons", {"ENABLE_THINKING": "false"}),
            "think token": ([*apertus2, "--think-end-token", "</think>"], "always reasons"),
            "prefill mode": ([*apertus2, "--prefill-mode", "reject"], "--prefill-mode must be"),
            "prefill without apertus2": ([*api, "--prefill-mode", "empty_think"],
                                         "--prefill-mode only applies with --chat-format apertus2"),
            "unknown format": ([*api, "--chat-format", "harmony"], "--chat-format must be"),
            "bos": ([*apertus2, "--bos"], "drop --bos"),
            "ambient bos": (apertus2, "drop --bos", {"BOS": "true"}),
            "convert": ([*apertus2, "--model", "/meg/ckpt", "--convert-to-hf", "--megatron-iter", "1000"],
                        "drop --convert-to-hf"),
            "effort": ([*apertus2, "--reasoning-effort", "xhigh"], "low, medium or high"),
        }
        for name, (arguments, message, *env) in cases.items():
            if arguments[0] != "posttrain":
                arguments = [*single, *arguments]
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                completed = _launch(*arguments, "--logs-root", tmp, **(env[0] if env else {}))
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(message, completed.stdout + completed.stderr)
        # An explicit --api-endpoint completions overrides an ambient API_CHAT_ENDPOINT.
        with tempfile.TemporaryDirectory() as tmp:
            completed = _launch(*single, *apertus2, "--api-endpoint", "completions", "--logs-root", tmp,
                                API_CHAT_ENDPOINT="true")
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_evaluate_checks_apertus2_runs(self) -> None:
        accepted = _chat_format()
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        # The prebuilt image runs this format too (the job installs apertus-common).
        prebuilt = _chat_format(EVAL_ENV_KIND="prebuilt")
        self.assertEqual(prebuilt.returncode, 0, prebuilt.stderr)
        refusals = {
            "not openai": ({"LM_EVAL_BACKEND": "vllm"}, "renders every task as chat"),
            "chat endpoint": ({"API_CHAT_ENDPOINT": "true"}, "renders every task as chat"),
            "template off": ({"APPLY_CHAT_TEMPLATE": "false"}, "renders every task as chat"),
            "no context": ({"MAX_LENGTH": ""}, "needs MAX_LENGTH"),
            "no tokenizer": ({"TOKENIZER": ""}, "needs TOKENIZER"),
            "bos": ({"BOS": "true"}, "unset BOS"),
            "prefill mode": ({"PREFILL_MODE": "reject"}, "PREFILL_MODE must be"),
            "thinking switch": ({"ENABLE_THINKING_EXPLICIT": "1"}, "always reasons"),
            "think token": ({"THINK_END_TOKEN": "</think>"}, "always reasons"),
            "autodetect": ({"AUTODETECT_THINK_TOKENS": "True"}, "always reasons"),
            "thinking metrics": ({"TRACK_THINKING_METRICS": "True"}, "always reasons"),
        }
        for name, (env, message) in refusals.items():
            with self.subTest(name):
                completed = _chat_format(**env)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(message, completed.stderr)

    def test_evaluate_selects_apertus2_completions(self) -> None:
        def model(**env: str) -> str:
            return _openai_model(**env).stdout.splitlines()[-1]

        apertus2 = _openai_model(CHAT_FORMAT="apertus2")
        self.assertEqual(apertus2.returncode, 0, apertus2.stderr)
        self.assertEqual(
            apertus2.stdout.splitlines()[-1],
            "apertus2-completions model=m,base_url=http://host:8000/v1/completions,num_concurrent=8,"
            "max_retries=3,timeout=3600,tokenizer=/ckpt,prefill=after_think,reasoning_effort=medium,"
            "max_gen_toks=32768,max_model_len=65536",
        )
        # ruler's detected length bounds the template path's context, not the server's.
        self.assertTrue(model(CHAT_FORMAT="apertus2", BACKEND_MAX_LENGTH="8792").endswith(",max_model_len=65536"))
        # The template path keeps its own model, context argument and chat_template_args token.
        template = model()
        self.assertTrue(template.startswith("local-completions "))
        self.assertTrue(template.endswith(",max_length=65536,clamp_gen_toks=True 'chat_template_args={...}'"))

    def test_apertus2_samples_like_a_thinking_run(self) -> None:
        self.assertEqual(
            _gen_kwargs(CHAT_FORMAT="apertus2", REASONING_EFFORT="medium"),
            "do_sample=true,temperature=0.6,top_p=0.95,max_gen_toks=32768",
        )
        self.assertEqual(
            _gen_kwargs(CHAT_FORMAT="apertus2", REASONING_EFFORT="medium", THINK_TEMPERATURE="0.8",
                        THINK_TOP_P="1.0", THINK_TOP_K="20", THINK_MIN_P="0.05",
                        THINK_REPETITION_PENALTY="1.05"),
            "do_sample=true,temperature=0.8,top_p=1.0,top_k=20,min_p=0.05,repetition_penalty=1.05,"
            "max_gen_toks=32768",
        )

    def test_sampling_flags_reach_the_job_and_the_run_config(self) -> None:
        apertus2 = ["--backend", "openai", "--api-base-url", "http://localhost:8000",
                    "--api-model-name", "test-model", "--chat-format", "apertus2",
                    "--tokenizer", "/ckpt/tokenizer", "--max-length", "65536"]

        def config(*extra: str) -> dict:
            with tempfile.TemporaryDirectory() as tmp:
                completed = _launch("single", "--task", "gsm8k", *apertus2, *extra, "--logs-root", tmp)
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                self.last_output = completed.stdout + completed.stderr
                return _run_config(tmp)["configuration"]

        # No flag: nothing is exported, so an earlier run's configuration (and resume signature) is unchanged.
        plain = config()
        self.assertEqual((plain["think_temperature"], plain["think_top_p"]), ("", ""))
        # Flags are resolved at launch, exported explicitly and recorded; "none" leaves a sampler out.
        flagged = config("--temperature", "1", "--top-p", "none", "--top-k", "20")
        self.assertEqual(
            (flagged["think_temperature"], flagged["think_top_p"], flagged["think_top_k"]),
            ("1.0", "none", "20"),
        )
        self.assertEqual(
            _gen_kwargs(CHAT_FORMAT="apertus2", REASONING_EFFORT="medium", THINK_TEMPERATURE="1.0",
                        THINK_TOP_P="none", THINK_TOP_K="20", THINK_MIN_P="none",
                        THINK_REPETITION_PENALTY="none"),
            "do_sample=true,temperature=1.0,top_k=20,max_gen_toks=32768",
        )
        # A profile replaces the framework default; greedy on a thinking run is called out.
        greedy = config("--sampling-profile", "greedy")
        self.assertEqual((greedy["think_temperature"], greedy["think_top_p"]), ("0.0", "none"))
        self.assertIn("greedy decoding", self.last_output)

    def test_sampling_flags_on_a_run_that_does_not_think(self) -> None:
        api = ["--backend", "openai", "--api-base-url", "http://localhost:8000", "--api-model-name", "test-model"]
        with tempfile.TemporaryDirectory() as tmp:
            refused = _launch("single", "--task", "gsm8k", *api, "--top-p", "0.9", "--logs-root", tmp)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("need --temperature", refused.stdout + refused.stderr)
        with tempfile.TemporaryDirectory() as tmp:
            invalid = _launch("single", "--task", "gsm8k", *api, "--temperature", "-1", "--logs-root", tmp)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertIn("temperature must be >= 0", invalid.stdout + invalid.stderr)
        with tempfile.TemporaryDirectory() as tmp:
            sampled = _launch("single", "--task", "gsm8k", *api, "--temperature", "0.7", "--top-k", "20",
                              "--logs-root", tmp)
            self.assertEqual(sampled.returncode, 0, sampled.stdout + sampled.stderr)
            config = _run_config(tmp)["configuration"]
        self.assertEqual((config["nothink_temperature"], config["nothink_top_p"], config["think_top_k"]),
                         ("0.7", "none", "20"))
        self.assertEqual(
            _gen_kwargs(NOTHINK_TEMPERATURE="0.7", NOTHINK_TOP_P="none", THINK_TOP_K="20"),
            "do_sample=true,temperature=0.7,top_k=20",
        )
        self.assertEqual(_gen_kwargs(NOTHINK_TEMPERATURE="0.7"), "do_sample=true,temperature=0.7,top_p=0.95")

    def test_invalid_sampling_environment_fails_the_job(self) -> None:
        # A bad value must stop the job, not be dropped from the request.
        for env in ({"CHAT_FORMAT": "apertus2", "THINK_TOP_P": "3"}, {"NOTHINK_TEMPERATURE": "-1"}):
            with self.subTest(env):
                with self.assertRaises(subprocess.CalledProcessError):
                    _gen_kwargs(**env)

    def test_apertus2_requirements_pin_full_commits(self) -> None:
        pins = [
            line for line in (REPO_ROOT / "requirements/apertus2.txt").read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertTrue(pins)
        for pin in pins:
            self.assertRegex(pin, r" @ git\+https://github\.com/swiss-ai/[\w.-]+\.git@[0-9a-f]{40}$")

    def test_harness_pins_only_the_swiss_ai_fork(self) -> None:
        pins = [
            line.split()
            for line in (REPO_ROOT / "requirements/lm-eval-harness.txt").read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertEqual(
            sorted(repo for repo, _ in pins),
            ["swiss-ai/lm-evaluation-harness"],
        )
        for _, commit in pins:
            self.assertRegex(commit, r"^[0-9a-f]{40}$")


if __name__ == "__main__":
    unittest.main()
