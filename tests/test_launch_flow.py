import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.lm_eval_harness import job_settings


REPO_ROOT = Path(__file__).resolve().parents[1]

# Launcher inputs a caller's shell may export; the dry runs start without them.
_LAUNCH_INPUTS = {
    "TOKENIZER", "MAX_LENGTH", "BOS", "CHAT_FORMAT", "PREFILL_MODE", "ANSWER_FRAME", "EVAL_PREBUILT_ENV",
    "LM_EVAL_HARNESS_BRANCH", "LM_EVAL_HARNESS_REPO", "EVAL_ENV_CACHE_ROOT", "API_CHAT_ENDPOINT", "ENABLE_THINKING",
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


# The launcher needs an endpoint and a served name.
OPENAI_ARGS = (
    "--api-base-url", "http://localhost:8000", "--api-model-name", "test-model",
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


def _gen_kwargs(**env: str) -> str:
    """The --gen_kwargs of a run with this environment (scripts/lm_eval_harness/job_settings.py)."""
    base = {"MODEL": "m", "API_BASE_URL": "http://host:8000", "TOKENIZER": "/ckpt", "MAX_LENGTH": "65536",
            "MAX_NEW_TOKENS": "32768"}
    outputs, _ = job_settings.build({**base, **env})
    return outputs["GEN_KWARGS"]


class LaunchFlowTests(unittest.TestCase):
    def test_mode_is_required_and_only_single_and_custom_exist(self) -> None:
        for arguments, message in ((["--model", "test/model"], "a mode is required"),
                                   (["posttrain", "--model", "test/model"], "invalid mode 'posttrain'")):
            completed = _launch(*arguments)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(message, completed.stdout + completed.stderr)

    def test_lm_eval_runs_in_a_cpu_only_job(self) -> None:
        output = _dry_run(
            "single",
            "--task",
            "hellaswag",
            "--api-base-url",
            "http://localhost:8000",
            "--api-model-name",
            "test-model",
        )
        self.assertIn("scripts/lm_eval_harness/run_lm_eval.sbatch test-model test-model", output)
        self.assertNotIn("#SBATCH --gres", SBATCH)
        self.assertNotIn("#SBATCH --exclusive", SBATCH)

    def test_the_backend_flags_are_gone(self) -> None:
        for flag in (("--backend", "openai"), ("--megatron-iter", "4000000"), ("--script", "models.sh")):
            with self.subTest(flag=flag[0]), tempfile.TemporaryDirectory() as tmp:
                completed = _launch("single", "--task", "hellaswag", *OPENAI_ARGS, *flag, "--logs-root", tmp)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("Unknown option", completed.stdout + completed.stderr)

    def test_trust_remote_code_flag_is_accepted(self) -> None:
        output = _dry_run(
            "single",
            "--task",
            "hellaswag",
            "--api-base-url",
            "http://localhost:8000",
            "--api-model-name",
            "test-model",
            "--trust-remote-code",
        )
        self.assertNotIn("Unknown option", output)

    def test_orchestrator_records_bos_by_client_as_applied(self) -> None:
        orchestrator = (REPO_ROOT / "scripts/lm_eval_harness/evaluation_orchestrator.sh").read_text()
        func = orchestrator[
            orchestrator.index("_eval_resolved_bos_by_client() {"): orchestrator.index("_eval_create_run_config() {")
        ]

        def recorded(**env: str) -> str:
            exports = "".join(f"export {k}={v}\n" for k, v in env.items())
            script = f"set -euo pipefail\n{exports}{func}\n_eval_resolved_bos_by_client\n"
            return subprocess.run(["bash", "-c", script], text=True, capture_output=True).stdout.strip()

        self.assertEqual(recorded(APPLY_CHAT_TEMPLATE="true"), "true")
        self.assertEqual(recorded(APPLY_CHAT_TEMPLATE="false"), "false")
        self.assertEqual(recorded(APPLY_CHAT_TEMPLATE="by-type"), "chat=true,raw=false")
        self.assertEqual(recorded(APPLY_CHAT_TEMPLATE="true", BOS_BY_CLIENT="false"), "false")
        self.assertEqual(recorded(APPLY_CHAT_TEMPLATE="true", API_CHAT_ENDPOINT="true"), "false")
        self.assertEqual(recorded(APPLY_CHAT_TEMPLATE="true", CHAT_FORMAT="apertus2"), "false")

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
                    "single", "--task", "hellaswag", "--model", "test/model", *OPENAI_ARGS,
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

    def test_prebuilt_image_has_the_pinned_apertus_common(self) -> None:
        # The image installs requirements/apertus2.txt and the job checks the import: the launcher
        # builds and names no wheel.
        dockerfile = (REPO_ROOT / "containers/Dockerfile.eval").read_text()
        self.assertIn("-r /tmp/requirements/apertus2.txt", dockerfile)
        self.assertIn("lm_eval, apertus_common'", SBATCH)
        self.assertNotIn("EVAL_APERTUS_COMMON_WHEELS", SBATCH)
        self.assertNotIn("EVAL_APERTUS_COMMON_WHEELS", (REPO_ROOT / "scripts/lm_eval_harness/evaluation_orchestrator.sh").read_text())

    def test_prebuilt_container_config_names_the_edf_of_the_jobs(self) -> None:
        api = ["single", "--task", "gsm8k", "--api-base-url", "http://localhost:8000",
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

    def test_aggregation_uploads_from_the_image_venv(self) -> None:
        # The W&B upload of aggregate_chunks.sbatch (merging chunks, or results found complete) runs
        # in the same container: without the venv on PATH its python has no wandb.
        text = (REPO_ROOT / "scripts/lm_eval_harness/aggregate_chunks.sbatch").read_text()
        upload = text[text.index("WANDB_CMD="):]
        self.assertIn('export PATH=\\"\\${LM_EVAL_VENV:-/opt/lm-eval-venv}/bin:\\$PATH\\"', upload)
        self.assertLess(upload.index("export PATH="), upload.index("$WANDB_CMD\n    \""))

    def test_prebuilt_environment_overlays_another_harness_commit(self) -> None:
        commit = "32b8b593890ec9a983c9aadf81a6a08d840beb63"
        args = ("single", "--task", "hellaswag", "--model", "test/model", *OPENAI_ARGS, "--harness-branch")
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
                "single", "--task", "hellaswag", "--model", "test/model", *OPENAI_ARGS,
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
                    "single", "--task", "hellaswag", "--model", "test/model", *OPENAI_ARGS, "--harness-repo", repo,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(error, completed.stderr)

    def test_max_length_reaches_the_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            completed = _launch(
                "single", "--task", "uncheatable_eval", "--model", "test/model", *OPENAI_ARGS,
                "--max-length", "65536", "--logs-root", tmp,
            )
            output = completed.stdout + completed.stderr
            self.assertEqual(completed.returncode, 0, output)
            self.assertIn("Max length: 65536", output)
            self.assertEqual(_run_config(tmp)["configuration"]["max_length"], "65536")

    def test_max_length_must_be_a_positive_integer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            completed = _launch(
                "single", "--task", "uncheatable_eval", "--model", "test/model", *OPENAI_ARGS,
                "--max-length", "32k", "--logs-root", tmp,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("--max-length must be a positive integer", completed.stderr)

    def test_api_retry_and_concurrency_flags_reach_the_openai_run(self) -> None:
        output = _dry_run(
            "single", "--task", "hellaswag", "--api-base-url", "http://localhost:8000",
            "--api-model-name", "test-model", "--api-max-retries", "20", "--api-num-concurrent", "4",
        )
        self.assertIn("API max retries: 20, concurrent requests: 4", output)

    def test_api_retry_and_concurrency_default_to_the_sbatchs_own(self) -> None:
        output = _dry_run(
            "single", "--task", "hellaswag", "--api-base-url", "http://localhost:8000",
            "--api-model-name", "test-model",
        )
        self.assertIn("API max retries: 3 (default), concurrent requests: 8 (default)", output)

    def test_api_retry_and_concurrency_must_be_positive_integers(self) -> None:
        for flag, value in (("--api-max-retries", "many"), ("--api-num-concurrent", "0")):
            with tempfile.TemporaryDirectory() as tmp:
                completed = _launch(
                    "single", "--task", "hellaswag", "--api-base-url", "http://localhost:8000",
                    "--api-model-name", "test-model", flag, value, "--logs-root", tmp,
                )
            self.assertNotEqual(completed.returncode, 0, flag)
            self.assertIn(f"{flag} must be a positive integer", completed.stderr + completed.stdout)

    def test_reasoning_effort_is_a_named_part_of_the_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            completed = _launch(
                "single", "--task", "aime25", "--model", "openai/gpt-oss-120b", *OPENAI_ARGS,
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
                "single", "--task", "aime25", "--model", "test/model", *OPENAI_ARGS, "--logs-root", tmp,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            config = _run_config(tmp)
        self.assertNotIn("reasoning_effort", config["configuration"])

    def test_reasoning_effort_is_refused_where_it_cannot_reach_the_template(self) -> None:
        cases = {
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
        api = ["--api-base-url", "http://localhost:8000",
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
        api = ["--api-base-url", "http://localhost:8000",
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
        api = ["--api-base-url", "http://localhost:8000",
               "--api-model-name", "test-model"]
        cases = {
            "unknown value": (
                [*api, "--api-endpoint", "responses"],
                "--api-endpoint must be 'completions' or 'chat-completions'",
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

    def test_openai_takes_thinking_and_its_metrics(self) -> None:
        # The harness strips and measures the reasoning on API models (lm-evaluation-harness #111).
        api = ["--api-base-url", "http://localhost:8000",
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
                "single", "--task", "aime25", "--api-base-url", "http://localhost:8000", "--api-model-name", "test-model",
                "--enable-thinking", "--track-thinking-metrics", "true", "--logs-root", tmp,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("no reasoning close token", completed.stdout + completed.stderr)

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
                "single", "--task", "aime25", "--api-base-url", "http://localhost:8000", "--api-model-name", "test-model",
                "--no-chat-template", "--enable-thinking", "--logs-root", tmp,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("drop --no-chat-template", completed.stdout + completed.stderr)

    def test_launch_summary_shows_partition_and_qos(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            completed = _launch(
                "single", "--task", "hellaswag", "--model", "test/model", *OPENAI_ARGS,
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

    def test_chat_template_by_type_is_recorded_in_the_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            completed = _launch(
                "single", "--task", "gsm8k,arc_easy", "--api-base-url", "http://localhost:8000", "--api-model-name", "test-model",
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
        apertus2 = ["--api-base-url", "http://localhost:8000",
                    "--api-model-name", "test-model", "--chat-format", "apertus2",
                    "--tokenizer", "/ckpt/tokenizer", "--max-length", "65536"]
        with tempfile.TemporaryDirectory() as tmp:
            completed = _launch("single", "--task", "gsm8k", *apertus2, "--logs-root", tmp)
            output = completed.stdout + completed.stderr
            self.assertEqual(completed.returncode, 0, output)
            self.assertIn("Chat format: apertus2 (prefill=after_think)", output)
            self.assertIn("Name:   test-model-apertus2-after_think-effort-medium", output)
            self.assertNotIn("Answer frame", output)
            # Built environments get the pinned harness, which has apertus2-completions.
            self.assertIn(f"(swiss-ai/lm-evaluation-harness@{_harness_pin()})", output)
            config = _run_config(tmp)["configuration"]
        # The default run records no answer frame; --answer-frame claim is named, recorded and announced.
        self.assertNotIn("answer_frame", config)
        with tempfile.TemporaryDirectory() as tmp:
            claimed = _launch("single", "--task", "gsm8k", *apertus2, "--answer-frame", "claim",
                              "--logs-root", tmp)
            claimed_output = claimed.stdout + claimed.stderr
            self.assertEqual(claimed.returncode, 0, claimed_output)
            self.assertIn("Answer frame: claim", claimed_output)
            self.assertIn("Name:   test-model-apertus2-after_think-claim-effort-medium", claimed_output)
            self.assertEqual(_run_config(tmp)["configuration"]["answer_frame"], "claim")
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
        api = ["--api-base-url", "http://localhost:8000",
               "--api-model-name", "test-model"]
        apertus2 = [*api, "--chat-format", "apertus2", "--tokenizer", "/ckpt", "--max-length", "65536"]
        single = ["single", "--task", "gsm8k"]
        cases = {
            "no tokenizer": ([*api, "--chat-format", "apertus2", "--max-length", "65536"],
                             "needs --tokenizer"),
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
            "answer frame": ([*apertus2, "--answer-frame", "think"], "--answer-frame must be"),
            "answer frame without apertus2": ([*api, "--answer-frame", "claim"],
                                              "--answer-frame only applies with --chat-format apertus2"),
            "unknown format": ([*api, "--chat-format", "harmony"], "--chat-format must be"),
            "bos": ([*apertus2, "--bos"], "drop --bos"),
            "ambient bos": (apertus2, "drop --bos", {"BOS": "true"}),
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
        apertus2 = ["--api-base-url", "http://localhost:8000",
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
        api = ["--api-base-url", "http://localhost:8000", "--api-model-name", "test-model"]
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
                with self.assertRaisesRegex(ValueError, "Invalid sampling settings"):
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
