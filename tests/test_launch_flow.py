import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


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
            env=os.environ | {"WANDB_PROJECT": "launcher-test"},
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
            env=os.environ
            | {"WANDB_PROJECT": "launcher-test", "EVAL_PREBUILT_ENV": "1"}
            | extra_env,
            text=True,
            capture_output=True,
        )
        manifests = list(Path(tmp).rglob("environment.sh"))
        manifest = manifests[0].read_text() if manifests else ""
    return completed, manifest


def _launch(*arguments: str) -> subprocess.CompletedProcess:
    """A dry run that may fail; the caller passes --logs-root."""
    return subprocess.run(
        ["bash", "scripts/lm_eval_harness/launch_evaluations.sh", *arguments, "--debug"],
        cwd=REPO_ROOT,
        env=os.environ | {"WANDB_PROJECT": "launcher-test"},
        text=True,
        capture_output=True,
    )


def _run_config(logs_root: str) -> dict:
    configs = list(Path(logs_root).rglob("run_config.json"))
    assert len(configs) == 1, configs
    return json.loads(configs[0].read_text())


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
        self.assertIn("scripts/lm_eval_harness/evaluate_api.sbatch test-model test-model", output)
        self.assertNotIn("scripts/lm_eval_harness/evaluate.sbatch test-model test-model", output)
        wrapper = (REPO_ROOT / "scripts/lm_eval_harness/evaluate_api.sbatch").read_text()
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
        # Runs evaluate.sbatch's own block, so the test follows the script.
        sbatch = (REPO_ROOT / "scripts/lm_eval_harness/evaluate.sbatch").read_text()
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

    def test_prebuilt_environment_refuses_a_harness_branch(self) -> None:
        completed, _ = _prebuilt_dry_run(
            "single", "--task", "hellaswag", "--model", "test/model",
            LM_EVAL_HARNESS_BRANCH="some-branch",
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("cannot be used with EVAL_PREBUILT_ENV=1", completed.stderr)

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
        sbatch = (REPO_ROOT / "scripts/lm_eval_harness/evaluate.sbatch").read_text()
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
        sbatch = (REPO_ROOT / "scripts/lm_eval_harness/evaluate.sbatch").read_text()
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
        sbatch = (REPO_ROOT / "scripts/lm_eval_harness/evaluate.sbatch").read_text()
        self.assertIn("    MAX_NEW_TOKENS=32768\n", sbatch)
        block = sbatch[sbatch.index('GEN_KWARGS=""'): sbatch.index('MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-""}')]

        def gen_kwargs(**env: str) -> str:
            script = block + '\nprintf "%s" "$GEN_KWARGS" >&2\n'
            base = {"PATH": os.environ["PATH"], "ENABLE_THINKING": "False", "REASONING_EFFORT": "", "MAX_NEW_TOKENS": "32768"}
            done = subprocess.run(["bash", "-c", script], env=base | env, capture_output=True, text=True, check=True)
            return done.stderr

        self.assertEqual(
            gen_kwargs(ENABLE_THINKING="True"),
            "do_sample=true,temperature=0.6,top_p=0.95,max_gen_toks=32768",
        )
        self.assertEqual(gen_kwargs(REASONING_EFFORT="low", MAX_NEW_TOKENS="8192"), "max_gen_toks=8192")
        self.assertEqual(gen_kwargs(), "")

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
        sbatch = (REPO_ROOT / "scripts/lm_eval_harness/evaluate.sbatch").read_text()
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
        sbatch = (REPO_ROOT / "scripts/lm_eval_harness/evaluate.sbatch").read_text()
        sglang_args = next(
            line for line in sbatch.splitlines()
            if "tokenizer_path=$TOKENIZER" in line and "COMMON_MODEL_ARGS=" in line
        )
        self.assertIn("${THINK_ARGS}", sglang_args)

    def test_evaluate_passes_reasoning_effort_as_its_own_model_args_token(self) -> None:
        # The harness refuses a first --model_args token containing '{', and argparse gives
        # `--model_args='...'` exactly one value: the dict must be a second, space-separated token.
        sbatch = (REPO_ROOT / "scripts/lm_eval_harness/evaluate.sbatch").read_text()
        self.assertIn(
            r"""CHAT_TEMPLATE_ARGS=" 'chat_template_args={\"reasoning_effort\":\"${REASONING_EFFORT}\"}'" """.rstrip(),
            sbatch,
        )
        for launcher in ("python -m lm_eval", "python -m accelerate.commands.launch --num_processes=4  -m lm_eval"):
            self.assertIn(
                f"{launcher} --model $LM_EVAL_BACKEND --model_args '$COMMON_MODEL_ARGS'$CHAT_TEMPLATE_ARGS ",
                sbatch,
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
        sbatch = (REPO_ROOT / "scripts/lm_eval_harness/evaluate.sbatch").read_text()
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
