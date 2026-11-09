import json
import os
import subprocess
import unittest
from pathlib import Path

from scripts.common import thinking
from scripts.evalscope import run as evalscope_run
from scripts.harbor import run as harbor_run

REPO_ROOT = Path(__file__).resolve().parents[1]


class ThinkingModuleTest(unittest.TestCase):
    def test_nothing_set_sends_nothing(self):
        values = thinking.resolve()
        self.assertEqual(values, {})
        self.assertEqual(thinking.extra_body(values), {})
        self.assertEqual(thinking.litellm_kwargs(values), {})
        self.assertEqual(thinking.merge_extra_body(None, values), {})
        self.assertEqual(thinking.merge_extra_body({"min_p": 0.1}, values), {"min_p": 0.1})

    def test_switch_is_a_template_kwarg_only(self):
        values = thinking.resolve(enable_thinking=False)
        self.assertEqual(thinking.extra_body(values), {"chat_template_kwargs": {"enable_thinking": False}})
        self.assertEqual(
            thinking.litellm_kwargs(values), {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
        )

    def test_effort_reaches_the_server_twice(self):
        values = thinking.resolve(reasoning_effort="low")
        self.assertEqual(thinking.extra_body(values), {"chat_template_kwargs": {"reasoning_effort": "low"}})
        self.assertEqual(
            thinking.litellm_kwargs(values),
            {"reasoning_effort": "low", "extra_body": {"chat_template_kwargs": {"reasoning_effort": "low"}}},
        )

    def test_merge_keeps_other_entries_and_the_settings_win(self):
        values = thinking.resolve(enable_thinking=True, reasoning_effort="high")
        merged = thinking.merge_extra_body(
            {"min_p": 0.05, "chat_template_kwargs": {"enable_thinking": False, "other": 1}}, values
        )
        self.assertEqual(
            merged,
            {"min_p": 0.05, "chat_template_kwargs": {"enable_thinking": True, "other": 1, "reasoning_effort": "high"}},
        )

    def test_invalid_values(self):
        with self.assertRaises(ValueError):
            thinking.resolve(reasoning_effort="extreme")
        with self.assertRaises(ValueError):
            thinking.resolve(enable_thinking="yes")

    def _cli(self, *args):
        return subprocess.run(
            ["python3", "-m", "scripts.common.thinking", *args],
            cwd=REPO_ROOT, env={"PATH": os.environ["PATH"]}, capture_output=True, text=True,
        )

    def test_cli_formats(self):
        done = self._cli("--no-enable-thinking", "--reasoning-effort", "medium", "--format", "shell")
        self.assertEqual(
            done.stdout.splitlines(),
            ["THINKING_ENABLE=false", "THINKING_EFFORT=medium",
             "THINKING_EXTRA_BODY='\"chat_template_kwargs\": {\"enable_thinking\": false, \"reasoning_effort\": \"medium\"}'"],
        )
        self.assertEqual(self._cli().stdout.strip(), "{}")
        self.assertEqual(self._cli("--format", "shell").stdout.splitlines(),
                         ["THINKING_ENABLE=", "THINKING_EFFORT=''", "THINKING_EXTRA_BODY=''"])
        bad = self._cli("--reasoning-effort", "extreme")
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("reasoning effort must be one of", bad.stderr)
        both = self._cli("--enable-thinking", "--no-enable-thinking")
        self.assertNotEqual(both.returncode, 0)


INSPECT = (REPO_ROOT / "scripts/inspect_ai/run_inspect_eval.sh").read_text()
INSPECT_BLOCK = INSPECT[
    INSPECT.index("# Sampling of the model under test (scripts/common/sampling.py"):
    INSPECT.index('COMMON_ARGS+=(${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})')
]


def _inspect(enable="", effort="", extra_args=(), **sampling_flags):
    """The `inspect eval` arguments the sampling and thinking block adds (COMMON_ARGS)."""
    env = {"PATH": os.environ["PATH"], "THINKING_ENABLE_FLAG": enable, "THINKING_EFFORT_FLAG": effort}
    env |= {f"SAMPLING_{key.upper()}_FLAG": value for key, value in sampling_flags.items()}
    script = (
        "COMMON_ARGS=()\nEXTRA_ARGS=(" + " ".join(f"'{arg}'" for arg in extra_args) + ")\n"
        + INSPECT_BLOCK + 'printf "%s\\n" "${COMMON_ARGS[@]}"\n'
    )
    return subprocess.run(["bash", "-c", script], cwd=REPO_ROOT, env=env, capture_output=True, text=True)


class InspectThinkingTest(unittest.TestCase):
    def test_default_sends_no_thinking_setting(self):
        done = _inspect()
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.split(), ["--temperature", "0.6", "--top-p", "0.95"])

    def test_effort_and_switch(self):
        done = _inspect(enable="--no-enable-thinking", effort="low")
        self.assertEqual(done.returncode, 0, done.stderr)
        lines = done.stdout.splitlines()
        self.assertEqual(lines[lines.index("--reasoning-effort") + 1], "low")
        self.assertEqual(
            json.loads(lines[lines.index("--extra-body") + 1]),
            {"chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "low"}},
        )

    def test_shares_the_extra_body_with_the_sampling_keys(self):
        done = _inspect(enable="--enable-thinking", min_p="0.05")
        lines = done.stdout.splitlines()
        self.assertEqual(
            json.loads(lines[lines.index("--extra-body") + 1]),
            {"min_p": 0.05, "chat_template_kwargs": {"enable_thinking": True}},
        )

    def test_a_literal_extra_body_is_called_out(self):
        done = _inspect(enable="--enable-thinking", extra_args=("--extra-body", "{}"))
        self.assertIn("replaces the min-p/repetition-penalty/thinking", done.stderr)

    def test_invalid_effort_stops_the_run(self):
        done = _inspect(effort="extreme")
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("reasoning effort must be one of", done.stderr)


TAU2 = (REPO_ROOT / "scripts/aaii/run_tau2_bench.sh").read_text()
TAU2_BLOCK = TAU2[
    TAU2.index("# Thinking and reasoning effort of the agent model"):TAU2.index("# The commit a tau2-bench ref names")
]


def _tau2(agent_args='{"api_base": "http://x/v1", "api_key": "k"}', enable="", effort=""):
    env = {"PATH": os.environ["PATH"], "AGENT_LLM_ARGS": agent_args,
           "THINKING_ENABLE_FLAG": enable, "THINKING_EFFORT_FLAG": effort}
    script = "set -e\n" + TAU2_BLOCK + '\nprintf "%s" "$AGENT_LLM_ARGS"\n'
    return subprocess.run(["bash", "-c", script], cwd=REPO_ROOT, env=env, capture_output=True, text=True)


class Tau2ThinkingTest(unittest.TestCase):
    def test_no_flag_leaves_the_agent_args_alone(self):
        done = _tau2()
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(json.loads(done.stdout), {"api_base": "http://x/v1", "api_key": "k"})

    def test_merges_over_the_agent_args(self):
        done = _tau2(
            '{"api_base": "http://x/v1", "extra_body": {"top_k": 20, "chat_template_kwargs": {"x": 1}}}',
            enable="--enable-thinking", effort="high",
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(
            json.loads(done.stdout),
            {"api_base": "http://x/v1", "reasoning_effort": "high",
             "extra_body": {"top_k": 20,
                            "chat_template_kwargs": {"x": 1, "enable_thinking": True, "reasoning_effort": "high"}}},
        )

    def test_the_user_simulator_does_not_get_the_settings(self):
        # USER_LLM_ARGS is copied from the agent's before this block runs.
        self.assertLess(TAU2.index('USER_LLM_ARGS="$AGENT_LLM_ARGS"'), TAU2.index(TAU2_BLOCK))

    def test_invalid_effort_stops_the_run(self):
        done = _tau2(effort="extreme")
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("reasoning effort must be one of", done.stderr)


class HarborThinkingTest(unittest.TestCase):
    def _aks(self, *extra):
        args = harbor_run.parse_args(
            ["--benchmark", "terminal_bench_2", "--model", "org/m", "--api-base-url", "https://x/v1",
             "--work-dir", "w", "--sandbox", "k8s", *extra]
        )
        cmd = harbor_run.harbor_command(args, "terminal_bench_2", Path("w/jobs"))
        return [cmd[i + 1] for i, part in enumerate(cmd) if part == "--ak"]

    def test_default_sends_nothing_extra(self):
        self.assertEqual(self._aks(), ["temperature=0.6", 'llm_kwargs={"extra_body":{"top_p":0.95}}'])

    def test_flags(self):
        aks = self._aks("--no-enable-thinking", "--reasoning-effort", "low")
        self.assertEqual(aks[0], "temperature=0.6")
        self.assertIn("reasoning_effort=low", aks)
        llm_kwargs = json.loads(next(ak for ak in aks if ak.startswith("llm_kwargs=")).split("=", 1)[1])
        self.assertEqual(
            llm_kwargs["extra_body"],
            {"top_p": 0.95, "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "low"}},
        )

    def test_flags_win_over_the_users_own_agent_kwargs(self):
        aks = self._aks(
            "--agent-kwarg", "reasoning_effort=high",
            "--agent-kwarg", 'llm_kwargs={"extra_body":{"chat_template_kwargs":{"enable_thinking":true,"x":1}}}',
            "--reasoning-effort", "low", "--no-enable-thinking",
        )
        self.assertIn("reasoning_effort=low", aks)
        self.assertNotIn("reasoning_effort=high", aks)
        llm_kwargs = json.loads(next(ak for ak in aks if ak.startswith("llm_kwargs=")).split("=", 1)[1])
        self.assertEqual(
            llm_kwargs["extra_body"]["chat_template_kwargs"],
            {"enable_thinking": False, "x": 1, "reasoning_effort": "low"},
        )

    def test_other_agents_and_gold_runs_refuse_the_flags(self):
        for extra in (["--agent", "mini-swe-agent", "--reasoning-effort", "low"], ["--gold", "--no-enable-thinking"]):
            args = harbor_run.parse_args(
                ["--benchmark", "terminal_bench_2", "--model", "m", "--api-base-url", "https://x/v1",
                 "--work-dir", "w", "--sandbox", "k8s", *extra]
            )
            with self.assertRaises(SystemExit):
                harbor_run.agent_kwargs(args)

    def test_invalid_effort(self):
        with self.assertRaises(SystemExit):
            self._aks("--reasoning-effort", "extreme")


class EvalScopeThinkingTest(unittest.TestCase):
    def _generation_config(self, *extra):
        args = evalscope_run.parse_args(
            ["--benchmark", "swe_bench", "--model", "m", "--api-base-url", "http://x/v1", "--work-dir", "/w", *extra]
        )
        return evalscope_run.generation_config(args)

    def test_default_sends_nothing_extra(self):
        self.assertEqual(self._generation_config(), {"temperature": 0.6, "top_p": 0.95})

    def test_flags_set_effort_and_template_kwargs(self):
        cfg = self._generation_config("--reasoning-effort", "low", "--no-enable-thinking", "--min-p", "0.05")
        self.assertEqual(
            cfg,
            {"temperature": 0.6, "top_p": 0.95, "reasoning_effort": "low",
             "extra_body": {"min_p": 0.05,
                            "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "low"}}},
        )

    def test_flags_win_over_generation_config(self):
        cfg = self._generation_config(
            "--reasoning-effort", "low", "--enable-thinking", "--generation-config",
            '{"reasoning_effort": "high", "extra_body": {"chat_template_kwargs": {"enable_thinking": false, "x": 1}}}',
        )
        self.assertEqual(cfg["reasoning_effort"], "low")
        self.assertEqual(
            cfg["extra_body"]["chat_template_kwargs"], {"enable_thinking": True, "x": 1, "reasoning_effort": "low"}
        )

    def test_completions_refuses_the_flags(self):
        with self.assertRaises(SystemExit):
            self._generation_config("--completions", "--enable-thinking")

    def test_invalid_effort(self):
        with self.assertRaises(SystemExit):
            self._generation_config("--reasoning-effort", "extreme")


if __name__ == "__main__":
    unittest.main()
