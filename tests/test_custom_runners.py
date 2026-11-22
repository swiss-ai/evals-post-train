"""Offline checks of scripts/custom/run_tau2_bench.sh's argument handling (no network, model or venv)."""

import http.server
import json
import os
import subprocess
import threading
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def run(*args):
    return subprocess.run(
        ["bash", "scripts/custom/run_tau2_bench.sh", *args],
        cwd=REPO,
        env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"},
        capture_output=True,
        text=True,
        timeout=60,
    )


class CustomTau2Test(unittest.TestCase):
    def test_help_documents_the_ref_and_the_cache(self):
        result = run("--help")
        self.assertEqual(result.returncode, 0)
        for flag in ("--git-ref", "--repo", "--domain", "--cache-dir", "--user-llm"):
            self.assertIn(flag, result.stdout)

    def test_domain_and_model_are_required(self):
        self.assertIn("Missing --domain", run("--model", "m").stderr)
        self.assertIn("Missing --model", run("--domain", "retail").stderr)

    def test_domain_and_save_to_are_plain_names(self):
        self.assertIn("domain name", run("--domain", "a b; rm", "--model", "m").stderr)
        self.assertIn("plain name", run("--domain", "retail", "--model", "m", "--save-to", "../x").stderr)

    def test_the_sampling_flags_reach_the_sampling_helper(self):
        # Past the sampling step the run needs uv, which this environment lacks: that is where it must stop.
        result = run("--domain", "retail", "--model", "m", "--api-base-url", "http://x/v1", "--sampling-profile", "thinking",
                     "--temperature", "0.6", "--top-p", "0.95", "--reasoning-effort", "high")
        self.assertNotIn("unrecognized arguments", result.stderr)
        self.assertIn("uv is required", result.stderr)
        bad = run("--domain", "retail", "--model", "m", "--api-base-url", "http://x/v1", "--sampling-profile", "nope")
        self.assertNotEqual(bad.returncode, 0)
        self.assertNotIn("uv is required", bad.stderr)

    def test_unknown_argument_is_refused(self):
        result = run("--domain", "retail", "--model", "m", "--bogus")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unknown argument: --bogus", result.stderr)


class RetrievalDefaultTest(unittest.TestCase):
    """Runs the script in prebuilt mode against a stub `tau2` that records its arguments."""

    def _tau2_args(self, domain, *extra):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d, "upstream")
            (root / ".venv/bin").mkdir(parents=True)
            (root / "data").mkdir()
            (root / "data/x").write_text("")
            stub = root / ".venv/bin/tau2"
            stub.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > {d}/args\nexit 0\n')
            stub.chmod(0o755)
            ref = Path(d, "ref")
            ref.write_text("v2.0.0 " + "a" * 40 + "\n")
            subprocess.run(
                ["bash", "scripts/custom/run_tau2_bench.sh", "--domain", domain, "--model", "m", "--api-base-url", "http://x/v1",
                 "--workdir", f"{d}/work", *extra],
                cwd=REPO, capture_output=True, text=True, timeout=60,
                env={"PATH": "/usr/bin:/bin", "HOME": d, "EVAL_PREBUILT_ENV": "1",
                     "TAU2_UPSTREAM_DIR": str(root), "TAU2_UPSTREAM_REF_FILE": str(ref)},
            )
            return Path(d, "args").read_text().split()

    def test_banking_defaults_to_the_offline_bm25(self):
        args = self._tau2_args("banking_knowledge")
        self.assertEqual(args[args.index("--retrieval-config") + 1], "bm25")

    def test_a_named_config_wins_and_other_domains_get_none(self):
        args = self._tau2_args("banking_knowledge", "--retrieval-config", "bm25_grep")
        self.assertEqual(args[args.index("--retrieval-config") + 1], "bm25_grep")
        self.assertNotIn("--retrieval-config", self._tau2_args("retail"))


class LlmArgsTest(unittest.TestCase):
    """The litellm args the runner hands tau2 for the agent and the simulated user (stub `tau2` records them)."""

    SECRET = "sk-test-secret-1234"

    def _llm_args(self, *extra, secret=SECRET):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d, "upstream")
            (root / ".venv/bin").mkdir(parents=True)
            (root / "data").mkdir()
            (root / "data/x").write_text("")
            stub = root / ".venv/bin/tau2"
            stub.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > {d}/args\nexit 0\n')
            stub.chmod(0o755)
            ref = Path(d, "ref")
            ref.write_text("v2.0.0 " + "a" * 40 + "\n")
            subprocess.run(
                ["bash", "scripts/custom/run_tau2_bench.sh", "--domain", "retail", "--model", "m", "--api-base-url", "http://x/v1",
                 "--workdir", f"{d}/work", *extra],
                cwd=REPO, capture_output=True, text=True, timeout=60,
                env={"PATH": "/usr/bin:/bin", "HOME": d, "EVAL_PREBUILT_ENV": "1", "TARGET_API_KEY": secret,
                     "TAU2_UPSTREAM_DIR": str(root), "TAU2_UPSTREAM_REF_FILE": str(ref)},
            )
            lines = Path(d, "args").read_text().splitlines()
            return lines, json.loads(lines[lines.index("--agent-llm-args") + 1]), json.loads(lines[lines.index("--user-llm-args") + 1])

    def test_the_sampling_profile_is_in_the_agents_args_and_without_it_nothing_is_sent(self):
        _, agent, _ = self._llm_args("--sampling-profile", "thinking")
        self.assertEqual((agent["temperature"], agent["top_p"]), (0.6, 0.95))
        _, agent, user = self._llm_args()
        self.assertNotIn("temperature", agent)
        self.assertNotIn("temperature", user)

    def test_the_key_is_named_by_its_environment_variable_never_passed(self):
        lines, agent, user = self._llm_args()
        self.assertNotIn(self.SECRET, "\n".join(lines))
        self.assertEqual(agent["api_key"], "os.environ/TARGET_API_KEY")
        self.assertEqual(user["api_key"], "os.environ/TARGET_API_KEY")

    def test_the_default_user_reasons_at_low_effort_with_a_capped_reply_and_the_agent_is_untouched(self):
        _, agent, user = self._llm_args("--reasoning-effort", "high")
        self.assertEqual(agent["reasoning_effort"], "high")
        self.assertNotIn("max_tokens", agent)
        self.assertEqual(user["reasoning_effort"], "low")
        self.assertEqual(user["max_tokens"], 4096)

    def test_user_llm_args_override_the_users_effort_and_cap(self):
        _, _, user = self._llm_args("--user-llm-args", '{"reasoning_effort": "medium", "max_tokens": 100}')
        self.assertEqual((user["reasoning_effort"], user["max_tokens"]), ("medium", 100))

    def test_another_user_model_gets_none_of_it(self):
        _, _, user = self._llm_args("--user-llm", "gpt-4.1")
        self.assertEqual(user, {})

    def test_a_user_on_its_own_endpoint_keeps_the_pins_and_names_its_own_key_not_the_value(self):
        _, _, user = self._llm_args("--user-llm", "openai/RCP/openai/gpt-oss-120b", "--user-api-base-url", "https://u.example/")
        self.assertEqual((user["api_base"], user["api_key"]), ("https://u.example/v1", "os.environ/USER_API_KEY"))
        self.assertEqual((user["reasoning_effort"], user["max_tokens"]), ("low", 4096))

    def test_a_user_endpoint_without_another_user_model_stops_the_run(self):
        with self.assertRaises(Exception):
            self._llm_args("--user-api-base-url", "https://u.example/v1")


class JudgeTest(unittest.TestCase):
    """The NL-assertion judge is the model under test: the runner's environment for tau2 and the sitecustomize shim."""

    SHIM = REPO / "scripts/custom/tau2_judge"

    def _tau2_env(self, *extra, env_extra=None, base=("--api-base-url", "http://x/v1")):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d, "upstream")
            (root / ".venv/bin").mkdir(parents=True)
            (root / "data").mkdir()
            (root / "data/x").write_text("")
            stub = root / ".venv/bin/tau2"
            stub.write_text(f'#!/bin/sh\nenv | grep -E "^(TAU2_JUDGE|PYTHONPATH)" > {d}/env\nexit 0\n')
            stub.chmod(0o755)
            ref = Path(d, "ref")
            ref.write_text("v2.0.0 " + "a" * 40 + "\n")
            result = subprocess.run(
                ["bash", "scripts/custom/run_tau2_bench.sh", "--domain", "retail", "--model", "m", *base,
                 "--workdir", f"{d}/work", *extra],
                cwd=REPO, capture_output=True, text=True, timeout=60,
                env={"PATH": "/usr/bin:/bin", "HOME": d, "EVAL_PREBUILT_ENV": "1", "TARGET_API_KEY": "sk-test-secret-1234",
                     "TAU2_UPSTREAM_DIR": str(root), "TAU2_UPSTREAM_REF_FILE": str(ref), **(env_extra or {})},
            )
            text = Path(d, "env").read_text() if Path(d, "env").exists() else ""
            return dict(line.split("=", 1) for line in text.splitlines()), result.stdout + result.stderr

    def test_with_an_endpoint_the_judge_is_the_model_under_test_at_low_effort_without_the_key(self):
        env, out = self._tau2_env("--reasoning-effort", "high")
        self.assertEqual(env["TAU2_JUDGE_LLM"], "openai/m")
        args = json.loads(env["TAU2_JUDGE_LLM_ARGS"])
        self.assertEqual((args["reasoning_effort"], args["max_tokens"], args["api_key"]), ("low", 4096, "os.environ/TARGET_API_KEY"))
        self.assertTrue(env["PYTHONPATH"].startswith(str(self.SHIM)))
        self.assertNotIn("sk-test-secret-1234", env["TAU2_JUDGE_LLM_ARGS"] + out)
        self.assertIn("NL-assertion judge: openai/m", out)

    def test_judge_llm_args_override_the_judges_effort_and_cap(self):
        env, _ = self._tau2_env("--judge-llm-args", '{"reasoning_effort": "medium", "max_tokens": 100}')
        args = json.loads(env["TAU2_JUDGE_LLM_ARGS"])
        self.assertEqual((args["reasoning_effort"], args["max_tokens"], args["api_key"]), ("medium", 100, "os.environ/TARGET_API_KEY"))

    def _endpoint_serving(self, *model_ids, key=None):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps({"data": [{"id": i} for i in model_ids]}).encode()
                authorized = key is None or self.headers.get("Authorization") == f"Bearer {key}"
                self.send_response(200 if self.path == "/v1/models" and authorized else 404)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return ("--api-base-url", f"http://127.0.0.1:{server.server_port}/v1")

    def test_judge_llm_names_the_served_model_and_the_launcher_finds_its_hosted_id(self):
        base = self._endpoint_serving("alice/openai/gpt-oss-120b", "RCP/openai/gpt-oss-120b", "m")
        env, out = self._tau2_env("--judge-llm", "openai/openai/gpt-oss-120b", base=base, env_extra={"LOGNAME": "alice"})
        self.assertEqual(env["TAU2_JUDGE_LLM"], "openai/alice/openai/gpt-oss-120b")
        args = json.loads(env["TAU2_JUDGE_LLM_ARGS"])
        self.assertEqual((args["reasoning_effort"], args["max_tokens"], args["api_key"]), ("low", 4096, "os.environ/TARGET_API_KEY"))
        self.assertIn("NL-assertion judge: openai/alice/openai/gpt-oss-120b\n", out)

    def test_judge_llm_falls_back_to_a_shared_hosted_model_as_named(self):
        base = self._endpoint_serving("RCP/openai/gpt-oss-120b")
        env, _ = self._tau2_env("--judge-llm", "openai/RCP/openai/gpt-oss-120b", base=base, env_extra={"LOGNAME": "alice"})
        self.assertEqual(env["TAU2_JUDGE_LLM"], "openai/RCP/openai/gpt-oss-120b")

    def test_judge_llm_that_is_not_served_stops_the_run_before_tau2_starts(self):
        base = self._endpoint_serving("m")
        env, out = self._tau2_env("--judge-llm", "openai/openai/gpt-oss-120b", base=base, env_extra={"LOGNAME": "alice"})
        self.assertEqual(env, {})  # the stub tau2 never ran
        self.assertIn("is not served on", out)
        self.assertIn("launch_judge.py", out)

    def test_a_judge_on_its_own_endpoint_keeps_the_pins_and_its_own_endpoint_and_key(self):
        judge_base = self._endpoint_serving("RCP/openai/gpt-oss-120b", key="sk-judge-9876")[1]
        env, out = self._tau2_env("--reasoning-effort", "high", "--judge-llm", "openai/RCP/openai/gpt-oss-120b",
                                  "--judge-api-base-url", judge_base, env_extra={"CSCS_SERVING_API": "sk-judge-9876"})
        self.assertEqual(env["TAU2_JUDGE_LLM"], "openai/RCP/openai/gpt-oss-120b")
        args = json.loads(env["TAU2_JUDGE_LLM_ARGS"])
        self.assertEqual((args["api_base"], args["api_key"]), (judge_base, "os.environ/JUDGE_API_KEY"))
        self.assertEqual((args["reasoning_effort"], args["max_tokens"]), ("low", 4096))
        self.assertNotIn("sk-judge-9876", env["TAU2_JUDGE_LLM_ARGS"] + out)
        self.assertNotIn("sk-test-secret-1234", env["TAU2_JUDGE_LLM_ARGS"] + out)
        self.assertIn(f"(through {judge_base})", out)

    def test_a_judge_endpoint_without_a_judge_model_stops_the_run(self):
        env, out = self._tau2_env("--judge-api-base-url", "http://127.0.0.1:1/v1")
        self.assertEqual(env, {})
        self.assertIn("--judge-api-base-url needs a --judge-llm", out)

    def test_a_judge_missing_on_its_own_endpoint_stops_the_run_naming_that_endpoint(self):
        judge_base = self._endpoint_serving("m")[1]
        env, out = self._tau2_env("--judge-llm", "openai/openai/gpt-oss-120b", "--judge-api-base-url", judge_base)
        self.assertEqual(env, {})
        self.assertIn(f"is not served on {judge_base}", out)

    def test_without_an_endpoint_the_judge_stays_upstreams(self):
        env, out = self._tau2_env("--user-llm", "gpt-4.1", base=())
        self.assertNotIn("TAU2_JUDGE_LLM", env)
        self.assertIn("upstream's own", out)

    def test_the_shim_repoints_the_judge_and_never_scores_on_a_missing_judgement(self):
        with tempfile.TemporaryDirectory() as d:
            pkg = Path(d, "tau2/evaluator")
            pkg.mkdir(parents=True)
            Path(d, "tau2/__init__.py").write_text("")
            (pkg / "__init__.py").write_text("")
            (pkg / "evaluator_nl_assertions.py").write_text(
                'import json, types\n'
                'DEFAULT_LLM_NL_ASSERTIONS = "gpt"\nDEFAULT_LLM_NL_ASSERTIONS_ARGS = {"x": 1}\n'
                'TWO = \'{"results": [{"met": true}, {"met": false}]}\'\n'
                'REPLIES = ["```json\\n" + TWO + "\\n```", TWO,'                      # fenced, then clean
                ' "{bad", \'{"results": []}\', None, TWO,'                             # bad json, empty list, None: retried
                '] + ["x"] * 8\n'  # never usable: eight unparsable replies, so it raises
                'def generate(**kwargs):\n    return types.SimpleNamespace(content=REPLIES.pop(0))\n'
                'class NLAssertionsEvaluator:\n'
                '    @classmethod\n'
                '    def evaluate_nl_assertions(cls, trajectory, nl_assertions):\n'
                '        return json.loads(generate().content)["results"]\n')
            code = ("import sys\n"
                    "m = sys.modules.get('tau2.evaluator.evaluator_nl_assertions')\n"
                    "if not m: print('untouched'); raise SystemExit\n"
                    "print(m.DEFAULT_LLM_NL_ASSERTIONS, m.DEFAULT_LLM_NL_ASSERTIONS_ARGS)\n"
                    "ev = m.NLAssertionsEvaluator\n"
                    "for _ in range(4):\n"
                    "    try: print(len(ev.evaluate_nl_assertions([], ['a', 'b'])))\n"
                    "    except ValueError as e: print('raised', e)\n")
            run = lambda env: subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                                             env={"PATH": "/usr/bin:/bin", "PYTHONPATH": f"{self.SHIM}:{d}", **env}).stdout
            out = run({"TAU2_JUDGE_LLM": "openai/m", "TAU2_JUDGE_LLM_ARGS": '{"max_tokens": 5}'})
            lines = [line for line in out.splitlines() if not line.startswith("NL-assertion judge reply unusable")]
            self.assertEqual(lines[0], "openai/m {'max_tokens': 5}")
            self.assertEqual(lines[1:4], ["2", "2", "2"])  # a fenced reply parses; a clean one passes; {bad, an empty list and None are retried
            self.assertTrue(lines[4].startswith("raised NL-assertion judge gave no usable reply in 8 attempts"), lines[4])  # never usable
            self.assertEqual(out.count("reply unusable"), 3 + 8)
            self.assertIn("0 verdicts for 2 assertions", out)
            self.assertEqual(run({}).strip(), "untouched")


class RelativeWorkdirTest(unittest.TestCase):
    def test_a_relative_workdir_is_resolved_before_tau2_runs_from_its_checkout(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d, "upstream")
            (root / ".venv/bin").mkdir(parents=True)
            (root / "data").mkdir()
            (root / "data/x").write_text("")
            stub = root / ".venv/bin/tau2"
            stub.write_text(f'#!/bin/sh\nprintf "%s" "$TAU2_DATA_DIR" > {d}/data_dir\nexit 0\n')
            stub.chmod(0o755)
            ref = Path(d, "ref")
            ref.write_text("v2.0.0 " + "a" * 40 + "\n")
            subprocess.run(
                ["bash", "scripts/custom/run_tau2_bench.sh", "--domain", "retail", "--model", "m", "--api-base-url", "http://x/v1",
                 "--workdir", os.path.relpath(f"{d}/work", REPO)],
                cwd=REPO, capture_output=True, text=True, timeout=60,
                env={"PATH": "/usr/bin:/bin", "HOME": d, "EVAL_PREBUILT_ENV": "1",
                     "TAU2_UPSTREAM_DIR": str(root), "TAU2_UPSTREAM_REF_FILE": str(ref)},
            )
            data_dir = Path(d, "data_dir").read_text()
            self.assertTrue(os.path.isabs(data_dir), data_dir)
            self.assertEqual(os.path.realpath(data_dir), os.path.realpath(f"{d}/work/tau2-data"))


class SummaryTest(unittest.TestCase):
    """The summary of results.json, with a stub tau2 that writes one."""

    def _run(self, sims, *extra):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d, "upstream")
            (root / ".venv/bin").mkdir(parents=True)
            (root / "data").mkdir()
            (root / "data/x").write_text("")
            results = Path(d, "results.json")
            results.write_text(json.dumps({"simulations": sims}))
            stub = root / ".venv/bin/tau2"
            stub.write_text(f'#!/bin/sh\nmkdir -p "$TAU2_DATA_DIR/simulations/t" && cp {results} "$TAU2_DATA_DIR/simulations/t/results.json"\n')
            stub.chmod(0o755)
            ref = Path(d, "ref")
            ref.write_text("v2.0.0 " + "a" * 40 + "\n")
            return subprocess.run(
                ["bash", "scripts/custom/run_tau2_bench.sh", "--domain", "retail", "--model", "m", "--api-base-url", "http://x/v1",
                 "--workdir", f"{d}/work", "--save-to", "t", *extra],
                cwd=REPO, capture_output=True, text=True, timeout=60,
                env={"PATH": "/usr/bin:/bin", "HOME": d, "EVAL_PREBUILT_ENV": "1",
                     "TAU2_UPSTREAM_DIR": str(root), "TAU2_UPSTREAM_REF_FILE": str(ref)},
            )

    @staticmethod
    def _sim(task, reward, why="user_stop"):
        return {"task_id": task, "termination_reason": why, "reward_info": {"reward": reward} if reward is not None else None}

    def test_a_complete_result_exits_zero(self):
        result = self._run([self._sim("0", 1.0), self._sim("1", 0.0)], "--num-tasks", "2")
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout[result.stdout.index("{"):])
        self.assertEqual((summary["avg_reward"], summary["n_unscored"], summary["n_missing"]), (0.5, 0, 0))

    def test_a_task_without_a_reward_is_reported_and_fails_the_run(self):
        result = self._run([self._sim("0", 1.0), self._sim("2", None, "infrastructure_error")], "--num-tasks", "2")
        self.assertEqual(result.returncode, 1)
        self.assertIn('"n_unscored": 1', result.stdout)
        self.assertIn("task 2: no reward (infrastructure_error)", result.stderr)

    def test_a_simulation_missing_from_the_file_fails_the_run(self):
        result = self._run([self._sim("0", 1.0)], "--num-tasks", "3")
        self.assertEqual(result.returncode, 1)
        self.assertIn("2 of 3 simulations are not in results.json", result.stderr)


if __name__ == "__main__":
    unittest.main()
