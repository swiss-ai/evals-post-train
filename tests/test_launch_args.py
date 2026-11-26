"""scripts/lm_eval_harness/launch_args.py: what the launcher's flags mean together, as a function."""

import ast
import os
import subprocess
import sys
import unittest
from pathlib import Path

from scripts.lm_eval_harness import launch_args

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (REPO_ROOT / "scripts/lm_eval_harness/launch_evaluations.sh").read_text()

BASE = {"API_BASE_URL_FLAG": "http://x:8000", "API_MODEL_NAME_FLAG": "m"}


def resolve(**env):
    """The Resolution for these launcher variables, every other one empty (a stand-in for the helper runs)."""
    def helper(args, **kwargs):
        raise AssertionError("no helper expected: %s" % args)

    return launch_args.resolve({**BASE, **env}, run=helper)


def refused(**env):
    with unittest.TestCase().assertRaises(launch_args.Refused) as caught:
        resolve(**env)
    return str(caught.exception)


class ThinkingTests(unittest.TestCase):
    def test_the_umbrella_implies_the_rest(self):
        out = resolve(THINKING_UMBRELLA="true")
        self.assertEqual(out.variables["ENABLE_THINKING_OVERRIDE"], "true")
        self.assertEqual(out.variables["TRACK_THINKING_METRICS"], "true")
        self.assertEqual(out.variables["LOG_LENGTH_METRICS"], "true")
        self.assertEqual(out.variables["AUTODETECT_THINK_TOKENS"], "true")  # a close token must come from somewhere
        self.assertEqual(out.variables["CHAT_TEMPLATE_OVERRIDE"], "true")  # and the template has to be rendered
        self.assertEqual(out.exports["ENABLE_THINKING"], "true")
        self.assertEqual((out.variables["THINKING_TOUCHED"], out.variables["THINKING_METRICS_ASKED"]), ("true", "true"))
        # A close token the user named is not replaced by autodetection.
        named = resolve(THINKING_UMBRELLA="true", THINK_END_TOKEN="</think>", THINK_START_TOKEN="<think>")
        self.assertEqual(named.variables["AUTODETECT_THINK_TOKENS"], "")

    def test_enable_thinking_alone_only_switches_the_template(self):
        out = resolve(ENABLE_THINKING_OVERRIDE="true")
        self.assertEqual(out.variables["THINKING_METRICS_ASKED"], "false")  # nothing is stripped or measured
        self.assertEqual(out.variables["CHAT_TEMPLATE_OVERRIDE"], "true")
        by_type = resolve(ENABLE_THINKING_OVERRIDE="true", CHAT_TEMPLATE_OVERRIDE="by-type")
        self.assertEqual(by_type.variables["CHAT_TEMPLATE_OVERRIDE"], "by-type")  # the generative tasks keep it

    def test_nothing_is_exported_that_was_not_set(self):
        out = resolve()
        self.assertEqual(out.exports, {"API_BASE_URL": "http://x:8000", "API_MODEL_NAME": "m"})  # the flags given
        self.assertEqual(out.variables["THINKING_TOUCHED"], "false")

    def test_refused_combinations(self):
        for env, message in (
            ({"TRACK_THINKING_METRICS": "maybe"}, "--track-thinking-metrics expects 'true' or 'false'"),
            ({"THINK_END_TOKEN": "a,b"}, "THINK_END_TOKEN must not contain a comma"),
            ({"THINKING_UMBRELLA": "true", "CHAT_TEMPLATE_OVERRIDE": "false"}, "drop --no-chat-template"),
            ({"THINK_END_TOKEN": "</think>", "CHAT_TEMPLATE_OVERRIDE": "false"}, "thinking metrics require the chat template"),
            ({"ENABLE_THINKING_OVERRIDE": "false", "CHAT_TEMPLATE_OVERRIDE": "false"}, "drop --no-chat-template"),
            ({"TRACK_THINKING_METRICS": "true"}, "no reasoning close token is known"),
            ({"REASONING_EFFORT": "very high"}, "--reasoning-effort expects a level"),
            ({"REASONING_EFFORT": "high", "CHAT_TEMPLATE_OVERRIDE": "false"}, "drop --no-chat-template"),
        ):
            with self.subTest(message):
                self.assertIn(message, refused(**env))

    def test_a_missing_open_token_and_dropped_metrics_are_warned_about(self):
        heard = []
        launch_args.resolve({**BASE, "THINK_END_TOKEN": "</think>", "ENABLE_THINKING_OVERRIDE": "true",
                             "TRACK_THINKING_METRICS": "true"}, warn=heard.append)
        self.assertTrue(any("no reasoning open token" in line for line in heard))
        out = resolve(TRACK_THINKING_METRICS="false", LOG_LENGTH_METRICS="true")
        self.assertTrue(any("drops thinking_length_*" in line for line in out.warnings))

    def test_a_warning_is_heard_even_when_a_later_check_refuses_the_run(self):
        # The sampling check refuses after the warning about dropped thinking_length_* arose.
        def helper(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout="SAMPLING_TEMPERATURE=\nSAMPLING_TOP_P=0.9\n"
                                               "SAMPLING_TOP_K=\nSAMPLING_MIN_P=\nSAMPLING_REPETITION_PENALTY=\n")

        heard = []
        with self.assertRaises(launch_args.Refused):
            launch_args.resolve({**BASE, "TRACK_THINKING_METRICS": "false", "LOG_LENGTH_METRICS": "true",
                                 "SAMPLING_TOP_P_FLAG": "0.9"}, run=helper, warn=heard.append)
        self.assertTrue(any("drops thinking_length_*" in line for line in heard))


class Apertus2Tests(unittest.TestCase):
    APERTUS2 = {"CHAT_FORMAT_FLAG": "apertus2", "CUSTOM_TOKENIZER": "/ckpt", "MAX_LENGTH": "65536"}

    def test_the_format_always_reasons_and_exports_its_settings(self):
        out = resolve(**self.APERTUS2)
        self.assertEqual(out.variables["REASONING_EFFORT"], "medium")
        self.assertEqual(out.variables["LOG_LENGTH_METRICS"], "true")
        self.assertEqual(out.exports["CHAT_FORMAT"], "apertus2")
        self.assertEqual((out.exports["PREFILL_MODE"], out.exports["ANSWER_FRAME"]), ("after_think", "reply"))
        self.assertEqual(out.unset, [])
        self.assertEqual(resolve().unset, ["CHAT_FORMAT", "PREFILL_MODE", "ANSWER_FRAME"])  # the template format

    def test_refusals(self):
        for override, message in (
            ({"CUSTOM_TOKENIZER": ""}, "needs --tokenizer"),
            ({"MAX_LENGTH": ""}, "needs --tokenizer"),
            ({"BOS_FLAG": "true"}, "renders the BOS token itself"),
            ({"CHAT_TEMPLATE_OVERRIDE": "false"}, "renders every task as chat"),
            ({"CHAT_TEMPLATE_OVERRIDE": "by-type"}, "renders every task as chat"),
            ({"API_ENDPOINT_FLAG": "chat-completions"}, "renders every task as chat"),
            ({"API_CHAT_ENDPOINT": "true"}, "renders every task as chat"),
            ({"THINKING_UMBRELLA": "true"}, "always reasons"),
            ({"ENABLE_THINKING": "true"}, "always reasons"),
            ({"TRACK_THINKING_METRICS": "false"}, "always reasons"),
            ({"PREFILL_MODE_FLAG": "bad"}, "--prefill-mode must be"),
            ({"ANSWER_FRAME_FLAG": "bad"}, "--answer-frame must be"),
            ({"REASONING_EFFORT": "extreme"}, "takes --reasoning-effort low, medium or high"),
        ):
            with self.subTest(message, **override):
                self.assertIn(message, refused(**{**self.APERTUS2, **override}))
        # an explicit --api-endpoint completions wins over an ambient API_CHAT_ENDPOINT=true
        self.assertEqual(resolve(**self.APERTUS2, API_CHAT_ENDPOINT="true", API_ENDPOINT_FLAG="completions")
                         .exports["API_CHAT_ENDPOINT"], "false")

    def test_the_apertus2_options_need_the_apertus2_format(self):
        self.assertIn("--prefill-mode only applies", refused(PREFILL_MODE_FLAG="empty_think"))
        self.assertIn("--answer-frame only applies", refused(ANSWER_FRAME_FLAG="claim"))
        self.assertIn("--chat-format must be 'template' or 'apertus2'", refused(CHAT_FORMAT_FLAG="harmony"))


class EndpointTests(unittest.TestCase):
    def test_an_endpoint_and_a_served_name_are_required(self):
        self.assertIn("--api-base-url <url> is required", refused(API_BASE_URL_FLAG=""))
        self.assertIn("--api-model-name <name> is required", refused(API_MODEL_NAME_FLAG=""))
        ambient = launch_args.resolve({"API_BASE_URL": "http://x", "API_MODEL_NAME": "served"}, run=None)
        self.assertEqual(ambient.variables["MODEL_PATH"], "served")  # --model defaults from the served name
        self.assertEqual(resolve(MODEL_PATH="mine").variables["MODEL_PATH"], "mine")

    def test_the_endpoint_flags_are_checked_and_exported(self):
        self.assertIn("--api-max-retries must be a positive integer", refused(API_MAX_RETRIES_FLAG="0"))
        self.assertIn("--api-num-concurrent must be a positive integer", refused(API_NUM_CONCURRENT_FLAG="many"))
        self.assertIn("--api-endpoint must be", refused(API_ENDPOINT_FLAG="responses"))
        chat = resolve(API_ENDPOINT_FLAG="chat-completions", API_MAX_RETRIES_FLAG="20")
        self.assertEqual(chat.exports["API_CHAT_ENDPOINT"], "true")
        self.assertEqual(chat.exports["API_MAX_RETRIES"], "20")
        self.assertEqual(chat.exports["API_BASE_URL"], "http://x:8000")
        for template in ("false", "by-type"):
            self.assertIn("chat-completions needs the chat template on every task",
                          refused(API_ENDPOINT_FLAG="chat-completions", CHAT_TEMPLATE_OVERRIDE=template))

    def test_a_flag_wins_over_the_ambient_variable(self):
        out = launch_args.resolve({**BASE, "API_REQUESTS_PER_MINUTE_FLAG": "30", "API_REQUESTS_PER_MINUTE": "5",
                                   "JUDGE_MODEL_PREFIX": "", "USER": "me"})
        self.assertEqual(out.exports["API_REQUESTS_PER_MINUTE"], "30")
        self.assertEqual(out.exports["JUDGE_MODEL_PREFIX"], "me")


class RepeatsAndSamplingTests(unittest.TestCase):
    def test_repeats_are_checked_and_layered_into_a_policy_file(self):
        self.assertIn("--repeats takes a positive integer", refused(REPEATS_FLAG="x"))
        self.assertIn("--repeat-metric takes pass@K", refused(REPEAT_METRIC_FLAG="mean"))
        calls = []

        def helper(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, stdout="")

        out = launch_args.resolve({**BASE, "REPEATS_FLAG": "4", "REPEAT_METRIC_FLAG": "pass@2,4",
                                   "EVAL_REPEAT_CONFIG": "policy.json", "ENABLE_THINKING_OVERRIDE": "true"}, run=helper)
        self.assertEqual(len(calls), 1)
        self.assertIn("--policy", calls[0])
        self.assertEqual(calls[0][calls[0].index("--repeats") + 1], "4")
        self.assertEqual(calls[0][calls[0].index("--pass-at") + 1], "2,4")
        self.assertEqual(out.exports["EVAL_REPEAT_CONFIG"], calls[0][calls[0].index("--output") + 1])
        # a helper that refuses stops the launcher, having said why itself
        with self.assertRaises(launch_args.Refused) as caught:
            launch_args.resolve({**BASE, "REPEATS_FLAG": "4"}, run=lambda args, **kw: subprocess.CompletedProcess(args, 1))
        self.assertIsNone(caught.exception.args[0])

    def test_repeats_of_a_greedy_run_are_refused(self):
        # no sampling on a run that does not think; the policy helper is run first
        with self.assertRaises(launch_args.Refused) as caught:
            launch_args.resolve({**BASE, "REPEATS_FLAG": "4"},
                                run=lambda args, **kw: subprocess.CompletedProcess(args, 0, stdout=""))
        self.assertIn("identical samples", caught.exception.args[0])

    def test_sampling_flags_on_a_run_that_does_not_think_need_a_temperature(self):
        def helper(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout="SAMPLING_TEMPERATURE=\nSAMPLING_TOP_P=0.9\n"
                                               "SAMPLING_TOP_K=\nSAMPLING_MIN_P=\nSAMPLING_REPETITION_PENALTY=\n")

        with self.assertRaises(launch_args.Refused) as caught:
            launch_args.resolve({**BASE, "SAMPLING_TOP_P_FLAG": "0.9"}, run=helper)
        self.assertIn("need --temperature", caught.exception.args[0])

    def test_sampling_flags_are_exported_as_think_or_nothink_settings(self):
        def helper(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout="SAMPLING_TEMPERATURE=0.7\nSAMPLING_TOP_P=\n"
                                               "SAMPLING_TOP_K=20\nSAMPLING_MIN_P=\nSAMPLING_REPETITION_PENALTY=\n")

        plain = launch_args.resolve({**BASE, "SAMPLING_TEMPERATURE_FLAG": "0.7"}, run=helper)
        self.assertEqual((plain.exports["NOTHINK_TEMPERATURE"], plain.exports["NOTHINK_TOP_P"]), ("0.7", "none"))
        self.assertEqual(plain.exports["THINK_TOP_K"], "20")
        thinking = launch_args.resolve({**BASE, "SAMPLING_TEMPERATURE_FLAG": "0.7", "ENABLE_THINKING_OVERRIDE": "true"},
                                       run=helper)
        self.assertEqual(thinking.exports["THINK_TEMPERATURE"], "0.7")
        self.assertEqual(thinking.exports["THINK_TOP_P"], "none")  # "none" leaves a sampler out
        self.assertNotIn("NOTHINK_TEMPERATURE", thinking.exports)


class CommandLineTests(unittest.TestCase):
    def run_cli(self, **env):
        full = {"PATH": os.environ["PATH"], **{k: v for k, v in BASE.items()}, **env}
        return subprocess.run([sys.executable, "-m", "scripts.lm_eval_harness.launch_args"], cwd=REPO_ROOT, env=full,
                              capture_output=True, text=True)

    def test_prints_statements_the_shell_can_eval(self):
        done = self.run_cli(ENABLE_THINKING_OVERRIDE="true", REASONING_EFFORT="high")
        self.assertEqual(done.returncode, 0, done.stderr)
        script = 'eval "$1"; printf "%s\\n" "$CHAT_TEMPLATE_OVERRIDE" "$ENABLE_THINKING" "$REASONING_EFFORT"'
        got = subprocess.run(["bash", "-c", script, "bash", done.stdout], capture_output=True, text=True, check=True)
        self.assertEqual(got.stdout.split("\n")[:3], ["true", "true", "high"])

    def test_a_refusal_prints_its_message_and_exits_non_zero(self):
        done = self.run_cli(API_BASE_URL_FLAG="")
        self.assertEqual(done.returncode, 1)
        self.assertEqual(done.stdout, "")
        self.assertIn("Error: --api-base-url <url> is required", done.stderr)

    def test_the_module_is_plain_enough_for_the_launchers_python(self):
        ast.parse(Path(launch_args.__file__).read_text(), feature_version=(3, 6))


class LauncherContractTests(unittest.TestCase):
    def test_the_launcher_hands_over_exactly_the_flags_the_module_reads(self):
        call = LAUNCHER[LAUNCHER.index("LAUNCH_OUT=$("):LAUNCHER.index("python3 -m scripts.lm_eval_harness.launch_args")]
        handed = [token.split("=")[0] for token in call.replace("LAUNCH_OUT=$(", "").replace("\\\n", " ").split()]
        self.assertEqual(sorted(handed), sorted(launch_args.FLAGS))
        for name in launch_args.FLAGS:
            self.assertIn('%s="$%s"' % (name, name), call)
        # each is a variable the launcher's parser sets (or defaults) before the call
        before = LAUNCHER[:LAUNCHER.index("LAUNCH_OUT=$(")]
        for name in launch_args.FLAGS:
            self.assertRegex(before, r'(?m)^\s*%s=|\b%s="' % (name, name), name)

    def test_the_output_is_captured_before_eval_so_a_refusal_stops_the_launcher(self):
        self.assertIn("python3 -m scripts.lm_eval_harness.launch_args) || exit 1", LAUNCHER)
        self.assertIn('eval "$LAUNCH_OUT"', LAUNCHER)
        self.assertNotIn('eval "$(python3 -m scripts.lm_eval_harness.launch_args', LAUNCHER)

    def test_the_ambient_variables_are_not_handed_over_so_the_jobs_do_not_inherit_the_flags(self):
        for name in launch_args.AMBIENT:
            self.assertNotIn('%s="$%s" ' % (name, name), LAUNCHER[LAUNCHER.index("LAUNCH_OUT=$("):][:2500], name)

    def test_the_resolution_comes_before_anything_that_uses_it(self):
        call = LAUNCHER.index('eval "$LAUNCH_OUT"')
        for use in ("THINKING_TOUCHED\" == \"true\"", 'auto_derive_name "$MODEL_PATH"', "export APPLY_CHAT_TEMPLATE"):
            self.assertLess(call, LAUNCHER.index(use), use)


if __name__ == "__main__":
    unittest.main()
