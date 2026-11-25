"""scripts/lm_eval_harness/bos_pass.py: the BOS settings of one lm-eval command, as a function."""

import itertools
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.lm_eval_harness import bos_pass, job_settings

REPO_ROOT = Path(__file__).resolve().parents[1]
SBATCH = (REPO_ROOT / "scripts/lm_eval_harness/run_lm_eval.sbatch").read_text()

NOT_ADDED = {"bos_token": "<s>", "add_bos_token": False}  # e.g. the Apertus 2 instruct tokenizer
ADDED = {"bos_token": "<s>", "add_bos_token": True}  # a tokenizer that adds it itself


def resolve(tokenizer_config=None, template=True, warn=None, **override):
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "tokenizer_config.json").write_text(json.dumps(tokenizer_config or {}))
        settings = {"BOS_REQUESTED": "auto", "TOKENIZER": tmp, "API_LM_EVAL_MODEL": "local-completions",
                    "API_BASE_URL": "http://x/v1/completions", "API_MODEL_NAME": "m", **override}
        resolved = bos_pass.resolve(settings, template, warn=warn)
    resolved["PASS_PROBE"] = resolved["PASS_PROBE"].replace(tmp, "<tokenizer>")
    return resolved


class BosTests(unittest.TestCase):
    def test_bos_is_resolved_for_each_command_by_whether_it_applies_the_chat_template(self):
        # BOS=auto: the config rule applies to a command without the chat template only
        self.assertEqual(resolve(NOT_ADDED, template=False)["PASS_BOS"], "true")
        self.assertEqual(resolve(NOT_ADDED, template=True)["PASS_BOS"], "false")
        self.assertEqual(resolve(ADDED, template=False)["PASS_BOS"], "false")
        self.assertEqual(resolve(ADDED, template=True)["PASS_BOS"], "false")
        # --bos / --no-bos decide every command
        for template in (True, False):
            self.assertEqual(resolve(ADDED, template, BOS_REQUESTED="true")["PASS_BOS"], "true")
            self.assertEqual(resolve(NOT_ADDED, template, BOS_REQUESTED="false")["PASS_BOS"], "false")

    def test_the_bos_reaches_the_model_only_on_the_completions_endpoint(self):
        on = resolve(NOT_ADDED, template=False)
        self.assertEqual(on["PASS_MODEL_ARGS"], ",add_bos_token=True")
        self.assertIn("Adding BOS token.", on["PASS_NOTES"])
        self.assertIn("BOS=auto: prepending the BOS token", on["PASS_NOTES"])
        for model in ("local-chat-completions", "apertus2-completions"):
            self.assertEqual(resolve(NOT_ADDED, template=False, API_LM_EVAL_MODEL=model)["PASS_MODEL_ARGS"], "")

    def test_a_failing_bos_helper_is_reported_and_does_not_kill_the_job(self):
        def broken(tokenizer, chat_template):
            raise SyntaxError("future feature annotations is not defined")

        heard = []
        resolved = bos_pass.resolve({"BOS_REQUESTED": "auto", "TOKENIZER": "t", "API_LM_EVAL_MODEL": "x"}, False,
                                    needs_bos=broken, warn=heard.append)
        self.assertEqual(resolved["PASS_BOS"], "false")
        self.assertTrue(any("SyntaxError" in line for line in heard))
        self.assertTrue(any("BOS=auto could not be resolved" in line for line in heard))

    def test_bos_by_client_applies_to_commands_with_a_chat_template(self):
        by_client = lambda requested, template, **kw: resolve(
            template=template, BOS_BY_CLIENT=requested, **kw)["PASS_BOS_BY_CLIENT"]
        self.assertEqual(by_client("true", True), "true")
        self.assertEqual(by_client("false", True), "false")
        self.assertEqual(by_client("false", False), "false")
        ignored = resolve(template=False, BOS_BY_CLIENT="true")
        self.assertEqual(ignored["PASS_BOS_BY_CLIENT"], "false")
        self.assertIn("ignored for this one (no chat template)", ignored["PASS_NOTES"])
        # unset (auto): on for template commands where it can apply, silent without a template or elsewhere
        self.assertEqual(by_client("auto", True), "true")
        self.assertEqual(resolve(template=True)["PASS_BOS_BY_CLIENT"], "true")
        self.assertEqual(by_client("auto", False), "false")
        self.assertEqual(by_client("auto", True, API_LM_EVAL_MODEL="local-chat-completions"), "false")
        self.assertEqual(resolve(template=True)["PASS_MODEL_ARGS"], ",bos_by_client=True")

    def test_bos_probe_runs_on_the_completions_endpoint_with_its_flags(self):
        raw = resolve(NOT_ADDED, template=False)["PASS_PROBE"]
        self.assertIn("python -m scripts.lm_eval_harness.bos_probe --url 'http://x/v1/completions' --model 'm'", raw)
        self.assertIn("--bos 'true' --bos-requested 'auto'", raw)
        self.assertNotIn("--chat-template", raw)
        self.assertNotIn("--bos-by-client", raw)
        chat = resolve(template=True)["PASS_PROBE"]
        self.assertTrue(chat.endswith("--chat-template --bos-by-client"))
        self.assertEqual(resolve(BOS_CHECK="false")["PASS_PROBE"], "")
        self.assertEqual(resolve(API_LM_EVAL_MODEL="local-chat-completions")["PASS_PROBE"], "")

    def test_every_combination_has_a_consistent_answer(self):
        # The combinations the shell code this replaced was checked on (648: its answers were identical).
        for requested, by_client, model, template, check in itertools.product(
                ("auto", "true", "false"), ("", "auto", "true", "false"),
                ("local-completions", "local-chat-completions", "apertus2-completions"), (True, False),
                ("", "false", "true")):
            resolved = resolve(ADDED, template, BOS_REQUESTED=requested, BOS_BY_CLIENT=by_client,
                               API_LM_EVAL_MODEL=model, BOS_CHECK=check)
            self.assertIn(resolved["PASS_BOS"], ("true", "false"))
            if model != "local-completions":
                self.assertEqual((resolved["PASS_MODEL_ARGS"], resolved["PASS_PROBE"]), ("", ""))


class CommandLineTests(unittest.TestCase):
    def test_prints_assignments_the_shell_can_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "tokenizer_config.json").write_text(json.dumps(NOT_ADDED))
            env = {"PATH": os.environ["PATH"], "BOS_REQUESTED": "auto", "TOKENIZER": tmp, "PWD": str(REPO_ROOT),
                   "API_LM_EVAL_MODEL": "local-completions", "API_BASE_URL": "http://x/v1/completions",
                   "API_MODEL_NAME": "m"}
            done = subprocess.run([sys.executable, "-m", "scripts.lm_eval_harness.bos_pass", "--chat-template",
                                   "false"], cwd=REPO_ROOT, env=env, capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            script = 'eval "$1"; printf "%s\\n" "$PASS_BOS" "$PASS_MODEL_ARGS" "$PASS_NOTES"'
            got = subprocess.run(["bash", "-c", script, "bash", done.stdout], capture_output=True, text=True,
                                 check=True).stdout.split("\n")
        self.assertEqual(got[:2], ["true", ",add_bos_token=True"])
        self.assertIn("BOS=auto: prepending the BOS token", got[2])


class JobContractTests(unittest.TestCase):
    def test_the_job_exports_what_the_module_reads(self):
        exported = set()
        for line in SBATCH.splitlines():
            if line.startswith("export "):
                exported.update(line.split()[1:])
        self.assertLessEqual(set(bos_pass.INPUTS), exported)

    def test_make_cmd_captures_the_output_before_eval_so_a_failure_stops_the_job(self):
        self.assertIn('pass_out=$(python3 -m scripts.lm_eval_harness.bos_pass --chat-template "$3") \\', SBATCH)
        self.assertIn('eval "$pass_out"', SBATCH)
        self.assertNotIn('eval "$(python3 -m scripts.lm_eval_harness.bos_pass', SBATCH)

    def test_the_refusal_of_bos_by_client_where_it_cannot_apply_is_in_the_settings(self):
        base = {"MODEL": "m", "API_BASE_URL": "http://x:8000", "TOKENIZER": "/ckpt", "BOS_BY_CLIENT": "true",
                "API_NUM_CONCURRENT": "8", "API_MAX_RETRIES": "3", "API_REQUEST_TIMEOUT": "3600", "TASKS": "gsm8k"}
        self.assertEqual(job_settings.build(base)[0]["API_LM_EVAL_MODEL"], "local-completions")
        for override in ({"API_CHAT_ENDPOINT": "true", "APPLY_CHAT_TEMPLATE": "true"},
                         {"CHAT_FORMAT": "apertus2", "MAX_LENGTH": "65536", "BOS_BY_CLIENT": "true"}):
            with self.subTest(override), self.assertRaisesRegex(ValueError, "needs /v1/completions"):
                job_settings.build({**base, **override})


if __name__ == "__main__":
    unittest.main()
