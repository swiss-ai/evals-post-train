"""scripts/lm_eval_harness/model_args.py: the harness model and --model_args of a run, as a function."""

import ast
import os
import shlex
import subprocess
import sys
import unittest
from pathlib import Path

from scripts.lm_eval_harness import model_args

REPO_ROOT = Path(__file__).resolve().parents[1]
SBATCH = (REPO_ROOT / "scripts/lm_eval_harness/run_lm_eval.sbatch").read_text()

BASE = {'MODEL': 'org/model',
 'API_BASE_URL': 'http://host:8000',
 'API_MODEL_NAME': 'm',
 'CHAT_FORMAT': 'template',
 'API_CHAT_ENDPOINT': 'false',
 'APPLY_CHAT_TEMPLATE': 'true',
 'API_NUM_CONCURRENT': '8',
 'API_MAX_RETRIES': '3',
 'API_REQUEST_TIMEOUT': '3600',
 'API_REQUESTS_PER_MINUTE': '',
 'TOKENIZER': '/ckpt',
 'PREFILL_MODE': 'after_think',
 'ANSWER_FRAME': 'reply',
 'REASONING_EFFORT': '',
 'ENABLE_THINKING': 'False',
 'ENABLE_THINKING_EXPLICIT': '0',
 'THINK_START_TOKEN': '',
 'THINK_END_TOKEN': '',
 'AUTODETECT_THINK_TOKENS': 'False',
 'TRACK_THINKING_METRICS': '',
 'MAX_NEW_TOKENS': '32768',
 'TRUST_REMOTE_CODE': 'false',
 'MAX_LENGTH': '',
 'TASKS': 'gsm8k'}

# Expected values were taken from the shell code this module replaced (run_lm_eval.sbatch before it), run on
# the same settings: the behaviour is the shell's, not a new one.
GOLDEN = [
    ('plain',
     {},
     {'API_LM_EVAL_MODEL': 'local-completions',
      'COMMON_MODEL_ARGS': 'model=m,base_url=http://host:8000/v1/completions,num_concurrent=8,max_retries=3,timeout=3600,tokenizer_backend=huggingface,tokenizer=/ckpt,tokenized_requests=False',
      'CHAT_TEMPLATE_ARGS': '',
      'API_BASE_URL': 'http://host:8000/v1/completions',
      'API_MODEL_NAME': 'm',
      'CONTEXT_LEN': ''}),
    ('bare host with window',
     {'MAX_LENGTH': '65536'},
     {'API_LM_EVAL_MODEL': 'local-completions',
      'COMMON_MODEL_ARGS': 'model=m,base_url=http://host:8000/v1/completions,num_concurrent=8,max_retries=3,timeout=3600,tokenizer_backend=huggingface,tokenizer=/ckpt,tokenized_requests=False,max_length=65536,clamp_gen_toks=True',
      'CHAT_TEMPLATE_ARGS': '',
      'API_BASE_URL': 'http://host:8000/v1/completions',
      'API_MODEL_NAME': 'm',
      'CONTEXT_LEN': ''}),
    ('trailing slash and v1',
     {'API_BASE_URL': 'http://host:8000/v1/'},
     {'API_LM_EVAL_MODEL': 'local-completions',
      'COMMON_MODEL_ARGS': 'model=m,base_url=http://host:8000/v1/completions,num_concurrent=8,max_retries=3,timeout=3600,tokenizer_backend=huggingface,tokenizer=/ckpt,tokenized_requests=False',
      'CHAT_TEMPLATE_ARGS': '',
      'API_BASE_URL': 'http://host:8000/v1/completions',
      'API_MODEL_NAME': 'm',
      'CONTEXT_LEN': ''}),
    ('full completions url',
     {'API_BASE_URL': 'http://host:8000/v1/completions'},
     {'API_LM_EVAL_MODEL': 'local-completions',
      'COMMON_MODEL_ARGS': 'model=m,base_url=http://host:8000/v1/completions,num_concurrent=8,max_retries=3,timeout=3600,tokenizer_backend=huggingface,tokenizer=/ckpt,tokenized_requests=False',
      'CHAT_TEMPLATE_ARGS': '',
      'API_BASE_URL': 'http://host:8000/v1/completions',
      'API_MODEL_NAME': 'm',
      'CONTEXT_LEN': ''}),
    ('rate limit',
     {'API_REQUESTS_PER_MINUTE': '30'},
     {'API_LM_EVAL_MODEL': 'local-completions',
      'COMMON_MODEL_ARGS': 'model=m,base_url=http://host:8000/v1/completions,num_concurrent=8,max_retries=3,timeout=3600,requests_per_minute=30,tokenizer_backend=huggingface,tokenizer=/ckpt,tokenized_requests=False',
      'CHAT_TEMPLATE_ARGS': '',
      'API_BASE_URL': 'http://host:8000/v1/completions',
      'API_MODEL_NAME': 'm',
      'CONTEXT_LEN': ''}),
    ('thinking on the template path',
     {'ENABLE_THINKING': 'True',
 'ENABLE_THINKING_EXPLICIT': '1',
 'THINK_START_TOKEN': '<think>',
 'THINK_END_TOKEN': '</think>',
 'TRACK_THINKING_METRICS': 'True',
 'AUTODETECT_THINK_TOKENS': 'True'},
     {'API_LM_EVAL_MODEL': 'local-completions',
      'COMMON_MODEL_ARGS': 'model=m,base_url=http://host:8000/v1/completions,num_concurrent=8,max_retries=3,timeout=3600,tokenizer_backend=huggingface,tokenizer=/ckpt,tokenized_requests=False,max_gen_toks=32768,think_end_token=</think>,think_start_token=<think>,autodetect_think_tokens=True,track_thinking_metrics=True,enable_thinking=True',
      'CHAT_TEMPLATE_ARGS': '',
      'API_BASE_URL': 'http://host:8000/v1/completions',
      'API_MODEL_NAME': 'm',
      'CONTEXT_LEN': ''}),
    ('reasoning effort',
     {'REASONING_EFFORT': 'high', 'MAX_LENGTH': '65536'},
     {'API_LM_EVAL_MODEL': 'local-completions',
      'COMMON_MODEL_ARGS': 'model=m,base_url=http://host:8000/v1/completions,num_concurrent=8,max_retries=3,timeout=3600,tokenizer_backend=huggingface,tokenizer=/ckpt,tokenized_requests=False,max_gen_toks=32768,max_length=65536,clamp_gen_toks=True',
      'CHAT_TEMPLATE_ARGS': ' \'chat_template_args={\\"reasoning_effort\\":\\"high\\"}\'',
      'API_BASE_URL': 'http://host:8000/v1/completions',
      'API_MODEL_NAME': 'm',
      'CONTEXT_LEN': ''}),
    ('chat completions',
     {'API_CHAT_ENDPOINT': 'true',
 'MAX_LENGTH': '65536',
 'ENABLE_THINKING': 'True',
 'ENABLE_THINKING_EXPLICIT': '1',
 'THINK_END_TOKEN': '</think>'},
     {'API_LM_EVAL_MODEL': 'local-chat-completions',
      'COMMON_MODEL_ARGS': 'model=m,base_url=http://host:8000/v1/chat/completions,num_concurrent=8,max_retries=3,timeout=3600,tokenized_requests=False,max_gen_toks=32768,think_end_token=</think>,enable_thinking=True,max_length=65536,clamp_gen_toks=True,tokenizer_backend=huggingface,tokenizer=/ckpt',
      'CHAT_TEMPLATE_ARGS': '',
      'API_BASE_URL': 'http://host:8000/v1/chat/completions',
      'API_MODEL_NAME': 'm',
      'CONTEXT_LEN': ''}),
    ('chat completions without a window',
     {'API_CHAT_ENDPOINT': 'true'},
     {'API_LM_EVAL_MODEL': 'local-chat-completions',
      'COMMON_MODEL_ARGS': 'model=m,base_url=http://host:8000/v1/chat/completions,num_concurrent=8,max_retries=3,timeout=3600,tokenized_requests=False',
      'CHAT_TEMPLATE_ARGS': '',
      'API_BASE_URL': 'http://host:8000/v1/chat/completions',
      'API_MODEL_NAME': 'm',
      'CONTEXT_LEN': ''}),
    ('apertus2',
     {'CHAT_FORMAT': 'apertus2', 'REASONING_EFFORT': 'medium', 'MAX_LENGTH': '65536'},
     {'API_LM_EVAL_MODEL': 'apertus2-completions',
      'COMMON_MODEL_ARGS': 'model=m,base_url=http://host:8000/v1/completions,num_concurrent=8,max_retries=3,timeout=3600,tokenizer=/ckpt,prefill=after_think,reasoning_effort=medium,max_gen_toks=32768,max_model_len=65536',
      'CHAT_TEMPLATE_ARGS': '',
      'API_BASE_URL': 'http://host:8000/v1/completions',
      'API_MODEL_NAME': 'm',
      'CONTEXT_LEN': ''}),
    ('apertus2 claim frame',
     {'CHAT_FORMAT': 'apertus2', 'REASONING_EFFORT': 'medium', 'MAX_LENGTH': '65536', 'ANSWER_FRAME': 'claim'},
     {'API_LM_EVAL_MODEL': 'apertus2-completions',
      'COMMON_MODEL_ARGS': 'model=m,base_url=http://host:8000/v1/completions,num_concurrent=8,max_retries=3,timeout=3600,tokenizer=/ckpt,prefill=after_think,reasoning_effort=medium,max_gen_toks=32768,answer_frame=claim,max_model_len=65536',
      'CHAT_TEMPLATE_ARGS': '',
      'API_BASE_URL': 'http://host:8000/v1/completions',
      'API_MODEL_NAME': 'm',
      'CONTEXT_LEN': ''}),
    ('apertus2 on a ruler list',
     {'CHAT_FORMAT': 'apertus2', 'REASONING_EFFORT': 'medium', 'MAX_LENGTH': '65536', 'TASKS': 'ruler_8192'},
     {'API_LM_EVAL_MODEL': 'apertus2-completions',
      'COMMON_MODEL_ARGS': 'model=m,base_url=http://host:8000/v1/completions,num_concurrent=8,max_retries=3,timeout=3600,tokenizer=/ckpt,prefill=after_think,reasoning_effort=medium,max_gen_toks=32768,max_model_len=65536',
      'CHAT_TEMPLATE_ARGS': '',
      'API_BASE_URL': 'http://host:8000/v1/completions',
      'API_MODEL_NAME': 'm',
      'CONTEXT_LEN': '8192'}),
    ('trust remote code',
     {'TRUST_REMOTE_CODE': 'true'},
     {'API_LM_EVAL_MODEL': 'local-completions',
      'COMMON_MODEL_ARGS': 'model=m,base_url=http://host:8000/v1/completions,num_concurrent=8,max_retries=3,timeout=3600,tokenizer_backend=huggingface,tokenizer=/ckpt,tokenized_requests=False,trust_remote_code=True',
      'CHAT_TEMPLATE_ARGS': '',
      'API_BASE_URL': 'http://host:8000/v1/completions',
      'API_MODEL_NAME': 'm',
      'CONTEXT_LEN': ''}),
    ('ruler list',
     {'TASKS': 'ruler_8192,gsm8k', 'MAX_LENGTH': '65536'},
     {'API_LM_EVAL_MODEL': 'local-completions',
      'COMMON_MODEL_ARGS': 'model=m,base_url=http://host:8000/v1/completions,num_concurrent=8,max_retries=3,timeout=3600,tokenizer_backend=huggingface,tokenizer=/ckpt,tokenized_requests=False,max_length=8792,clamp_gen_toks=True',
      'CHAT_TEMPLATE_ARGS': '',
      'API_BASE_URL': 'http://host:8000/v1/completions',
      'API_MODEL_NAME': 'm',
      'CONTEXT_LEN': '8192'}),
    ('ruler path form',
     {'TASKS': 'niah/16384'},
     {'API_LM_EVAL_MODEL': 'local-completions',
      'COMMON_MODEL_ARGS': 'model=m,base_url=http://host:8000/v1/completions,num_concurrent=8,max_retries=3,timeout=3600,tokenizer_backend=huggingface,tokenizer=/ckpt,tokenized_requests=False',
      'CHAT_TEMPLATE_ARGS': '',
      'API_BASE_URL': 'http://host:8000/v1/completions',
      'API_MODEL_NAME': 'm',
      'CONTEXT_LEN': ''}),
    ('served name defaults to the model',
     {'API_MODEL_NAME': ''},
     {'API_LM_EVAL_MODEL': 'local-completions',
      'COMMON_MODEL_ARGS': 'model=org/model,base_url=http://host:8000/v1/completions,num_concurrent=8,max_retries=3,timeout=3600,tokenizer_backend=huggingface,tokenizer=/ckpt,tokenized_requests=False',
      'CHAT_TEMPLATE_ARGS': '',
      'API_BASE_URL': 'http://host:8000/v1/completions',
      'API_MODEL_NAME': 'org/model',
      'CONTEXT_LEN': ''}),
]


def build(**override):
    return model_args.build({**BASE, **override})


class GoldenTests(unittest.TestCase):
    def test_each_case_matches_what_the_shell_built(self):
        for name, override, expected in GOLDEN:
            with self.subTest(name):
                built = build(**override)
                self.assertEqual({key: built[key] for key in expected}, expected)


class BehaviourTests(unittest.TestCase):
    def test_chat_completions_clamp_has_a_tokenizer_to_count_with(self):
        # clamp_gen_toks counts the prompt with the client tokenizer; chat-completions used to have none,
        # so nothing was clamped and a 32768 thinking budget got a 400 on a 16k window.
        chat = build(API_CHAT_ENDPOINT="true", MAX_LENGTH="65536")
        self.assertEqual(chat["API_LM_EVAL_MODEL"], "local-chat-completions")
        self.assertIn(",max_length=65536,clamp_gen_toks=True,tokenizer_backend=huggingface,tokenizer=/ckpt",
                      chat["COMMON_MODEL_ARGS"])
        # The server still renders the template; with no window there is no clamp and no tokenizer.
        self.assertIn(",tokenized_requests=False", chat["COMMON_MODEL_ARGS"])
        self.assertNotIn("tokenizer_backend", build(API_CHAT_ENDPOINT="true")["COMMON_MODEL_ARGS"])

    def test_reasoning_effort_is_its_own_model_args_token(self):
        # The harness refuses a first --model_args token containing '{', and argparse gives
        # `--model_args='...'` exactly one value: the dict must be a second, space-separated token. The
        # harness splits it with shlex (POSIX) and reads the value with ast.literal_eval
        # (lm_eval/_cli/utils.py key_val_to_dict), so its quotes must survive shlex.
        token = build(REASONING_EFFORT="high")["CHAT_TEMPLATE_ARGS"]
        self.assertTrue(token.startswith(" '"))
        script = 'eval "set -- \'a=1\'$1"; printf "%s" "$2"'
        second = subprocess.run(["bash", "-c", script, "bash", token], capture_output=True, text=True,
                                check=True).stdout
        lexer = shlex.shlex(second, posix=True)
        lexer.whitespace, lexer.whitespace_split, lexer.commenters = ",", True, ""
        key, value = next(iter(lexer)).split("=", 1)
        self.assertEqual((key, ast.literal_eval(value)), ("chat_template_args", {"reasoning_effort": "high"}))
        self.assertIn("--model_args '$COMMON_MODEL_ARGS'$CHAT_TEMPLATE_ARGS ", SBATCH)

    def test_thinking_arguments_reach_the_template_paths_and_raise_the_budget(self):
        thinking = {"ENABLE_THINKING": "True", "ENABLE_THINKING_EXPLICIT": "1", "THINK_END_TOKEN": "</think>"}
        for endpoint in ("false", "true"):
            args = build(API_CHAT_ENDPOINT=endpoint, **thinking)["COMMON_MODEL_ARGS"]
            self.assertIn(",max_gen_toks=32768,think_end_token=</think>,enable_thinking=True", args)
        # A reasoning effort alone also gets the budget; the switch is forwarded only when set explicitly.
        effort = build(REASONING_EFFORT="low")["COMMON_MODEL_ARGS"]
        self.assertIn(",max_gen_toks=32768", effort)
        self.assertNotIn("enable_thinking", effort)

    def test_apertus2_reads_the_effort_itself_and_keeps_its_own_context(self):
        built = build(CHAT_FORMAT="apertus2", REASONING_EFFORT="medium", MAX_LENGTH="65536")
        self.assertEqual(built["API_LM_EVAL_MODEL"], "apertus2-completions")
        self.assertEqual(built["CHAT_TEMPLATE_ARGS"], "")  # no template to pass arguments to
        self.assertIn(",reasoning_effort=medium,max_gen_toks=32768,max_model_len=65536", built["COMMON_MODEL_ARGS"])
        self.assertNotIn("answer_frame", built["COMMON_MODEL_ARGS"])
        # ruler's detected length bounds the template path's context, not the server's.
        ruler = build(CHAT_FORMAT="apertus2", REASONING_EFFORT="medium", MAX_LENGTH="65536", TASKS="ruler_8192")
        self.assertTrue(ruler["COMMON_MODEL_ARGS"].endswith(",max_model_len=65536"))
        self.assertEqual(ruler["CONTEXT_LEN"], "8192")

    def test_a_ruler_list_names_its_context_with_headroom(self):
        self.assertEqual(model_args.ruler_context("gsm8k"), "")
        self.assertEqual(model_args.ruler_context("ruler"), "")
        self.assertEqual(model_args.ruler_context("ruler_8192"), "8192")
        self.assertEqual(model_args.ruler_context("ruler-4096,gsm8k"), "4096")
        self.assertEqual(model_args.ruler_context("niah/16384,ruler"), "16384")
        self.assertIn(",max_length=8792,", build(TASKS="ruler_8192")["COMMON_MODEL_ARGS"])

    def test_settings_that_cannot_work_are_refused(self):
        for override, message in (
            ({"REASONING_EFFORT": "very high"}, "REASONING_EFFORT must be a level"),
            ({"API_CHAT_ENDPOINT": "true", "APPLY_CHAT_TEMPLATE": "false"}, "requires the chat template"),
            ({"API_BASE_URL": ""}, "API_BASE_URL must be set"),
        ):
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                build(**override)


class CommandLineTests(unittest.TestCase):
    def run_cli(self, **override):
        env = {"PATH": os.environ["PATH"], **BASE, **override}
        return subprocess.run([sys.executable, "-m", "scripts.lm_eval_harness.model_args"], cwd=REPO_ROOT, env=env,
                              capture_output=True, text=True)

    def test_prints_assignments_the_shell_can_eval_and_notes_to_stderr(self):
        done = self.run_cli(REASONING_EFFORT="high", MAX_LENGTH="65536")
        self.assertEqual(done.returncode, 0, done.stderr)
        script = 'eval "$1"; printf "%s\\n" "$API_LM_EVAL_MODEL" "$COMMON_MODEL_ARGS" "$CHAT_TEMPLATE_ARGS"'
        got = subprocess.run(["bash", "-c", script, "bash", done.stdout], capture_output=True, text=True,
                             check=True).stdout.split("\n")
        built = build(REASONING_EFFORT="high", MAX_LENGTH="65536")
        self.assertEqual(got[:3], [built["API_LM_EVAL_MODEL"], built["COMMON_MODEL_ARGS"], built["CHAT_TEMPLATE_ARGS"]])
        self.assertIn("local-completions against http://host:8000/v1/completions", done.stderr)
        self.assertIn("Using context limit: 65536", done.stderr)

    def test_a_refused_setting_exits_non_zero_with_its_message(self):
        done = self.run_cli(API_BASE_URL="")
        self.assertEqual(done.returncode, 1)
        self.assertEqual(done.stdout, "")
        self.assertIn("API_BASE_URL must be set", done.stderr)

    def test_runs_on_the_old_python_the_job_may_have(self):
        # The job's own python3 outside the container can be 3.6: no newer syntax.
        ast.parse(Path(model_args.__file__).read_text(), feature_version=(3, 6))


class JobContractTests(unittest.TestCase):
    def test_the_job_exports_exactly_the_inputs_the_module_reads(self):
        line = next(l for l in SBATCH.splitlines() if l.startswith("export MODEL "))
        self.assertEqual(tuple(line.split()[1:]), model_args.INPUTS)

    def test_the_job_captures_the_output_before_eval_so_a_failure_stops_it(self):
        # `eval "$(cmd)"` returns eval's status, not cmd's: a refused setting would run on with empty arguments.
        self.assertIn('MODEL_ARGS_OUT=$(python3 -m scripts.lm_eval_harness.model_args) || die', SBATCH)
        self.assertNotIn('eval "$(python3 -m scripts.lm_eval_harness.model_args', SBATCH)
        self.assertIn('eval "$MODEL_ARGS_OUT"', SBATCH)


if __name__ == "__main__":
    unittest.main()
