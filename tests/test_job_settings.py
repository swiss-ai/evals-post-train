"""scripts/lm_eval_harness/job_settings.py: the settings of an lm-eval job, resolved and checked as a function."""

import ast
import os
import subprocess
import sys
import unittest
from pathlib import Path

from scripts.lm_eval_harness import job_settings

REPO_ROOT = Path(__file__).resolve().parents[1]
SBATCH = (REPO_ROOT / "scripts/lm_eval_harness/run_lm_eval.sbatch").read_text()

BASE = {'MODEL': 'org/model',
      'API_BASE_URL': 'http://host:8000',
      'API_MODEL_NAME': 'served',
      'API_NUM_CONCURRENT': '8',
      'API_MAX_RETRIES': '3',
      'API_REQUEST_TIMEOUT': '3600',
      'TASKS': 'gsm8k'}

# Expected values and notes were taken from the shell code this module replaced (run_lm_eval.sbatch before it),
# run on the same environment: the behaviour is the shell's, not a new one.
GOLDEN = [
    ('plain',
     {},
     {'TOKENIZER': 'org/model',
      'MAX_LENGTH': '',
      'MAX_NEW_TOKENS': '2048',
      'ENABLE_THINKING': 'False',
      'ENABLE_THINKING_EXPLICIT': '0',
      'AUTODETECT_THINK_TOKENS': 'False',
      'TRACK_THINKING_METRICS': '',
      'LOG_LENGTH_METRICS': 'False',
      'REASONING_EFFORT': '',
      'CHAT_FORMAT': 'template',
      'PREFILL_MODE': '',
      'ANSWER_FRAME': '',
      'GEN_KWARGS': '',
      'API_LM_EVAL_MODEL': 'local-completions',
      'COMMON_MODEL_ARGS': 'model=served,base_url=http://host:8000/v1/completions,num_concurrent=8,max_retries=3,timeout=3600,tokenizer_backend=huggingface,tokenizer=org/model,tokenized_requests=False',
      'CHAT_TEMPLATE_ARGS': '',
      'API_BASE_URL': 'http://host:8000/v1/completions',
      'API_MODEL_NAME': 'served',
      'CONTEXT_LEN': ''},
     ['TOKENIZER not set, using TOKENIZER=org/model',
      'local-completions against http://host:8000/v1/completions (model=served)']),
    ('tokenizer defaults to the model',
     {},
     {'TOKENIZER': 'org/model',
      'MAX_LENGTH': '',
      'MAX_NEW_TOKENS': '2048',
      'ENABLE_THINKING': 'False',
      'ENABLE_THINKING_EXPLICIT': '0',
      'AUTODETECT_THINK_TOKENS': 'False',
      'TRACK_THINKING_METRICS': '',
      'LOG_LENGTH_METRICS': 'False',
      'REASONING_EFFORT': '',
      'CHAT_FORMAT': 'template',
      'PREFILL_MODE': '',
      'ANSWER_FRAME': '',
      'GEN_KWARGS': '',
      'API_LM_EVAL_MODEL': 'local-completions',
      'COMMON_MODEL_ARGS': 'model=served,base_url=http://host:8000/v1/completions,num_concurrent=8,max_retries=3,timeout=3600,tokenizer_backend=huggingface,tokenizer=org/model,tokenized_requests=False',
      'CHAT_TEMPLATE_ARGS': '',
      'API_BASE_URL': 'http://host:8000/v1/completions',
      'API_MODEL_NAME': 'served',
      'CONTEXT_LEN': ''},
     ['TOKENIZER not set, using TOKENIZER=org/model',
      'local-completions against http://host:8000/v1/completions (model=served)']),
    ('catalog prefix is stripped from the tokenizer',
     {'MODEL': 'CSCS-Inference/swiss-ai/Apertus-v1.5-70B'},
     {'TOKENIZER': 'swiss-ai/Apertus-v1.5-70B',
      'MAX_LENGTH': '',
      'MAX_NEW_TOKENS': '2048',
      'ENABLE_THINKING': 'False',
      'ENABLE_THINKING_EXPLICIT': '0',
      'AUTODETECT_THINK_TOKENS': 'False',
      'TRACK_THINKING_METRICS': '',
      'LOG_LENGTH_METRICS': 'False',
      'REASONING_EFFORT': '',
      'CHAT_FORMAT': 'template',
      'PREFILL_MODE': '',
      'ANSWER_FRAME': '',
      'GEN_KWARGS': '',
      'API_LM_EVAL_MODEL': 'local-completions',
      'COMMON_MODEL_ARGS': 'model=served,base_url=http://host:8000/v1/completions,num_concurrent=8,max_retries=3,timeout=3600,tokenizer_backend=huggingface,tokenizer=swiss-ai/Apertus-v1.5-70B,tokenized_requests=False',
      'CHAT_TEMPLATE_ARGS': '',
      'API_BASE_URL': 'http://host:8000/v1/completions',
      'API_MODEL_NAME': 'served',
      'CONTEXT_LEN': ''},
     ['TOKENIZER not set, using TOKENIZER=swiss-ai/Apertus-v1.5-70B (catalog prefix stripped from '
      'MODEL=CSCS-Inference/swiss-ai/Apertus-v1.5-70B)',
      'local-completions against http://host:8000/v1/completions (model=served)']),
    ('a local path is its own tokenizer',
     {'MODEL': '/capstor/store/x/ckpt'},
     {'TOKENIZER': '/capstor/store/x/ckpt',
      'MAX_LENGTH': '',
      'MAX_NEW_TOKENS': '2048',
      'ENABLE_THINKING': 'False',
      'ENABLE_THINKING_EXPLICIT': '0',
      'AUTODETECT_THINK_TOKENS': 'False',
      'TRACK_THINKING_METRICS': '',
      'LOG_LENGTH_METRICS': 'False',
      'REASONING_EFFORT': '',
      'CHAT_FORMAT': 'template',
      'PREFILL_MODE': '',
      'ANSWER_FRAME': '',
      'GEN_KWARGS': '',
      'API_LM_EVAL_MODEL': 'local-completions',
      'COMMON_MODEL_ARGS': 'model=served,base_url=http://host:8000/v1/completions,num_concurrent=8,max_retries=3,timeout=3600,tokenizer_backend=huggingface,tokenizer=/capstor/store/x/ckpt,tokenized_requests=False',
      'CHAT_TEMPLATE_ARGS': '',
      'API_BASE_URL': 'http://host:8000/v1/completions',
      'API_MODEL_NAME': 'served',
      'CONTEXT_LEN': ''},
     ['TOKENIZER not set, using TOKENIZER=/capstor/store/x/ckpt',
      'local-completions against http://host:8000/v1/completions (model=served)']),
    ('a named tokenizer stays',
     {'TOKENIZER': '/ckpt'},
     {'TOKENIZER': '/ckpt',
      'MAX_LENGTH': '',
      'MAX_NEW_TOKENS': '2048',
      'ENABLE_THINKING': 'False',
      'ENABLE_THINKING_EXPLICIT': '0',
      'AUTODETECT_THINK_TOKENS': 'False',
      'TRACK_THINKING_METRICS': '',
      'LOG_LENGTH_METRICS': 'False',
      'REASONING_EFFORT': '',
      'CHAT_FORMAT': 'template',
      'PREFILL_MODE': '',
      'ANSWER_FRAME': '',
      'GEN_KWARGS': '',
      'API_LM_EVAL_MODEL': 'local-completions',
      'COMMON_MODEL_ARGS': 'model=served,base_url=http://host:8000/v1/completions,num_concurrent=8,max_retries=3,timeout=3600,tokenizer_backend=huggingface,tokenizer=/ckpt,tokenized_requests=False',
      'CHAT_TEMPLATE_ARGS': '',
      'API_BASE_URL': 'http://host:8000/v1/completions',
      'API_MODEL_NAME': 'served',
      'CONTEXT_LEN': ''},
     ['local-completions against http://host:8000/v1/completions (model=served)']),
    ('thinking raises the budget to the window',
     {'TOKENIZER': '/ckpt', 'ENABLE_THINKING': 'true', 'MAX_LENGTH': '262144'},
     {'TOKENIZER': '/ckpt',
      'MAX_LENGTH': '262144',
      'MAX_NEW_TOKENS': '262144',
      'ENABLE_THINKING': 'True',
      'ENABLE_THINKING_EXPLICIT': '1',
      'AUTODETECT_THINK_TOKENS': 'False',
      'TRACK_THINKING_METRICS': '',
      'LOG_LENGTH_METRICS': 'False',
      'REASONING_EFFORT': '',
      'CHAT_FORMAT': 'template',
      'PREFILL_MODE': '',
      'ANSWER_FRAME': '',
      'GEN_KWARGS': 'do_sample=true,temperature=0.6,top_p=0.95,max_gen_toks=262144',
      'API_LM_EVAL_MODEL': 'local-completions',
      'COMMON_MODEL_ARGS': 'model=served,base_url=http://host:8000/v1/completions,num_concurrent=8,max_retries=3,timeout=3600,tokenizer_backend=huggingface,tokenizer=/ckpt,tokenized_requests=False,max_gen_toks=262144,enable_thinking=True,max_length=262144,clamp_gen_toks=True',
      'CHAT_TEMPLATE_ARGS': '',
      'API_BASE_URL': 'http://host:8000/v1/completions',
      'API_MODEL_NAME': 'served',
      'CONTEXT_LEN': ''},
     ['ENABLE_THINKING=True or REASONING_EFFORT set: raising default MAX_NEW_TOKENS to 262144',
      'local-completions against http://host:8000/v1/completions (model=served)',
      'Using context limit: 262144',
      'Thinking sampling: do_sample=true temperature=0.6 top_p=0.95 top_k=<unset> min_p=<unset> '
      'repetition_penalty=<unset> max_gen_toks=262144',
      'Thinking/reasoning-effort run: max_gen_toks=262144 for every task (replaces task-local budgets)']),
    ('thinking without a window keeps 32768',
     {'TOKENIZER': '/ckpt', 'ENABLE_THINKING': 'true'},
     {'TOKENIZER': '/ckpt',
      'MAX_LENGTH': '',
      'MAX_NEW_TOKENS': '32768',
      'ENABLE_THINKING': 'True',
      'ENABLE_THINKING_EXPLICIT': '1',
      'AUTODETECT_THINK_TOKENS': 'False',
      'TRACK_THINKING_METRICS': '',
      'LOG_LENGTH_METRICS': 'False',
      'REASONING_EFFORT': '',
      'CHAT_FORMAT': 'template',
      'PREFILL_MODE': '',
      'ANSWER_FRAME': '',
      'GEN_KWARGS': 'do_sample=true,temperature=0.6,top_p=0.95,max_gen_toks=32768',
      'API_LM_EVAL_MODEL': 'local-completions',
      'COMMON_MODEL_ARGS': 'model=served,base_url=http://host:8000/v1/completions,num_concurrent=8,max_retries=3,timeout=3600,tokenizer_backend=huggingface,tokenizer=/ckpt,tokenized_requests=False,max_gen_toks=32768,enable_thinking=True',
      'CHAT_TEMPLATE_ARGS': '',
      'API_BASE_URL': 'http://host:8000/v1/completions',
      'API_MODEL_NAME': 'served',
      'CONTEXT_LEN': ''},
     ['ENABLE_THINKING=True or REASONING_EFFORT set: raising default MAX_NEW_TOKENS to 32768',
      'local-completions against http://host:8000/v1/completions (model=served)',
      'Thinking sampling: do_sample=true temperature=0.6 top_p=0.95 top_k=<unset> min_p=<unset> '
      'repetition_penalty=<unset> max_gen_toks=32768',
      'Thinking/reasoning-effort run: max_gen_toks=32768 for every task (replaces task-local budgets)']),
    ('an explicit budget wins',
     {'TOKENIZER': '/ckpt', 'ENABLE_THINKING': 'true', 'MAX_NEW_TOKENS': '8192', 'MAX_LENGTH': '262144'},
     {'TOKENIZER': '/ckpt',
      'MAX_LENGTH': '262144',
      'MAX_NEW_TOKENS': '8192',
      'ENABLE_THINKING': 'True',
      'ENABLE_THINKING_EXPLICIT': '1',
      'AUTODETECT_THINK_TOKENS': 'False',
      'TRACK_THINKING_METRICS': '',
      'LOG_LENGTH_METRICS': 'False',
      'REASONING_EFFORT': '',
      'CHAT_FORMAT': 'template',
      'PREFILL_MODE': '',
      'ANSWER_FRAME': '',
      'GEN_KWARGS': 'do_sample=true,temperature=0.6,top_p=0.95,max_gen_toks=8192',
      'API_LM_EVAL_MODEL': 'local-completions',
      'COMMON_MODEL_ARGS': 'model=served,base_url=http://host:8000/v1/completions,num_concurrent=8,max_retries=3,timeout=3600,tokenizer_backend=huggingface,tokenizer=/ckpt,tokenized_requests=False,max_gen_toks=8192,enable_thinking=True,max_length=262144,clamp_gen_toks=True',
      'CHAT_TEMPLATE_ARGS': '',
      'API_BASE_URL': 'http://host:8000/v1/completions',
      'API_MODEL_NAME': 'served',
      'CONTEXT_LEN': ''},
     ['local-completions against http://host:8000/v1/completions (model=served)',
      'Using context limit: 262144',
      'Thinking sampling: do_sample=true temperature=0.6 top_p=0.95 top_k=<unset> min_p=<unset> '
      'repetition_penalty=<unset> max_gen_toks=8192',
      'Thinking/reasoning-effort run: max_gen_toks=8192 for every task (replaces task-local budgets)']),
    ('effort alone gets the budget',
     {'TOKENIZER': '/ckpt', 'REASONING_EFFORT': 'low', 'MAX_LENGTH': '65536'},
     {'TOKENIZER': '/ckpt',
      'MAX_LENGTH': '65536',
      'MAX_NEW_TOKENS': '65536',
      'ENABLE_THINKING': 'False',
      'ENABLE_THINKING_EXPLICIT': '0',
      'AUTODETECT_THINK_TOKENS': 'False',
      'TRACK_THINKING_METRICS': '',
      'LOG_LENGTH_METRICS': 'False',
      'REASONING_EFFORT': 'low',
      'CHAT_FORMAT': 'template',
      'PREFILL_MODE': '',
      'ANSWER_FRAME': '',
      'GEN_KWARGS': 'max_gen_toks=65536',
      'API_LM_EVAL_MODEL': 'local-completions',
      'COMMON_MODEL_ARGS': 'model=served,base_url=http://host:8000/v1/completions,num_concurrent=8,max_retries=3,timeout=3600,tokenizer_backend=huggingface,tokenizer=/ckpt,tokenized_requests=False,max_gen_toks=65536,max_length=65536,clamp_gen_toks=True',
      'CHAT_TEMPLATE_ARGS': ' \'chat_template_args={\\"reasoning_effort\\":\\"low\\"}\'',
      'API_BASE_URL': 'http://host:8000/v1/completions',
      'API_MODEL_NAME': 'served',
      'CONTEXT_LEN': ''},
     ['ENABLE_THINKING=True or REASONING_EFFORT set: raising default MAX_NEW_TOKENS to 65536',
      'local-completions against http://host:8000/v1/completions (model=served)',
      'Using context limit: 65536',
      'Thinking/reasoning-effort run: max_gen_toks=65536 for every task (replaces task-local budgets)']),
    ('booleans are normalised',
     {'TOKENIZER': '/ckpt',
      'ENABLE_THINKING': 'yes',
      'LOG_LENGTH_METRICS': '1',
      'AUTODETECT_THINK_TOKENS': 'TRUE',
      'TRACK_THINKING_METRICS': 'no'},
     {'TOKENIZER': '/ckpt',
      'MAX_LENGTH': '',
      'MAX_NEW_TOKENS': '32768',
      'ENABLE_THINKING': 'True',
      'ENABLE_THINKING_EXPLICIT': '1',
      'AUTODETECT_THINK_TOKENS': 'True',
      'TRACK_THINKING_METRICS': 'False',
      'LOG_LENGTH_METRICS': 'True',
      'REASONING_EFFORT': '',
      'CHAT_FORMAT': 'template',
      'PREFILL_MODE': '',
      'ANSWER_FRAME': '',
      'GEN_KWARGS': 'do_sample=true,temperature=0.6,top_p=0.95,max_gen_toks=32768',
      'API_LM_EVAL_MODEL': 'local-completions',
      'COMMON_MODEL_ARGS': 'model=served,base_url=http://host:8000/v1/completions,num_concurrent=8,max_retries=3,timeout=3600,tokenizer_backend=huggingface,tokenizer=/ckpt,tokenized_requests=False,max_gen_toks=32768,autodetect_think_tokens=True,track_thinking_metrics=False,enable_thinking=True',
      'CHAT_TEMPLATE_ARGS': '',
      'API_BASE_URL': 'http://host:8000/v1/completions',
      'API_MODEL_NAME': 'served',
      'CONTEXT_LEN': ''},
     ['ENABLE_THINKING=True or REASONING_EFFORT set: raising default MAX_NEW_TOKENS to 32768',
      'local-completions against http://host:8000/v1/completions (model=served)',
      'Thinking sampling: do_sample=true temperature=0.6 top_p=0.95 top_k=<unset> min_p=<unset> '
      'repetition_penalty=<unset> max_gen_toks=32768',
      'Thinking/reasoning-effort run: max_gen_toks=32768 for every task (replaces task-local budgets)']),
    ('thinking samples with the defaults',
     {'TOKENIZER': '/ckpt', 'ENABLE_THINKING': 'true', 'THINK_END_TOKEN': '</think>'},
     {'TOKENIZER': '/ckpt',
      'MAX_LENGTH': '',
      'MAX_NEW_TOKENS': '32768',
      'ENABLE_THINKING': 'True',
      'ENABLE_THINKING_EXPLICIT': '1',
      'AUTODETECT_THINK_TOKENS': 'False',
      'TRACK_THINKING_METRICS': '',
      'LOG_LENGTH_METRICS': 'False',
      'REASONING_EFFORT': '',
      'CHAT_FORMAT': 'template',
      'PREFILL_MODE': '',
      'ANSWER_FRAME': '',
      'GEN_KWARGS': 'do_sample=true,temperature=0.6,top_p=0.95,max_gen_toks=32768',
      'API_LM_EVAL_MODEL': 'local-completions',
      'COMMON_MODEL_ARGS': 'model=served,base_url=http://host:8000/v1/completions,num_concurrent=8,max_retries=3,timeout=3600,tokenizer_backend=huggingface,tokenizer=/ckpt,tokenized_requests=False,max_gen_toks=32768,think_end_token=</think>,enable_thinking=True',
      'CHAT_TEMPLATE_ARGS': '',
      'API_BASE_URL': 'http://host:8000/v1/completions',
      'API_MODEL_NAME': 'served',
      'CONTEXT_LEN': ''},
     ['ENABLE_THINKING=True or REASONING_EFFORT set: raising default MAX_NEW_TOKENS to 32768',
      'local-completions against http://host:8000/v1/completions (model=served)',
      'Thinking sampling: do_sample=true temperature=0.6 top_p=0.95 top_k=<unset> min_p=<unset> '
      'repetition_penalty=<unset> max_gen_toks=32768',
      'Thinking/reasoning-effort run: max_gen_toks=32768 for every task (replaces task-local budgets)']),
    ('sampling overrides and none',
     {'TOKENIZER': '/ckpt',
      'ENABLE_THINKING': 'true',
      'THINK_TEMPERATURE': '0.8',
      'THINK_TOP_P': 'none',
      'THINK_TOP_K': '20',
      'THINK_MIN_P': '0.05'},
     {'TOKENIZER': '/ckpt',
      'MAX_LENGTH': '',
      'MAX_NEW_TOKENS': '32768',
      'ENABLE_THINKING': 'True',
      'ENABLE_THINKING_EXPLICIT': '1',
      'AUTODETECT_THINK_TOKENS': 'False',
      'TRACK_THINKING_METRICS': '',
      'LOG_LENGTH_METRICS': 'False',
      'REASONING_EFFORT': '',
      'CHAT_FORMAT': 'template',
      'PREFILL_MODE': '',
      'ANSWER_FRAME': '',
      'GEN_KWARGS': 'do_sample=true,temperature=0.8,top_k=20,min_p=0.05,max_gen_toks=32768',
      'API_LM_EVAL_MODEL': 'local-completions',
      'COMMON_MODEL_ARGS': 'model=served,base_url=http://host:8000/v1/completions,num_concurrent=8,max_retries=3,timeout=3600,tokenizer_backend=huggingface,tokenizer=/ckpt,tokenized_requests=False,max_gen_toks=32768,enable_thinking=True',
      'CHAT_TEMPLATE_ARGS': '',
      'API_BASE_URL': 'http://host:8000/v1/completions',
      'API_MODEL_NAME': 'served',
      'CONTEXT_LEN': ''},
     ['ENABLE_THINKING=True or REASONING_EFFORT set: raising default MAX_NEW_TOKENS to 32768',
      'local-completions against http://host:8000/v1/completions (model=served)',
      'Thinking sampling: do_sample=true temperature=0.8 top_p=<unset> top_k=20 min_p=0.05 '
      'repetition_penalty=<unset> max_gen_toks=32768',
      'Thinking/reasoning-effort run: max_gen_toks=32768 for every task (replaces task-local budgets)']),
    ('apertus2 defaults',
     {'CHAT_FORMAT': 'apertus2', 'TOKENIZER': '/ckpt', 'MAX_LENGTH': '65536'},
     {'TOKENIZER': '/ckpt',
      'MAX_LENGTH': '65536',
      'MAX_NEW_TOKENS': '65536',
      'ENABLE_THINKING': 'False',
      'ENABLE_THINKING_EXPLICIT': '0',
      'AUTODETECT_THINK_TOKENS': 'False',
      'TRACK_THINKING_METRICS': '',
      'LOG_LENGTH_METRICS': 'False',
      'REASONING_EFFORT': 'medium',
      'CHAT_FORMAT': 'apertus2',
      'PREFILL_MODE': 'after_think',
      'ANSWER_FRAME': 'reply',
      'GEN_KWARGS': 'do_sample=true,temperature=0.6,top_p=0.95,max_gen_toks=65536',
      'API_LM_EVAL_MODEL': 'apertus2-completions',
      'COMMON_MODEL_ARGS': 'model=served,base_url=http://host:8000/v1/completions,num_concurrent=8,max_retries=3,timeout=3600,tokenizer=/ckpt,prefill=after_think,reasoning_effort=medium,max_gen_toks=65536,max_model_len=65536',
      'CHAT_TEMPLATE_ARGS': '',
      'API_BASE_URL': 'http://host:8000/v1/completions',
      'API_MODEL_NAME': 'served',
      'CONTEXT_LEN': ''},
     ['ENABLE_THINKING=True or REASONING_EFFORT set: raising default MAX_NEW_TOKENS to 65536',
      'apertus2-completions against http://host:8000/v1/completions (model=served)',
      'Using context limit: 65536',
      'Thinking sampling: do_sample=true temperature=0.6 top_p=0.95 top_k=<unset> min_p=<unset> '
      'repetition_penalty=<unset> max_gen_toks=65536',
      'Thinking/reasoning-effort run: max_gen_toks=65536 for every task (replaces task-local budgets)']),
    ('apertus2 with its own sampling',
     {'CHAT_FORMAT': 'apertus2',
      'TOKENIZER': '/ckpt',
      'MAX_LENGTH': '65536',
      'PREFILL_MODE': 'empty_think',
      'ANSWER_FRAME': 'claim',
      'THINK_TEMPERATURE': '1.0',
      'THINK_TOP_K': '20',
      'THINK_REPETITION_PENALTY': 'none'},
     {'TOKENIZER': '/ckpt',
      'MAX_LENGTH': '65536',
      'MAX_NEW_TOKENS': '65536',
      'ENABLE_THINKING': 'False',
      'ENABLE_THINKING_EXPLICIT': '0',
      'AUTODETECT_THINK_TOKENS': 'False',
      'TRACK_THINKING_METRICS': '',
      'LOG_LENGTH_METRICS': 'False',
      'REASONING_EFFORT': 'medium',
      'CHAT_FORMAT': 'apertus2',
      'PREFILL_MODE': 'empty_think',
      'ANSWER_FRAME': 'claim',
      'GEN_KWARGS': 'do_sample=true,temperature=1.0,top_p=0.95,top_k=20,max_gen_toks=65536',
      'API_LM_EVAL_MODEL': 'apertus2-completions',
      'COMMON_MODEL_ARGS': 'model=served,base_url=http://host:8000/v1/completions,num_concurrent=8,max_retries=3,timeout=3600,tokenizer=/ckpt,prefill=empty_think,reasoning_effort=medium,max_gen_toks=65536,answer_frame=claim,max_model_len=65536',
      'CHAT_TEMPLATE_ARGS': '',
      'API_BASE_URL': 'http://host:8000/v1/completions',
      'API_MODEL_NAME': 'served',
      'CONTEXT_LEN': ''},
     ['ENABLE_THINKING=True or REASONING_EFFORT set: raising default MAX_NEW_TOKENS to 65536',
      'apertus2-completions against http://host:8000/v1/completions (model=served)',
      'Using context limit: 65536',
      'Thinking sampling: do_sample=true temperature=1.0 top_p=0.95 top_k=20 min_p=<unset> '
      'repetition_penalty=<unset> max_gen_toks=65536',
      'Thinking/reasoning-effort run: max_gen_toks=65536 for every task (replaces task-local budgets)']),
    ('no-think sampling',
     {'TOKENIZER': '/ckpt', 'NOTHINK_TEMPERATURE': '0.7'},
     {'TOKENIZER': '/ckpt',
      'MAX_LENGTH': '',
      'MAX_NEW_TOKENS': '2048',
      'ENABLE_THINKING': 'False',
      'ENABLE_THINKING_EXPLICIT': '0',
      'AUTODETECT_THINK_TOKENS': 'False',
      'TRACK_THINKING_METRICS': '',
      'LOG_LENGTH_METRICS': 'False',
      'REASONING_EFFORT': '',
      'CHAT_FORMAT': 'template',
      'PREFILL_MODE': '',
      'ANSWER_FRAME': '',
      'GEN_KWARGS': 'do_sample=true,temperature=0.7,top_p=0.95',
      'API_LM_EVAL_MODEL': 'local-completions',
      'COMMON_MODEL_ARGS': 'model=served,base_url=http://host:8000/v1/completions,num_concurrent=8,max_retries=3,timeout=3600,tokenizer_backend=huggingface,tokenizer=/ckpt,tokenized_requests=False',
      'CHAT_TEMPLATE_ARGS': '',
      'API_BASE_URL': 'http://host:8000/v1/completions',
      'API_MODEL_NAME': 'served',
      'CONTEXT_LEN': ''},
     ['local-completions against http://host:8000/v1/completions (model=served)',
      'No-think sampling: do_sample=true temperature=0.7 top_p=0.95 top_k=<unset> min_p=<unset> '
      'repetition_penalty=<unset>']),
    ('no-think sampling with its own top_p',
     {'TOKENIZER': '/ckpt', 'NOTHINK_TEMPERATURE': '0.3', 'NOTHINK_TOP_P': 'none', 'THINK_TOP_K': '20'},
     {'TOKENIZER': '/ckpt',
      'MAX_LENGTH': '',
      'MAX_NEW_TOKENS': '2048',
      'ENABLE_THINKING': 'False',
      'ENABLE_THINKING_EXPLICIT': '0',
      'AUTODETECT_THINK_TOKENS': 'False',
      'TRACK_THINKING_METRICS': '',
      'LOG_LENGTH_METRICS': 'False',
      'REASONING_EFFORT': '',
      'CHAT_FORMAT': 'template',
      'PREFILL_MODE': '',
      'ANSWER_FRAME': '',
      'GEN_KWARGS': 'do_sample=true,temperature=0.3,top_k=20',
      'API_LM_EVAL_MODEL': 'local-completions',
      'COMMON_MODEL_ARGS': 'model=served,base_url=http://host:8000/v1/completions,num_concurrent=8,max_retries=3,timeout=3600,tokenizer_backend=huggingface,tokenizer=/ckpt,tokenized_requests=False',
      'CHAT_TEMPLATE_ARGS': '',
      'API_BASE_URL': 'http://host:8000/v1/completions',
      'API_MODEL_NAME': 'served',
      'CONTEXT_LEN': ''},
     ['local-completions against http://host:8000/v1/completions (model=served)',
      'No-think sampling: do_sample=true temperature=0.3 top_p=<unset> top_k=20 min_p=<unset> '
      'repetition_penalty=<unset>']),
    ('a RULER list',
     {'TOKENIZER': '/ckpt', 'TASKS': 'ruler_8192,gsm8k', 'MAX_LENGTH': '65536'},
     {'TOKENIZER': '/ckpt',
      'MAX_LENGTH': '65536',
      'MAX_NEW_TOKENS': '2048',
      'ENABLE_THINKING': 'False',
      'ENABLE_THINKING_EXPLICIT': '0',
      'AUTODETECT_THINK_TOKENS': 'False',
      'TRACK_THINKING_METRICS': '',
      'LOG_LENGTH_METRICS': 'False',
      'REASONING_EFFORT': '',
      'CHAT_FORMAT': 'template',
      'PREFILL_MODE': '',
      'ANSWER_FRAME': '',
      'GEN_KWARGS': '',
      'API_LM_EVAL_MODEL': 'local-completions',
      'COMMON_MODEL_ARGS': 'model=served,base_url=http://host:8000/v1/completions,num_concurrent=8,max_retries=3,timeout=3600,tokenizer_backend=huggingface,tokenizer=/ckpt,tokenized_requests=False,max_length=8792,clamp_gen_toks=True',
      'CHAT_TEMPLATE_ARGS': '',
      'API_BASE_URL': 'http://host:8000/v1/completions',
      'API_MODEL_NAME': 'served',
      'CONTEXT_LEN': '8192'},
     ['local-completions against http://host:8000/v1/completions (model=served)',
      'Detected ruler context length: 8192 (context limit=8792 and lm-eval metadata)',
      'Using context limit: 8792']),
]


def build(**override):
    env = {**BASE, **override}
    return job_settings.build({name: value for name, value in env.items() if value is not None})


class GoldenTests(unittest.TestCase):
    def test_each_case_matches_what_the_shell_resolved(self):
        for name, override, expected, notes in GOLDEN:
            with self.subTest(name):
                outputs, got_notes = build(**override)
                self.assertEqual({key: outputs[key] for key in expected}, expected)
                self.assertEqual(got_notes, notes)
                self.assertEqual(outputs["SETTINGS_NOTES"], "\n".join(notes))


class RefusalTests(unittest.TestCase):
    def test_apertus2_runs_are_checked(self):
        apertus2 = {"CHAT_FORMAT": "apertus2", "TOKENIZER": "/ckpt", "MAX_LENGTH": "65536"}
        self.assertEqual(build(**apertus2)[0]["REASONING_EFFORT"], "medium")  # an apertus2 run always reasons
        for name, override, message in (
            ("chat endpoint", {"API_CHAT_ENDPOINT": "true"}, "renders every task as chat"),
            ("template off", {"APPLY_CHAT_TEMPLATE": "false"}, "renders every task as chat"),
            ("no context", {"MAX_LENGTH": ""}, "needs MAX_LENGTH"),
            ("no tokenizer", {"TOKENIZER": ""}, "needs TOKENIZER"),
            ("bos", {"BOS": "true"}, "unset BOS"),
            ("prefill mode", {"PREFILL_MODE": "reject"}, "PREFILL_MODE must be"),
            ("answer frame", {"ANSWER_FRAME": "think"}, "ANSWER_FRAME must be"),
            ("thinking switch", {"ENABLE_THINKING": "true"}, "always reasons"),
            ("think token", {"THINK_END_TOKEN": "</think>"}, "always reasons"),
            ("autodetect", {"AUTODETECT_THINK_TOKENS": "true"}, "always reasons"),
            ("thinking metrics", {"TRACK_THINKING_METRICS": "true"}, "always reasons"),
        ):
            with self.subTest(name), self.assertRaisesRegex(ValueError, message):
                build(**{**apertus2, **override})

    def test_other_settings_that_cannot_work_are_refused(self):
        for override, message in (
            ({"MAX_LENGTH": "32k"}, "MAX_LENGTH must be a positive integer"),
            ({"MAX_LENGTH": "0"}, "MAX_LENGTH must be a positive integer"),
            ({"ENABLE_THINKING": "ture"}, "ENABLE_THINKING must be true or false"),
            ({"LOG_LENGTH_METRICS": "bad"}, "LOG_LENGTH_METRICS must be true or false"),
            ({"CHAT_FORMAT": "harmony"}, "CHAT_FORMAT must be template or apertus2"),
            ({"BOS_BY_CLIENT": "sometimes"}, "BOS_BY_CLIENT must be auto, true or false"),
            ({"THINK_END_TOKEN": "a,b"}, "THINK_END_TOKEN must not contain a comma"),
            ({"API_MODEL_NAME": "a,b"}, "API_MODEL_NAME must not contain a comma"),
            ({"THINK_TEMPERATURE": "hot", "ENABLE_THINKING": "true"}, "Invalid sampling settings: THINK_"),
            ({"NOTHINK_TEMPERATURE": "-1"}, "Invalid sampling settings: NOTHINK_"),
        ):
            with self.subTest(message, **override), self.assertRaisesRegex(ValueError, message):
                build(**override)


class BehaviourTests(unittest.TestCase):
    def test_a_tokenizer_that_was_not_named_comes_from_the_model(self):
        self.assertEqual(job_settings.default_tokenizer("org/model")[0], "org/model")
        self.assertEqual(job_settings.default_tokenizer("CSCS-Inference/swiss-ai/Apertus-v1.5-70B")[0],
                         "swiss-ai/Apertus-v1.5-70B")
        self.assertEqual(job_settings.default_tokenizer("/capstor/store/x/ckpt")[0], "/capstor/store/x/ckpt")
        # Named, even empty: it is not replaced (an apertus2 run then refuses it).
        self.assertEqual(build(TOKENIZER="/ckpt")[0]["TOKENIZER"], "/ckpt")

    def test_thinking_budget_is_the_window_where_requests_are_clamped(self):
        # clamp_gen_toks (or apertus2's own clamp) lowers every request to window - prompt, so the budget is
        # the window; without one it keeps 32768, the largest task budget, so no task is lowered.
        budget = lambda **env: build(TOKENIZER="/ckpt", **env)[0]["MAX_NEW_TOKENS"]
        self.assertEqual(budget(ENABLE_THINKING="true", MAX_LENGTH="262144"), "262144")
        self.assertEqual(budget(ENABLE_THINKING="true", MAX_LENGTH="131072"), "131072")
        self.assertEqual(budget(ENABLE_THINKING="true"), job_settings.THINKING_BUDGET)
        self.assertEqual(job_settings.THINKING_BUDGET, "32768")
        self.assertEqual(budget(REASONING_EFFORT="low", MAX_LENGTH="262144"), "262144")
        self.assertEqual(budget(MAX_LENGTH="262144"), "2048")  # not thinking: the default stays
        # a caller's value wins
        self.assertEqual(budget(ENABLE_THINKING="true", MAX_NEW_TOKENS="8192", MAX_LENGTH="262144"), "8192")

    def test_gen_kwargs_carry_the_sampling_and_replace_task_budgets(self):
        # The thinking budget overrides every task's YAML max_gen_toks (mmlu_pro 2048, ifeval 1280).
        gen = lambda **env: build(**{"TOKENIZER": "/ckpt", "MAX_NEW_TOKENS": "32768", **env})[0]["GEN_KWARGS"]
        self.assertEqual(gen(ENABLE_THINKING="true"), "do_sample=true,temperature=0.6,top_p=0.95,max_gen_toks=32768")
        self.assertEqual(gen(REASONING_EFFORT="low", MAX_NEW_TOKENS="8192"), "max_gen_toks=8192")
        self.assertEqual(gen(), "")
        # apertus2 samples like a thinking run; a no-think run only when asked
        self.assertEqual(
            gen(CHAT_FORMAT="apertus2", MAX_LENGTH="65536"),
            "do_sample=true,temperature=0.6,top_p=0.95,max_gen_toks=32768",
        )
        self.assertEqual(gen(NOTHINK_TEMPERATURE="0.7"), "do_sample=true,temperature=0.7,top_p=0.95")
        self.assertEqual(gen(NOTHINK_TEMPERATURE="0.7", NOTHINK_TOP_P="none", THINK_TOP_K="20"),
                         "do_sample=true,temperature=0.7,top_k=20")


class OldPythonTests(unittest.TestCase):
    """The job's own `python3` outside the container can be 3.6 (bos_default.py says so)."""

    def test_the_modules_use_python_3_6_syntax(self):
        for module in (job_settings, job_settings.model_args):
            ast.parse(Path(module.__file__).read_text(), feature_version=(3, 6))

    def test_only_a_run_that_samples_needs_the_newer_sampling_module(self):
        self.assertNotIn("sampling", vars(job_settings))
        self.assertEqual(job_settings.SAMPLERS, __import__("scripts.common.sampling", fromlist=["x"]).SAMPLERS)
        script = ("import sys; from scripts.lm_eval_harness import job_settings\n"
                  "job_settings.build(%r)\nassert 'scripts.common.sampling' not in sys.modules" % BASE)
        done = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT, capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)


class CommandLineTests(unittest.TestCase):
    def run_cli(self, **override):
        env = {"PATH": os.environ["PATH"], **BASE, **override}
        return subprocess.run([sys.executable, "-m", "scripts.lm_eval_harness.job_settings"], cwd=REPO_ROOT, env=env,
                              capture_output=True, text=True)

    def test_prints_assignments_the_shell_can_eval(self):
        done = self.run_cli(REASONING_EFFORT="high", MAX_LENGTH="65536")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stderr, "")  # the notes come back as SETTINGS_NOTES, for the job to print
        script = 'eval "$1"; printf "%s\\n" "$API_LM_EVAL_MODEL" "$COMMON_MODEL_ARGS" "$CHAT_TEMPLATE_ARGS" "$GEN_KWARGS" "$SETTINGS_NOTES"'
        got = subprocess.run(["bash", "-c", script, "bash", done.stdout], capture_output=True, text=True,
                             check=True).stdout.split("\n")
        outputs, notes = build(REASONING_EFFORT="high", MAX_LENGTH="65536")
        self.assertEqual(got[:4], [outputs["API_LM_EVAL_MODEL"], outputs["COMMON_MODEL_ARGS"],
                                   outputs["CHAT_TEMPLATE_ARGS"], outputs["GEN_KWARGS"]])
        self.assertEqual(got[4:4 + len(notes)], notes)

    def test_a_refused_setting_exits_non_zero_with_its_message(self):
        done = self.run_cli(API_BASE_URL="")
        self.assertEqual(done.returncode, 1)
        self.assertEqual(done.stdout, "")
        self.assertIn("API_BASE_URL must be set", done.stderr)

    def test_an_unset_tokenizer_is_not_an_empty_one(self):
        env = {"PATH": os.environ["PATH"], **{k: v for k, v in BASE.items()}}
        done = subprocess.run([sys.executable, "-m", "scripts.lm_eval_harness.job_settings"], cwd=REPO_ROOT, env=env,
                              capture_output=True, text=True)
        self.assertIn("TOKENIZER=org/model", done.stdout)
        self.assertIn("TOKENIZER not set, using TOKENIZER=org/model", done.stdout)


class JobContractTests(unittest.TestCase):
    def test_the_job_exports_exactly_the_inputs_the_module_reads(self):
        line = next(l for l in SBATCH.splitlines() if l.startswith("export MODEL "))
        self.assertEqual(set(line.split()[1:]), set(job_settings.INPUTS))

    def test_the_job_captures_the_output_before_eval_so_a_failure_stops_it(self):
        # `eval "$(cmd)"` returns eval's status, not cmd's: a refused setting would run on with empty arguments.
        self.assertIn("SETTINGS_OUT=$(python3 -m scripts.lm_eval_harness.job_settings) || die", SBATCH)
        self.assertNotIn('eval "$(python3 -m scripts.lm_eval_harness', SBATCH)
        self.assertIn('eval "$SETTINGS_OUT"', SBATCH)

    def test_the_notes_are_printed_where_the_job_keeps_its_output(self):
        # They used to be echoed after the job started copying its output into run.log.
        self.assertLess(SBATCH.index('exec > >(tee -a "$HARNESS_EVAL_DIR/run.log")'), SBATCH.index('"$SETTINGS_NOTES"'))

    def test_the_settings_are_resolved_before_anything_uses_them(self):
        call = SBATCH.index('eval "$SETTINGS_OUT"')
        for use in ("--batch_size $BS", "$GEN_KWARGS", "$COMMON_MODEL_ARGS", "$API_LM_EVAL_MODEL"):
            self.assertLess(call, SBATCH.index(use), use)


if __name__ == "__main__":
    unittest.main()
