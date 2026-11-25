"""The harness model and its --model_args for one lm-eval run against an OpenAI-compatible endpoint.

run_lm_eval.sbatch used to build these strings in shell. `build()` is the same logic as a function of
the run's settings, so it can be called and tested without a Slurm job. The job runs it as

    eval "$(python3 -m scripts.lm_eval_harness.model_args)"

which reads the settings from the environment (the variables run_lm_eval.sbatch documents; INPUTS below
lists the ones it reads) and prints shell assignments for what the job needs: API_LM_EVAL_MODEL (the
harness model), COMMON_MODEL_ARGS, CHAT_TEMPLATE_ARGS, the normalised API_BASE_URL / API_MODEL_NAME /
API_CHAT_ENDPOINT and CONTEXT_LEN (the context length of a RULER task list, for lm-eval's --metadata).
Notes go to stderr; a setting that cannot work stops the job with its message and exit code 1.

Plain Python 3.6 syntax: the job's own `python3` outside the container can be that old.
"""

import os
import re
import shlex
import sys

INPUTS = (
    "MODEL", "API_BASE_URL", "API_MODEL_NAME", "API_CHAT_ENDPOINT", "API_NUM_CONCURRENT", "API_MAX_RETRIES",
    "API_REQUEST_TIMEOUT", "API_REQUESTS_PER_MINUTE", "CHAT_FORMAT", "APPLY_CHAT_TEMPLATE", "TOKENIZER",
    "PREFILL_MODE", "ANSWER_FRAME", "REASONING_EFFORT", "ENABLE_THINKING", "ENABLE_THINKING_EXPLICIT",
    "THINK_START_TOKEN", "THINK_END_TOKEN", "AUTODETECT_THINK_TOKENS", "TRACK_THINKING_METRICS", "MAX_NEW_TOKENS",
    "TRUST_REMOTE_CODE", "MAX_LENGTH", "TASKS",
)
OUTPUTS = (
    "API_LM_EVAL_MODEL", "COMMON_MODEL_ARGS", "CHAT_TEMPLATE_ARGS", "API_BASE_URL", "API_MODEL_NAME",
    "API_CHAT_ENDPOINT", "CONTEXT_LEN",
)

APERTUS2 = "apertus2-completions"
CHAT = "local-chat-completions"
COMPLETIONS = "local-completions"


def thinking_args(settings):
    """The reasoning model_args: only non-defaults, so an unset setting keeps the harness default."""
    args = ""
    if settings.get("THINK_END_TOKEN"):
        args += ",think_end_token=" + settings["THINK_END_TOKEN"]
    if settings.get("THINK_START_TOKEN"):
        args += ",think_start_token=" + settings["THINK_START_TOKEN"]
    if settings.get("AUTODETECT_THINK_TOKENS") == "True":
        args += ",autodetect_think_tokens=True"
    if settings.get("TRACK_THINKING_METRICS"):
        args += ",track_thinking_metrics=" + settings["TRACK_THINKING_METRICS"]
    return args


def chat_template_args(effort):
    """The chat-template arguments as a second --model_args token.

    The harness refuses a first token containing '{' (it must then be all JSON) but splits each further
    token with shlex (POSIX, dropping unescaped quotes) and reads its value with ast.literal_eval, so the
    quotes are escaped. It needs the `--model_args 'a' 'b'` form: argparse gives `--model_args='a'` exactly
    one value."""
    if not effort:
        return ""
    if not re.match(r"^[A-Za-z0-9_-]+$", effort):
        raise ValueError("REASONING_EFFORT must be a level like low, medium or high (got '%s')." % effort)
    return " 'chat_template_args={\\\"reasoning_effort\\\":\\\"%s\\\"}'" % effort


def endpoint(settings):
    """(harness model, endpoint URL) for the settings: the Apertus 2 format, chat-completions or completions.

    /v1/completions is the default because it serves both generative and log-likelihood tasks, so mixed
    suites work; with the chat template on, the harness renders the model's template client-side. A purely
    generative suite can use the server-side template with API_CHAT_ENDPOINT=true."""
    url = settings.get("API_BASE_URL", "")
    if not url:
        raise ValueError("API_BASE_URL must be set (e.g. http://<host>:8000/v1).")
    if settings.get("CHAT_FORMAT") == "apertus2":
        model, suffix = APERTUS2, "completions"
    elif settings.get("API_CHAT_ENDPOINT") == "true":
        if settings.get("APPLY_CHAT_TEMPLATE") != "true":
            raise ValueError("API_CHAT_ENDPOINT=true requires the chat template (chat/completions takes messages).")
        model, suffix = CHAT, "chat/completions"
    else:
        model, suffix = COMPLETIONS, "completions"
    # A bare host, a /v1 root, or an already-full endpoint URL.
    if not url.endswith("/chat/completions") and not url.endswith("/completions"):
        url = url[:-1] if url.endswith("/") else url
        if not url.endswith("/v1"):
            url += "/v1"
        url += "/" + suffix
    return model, url


def ruler_context(tasks):
    """The context length a RULER task list names (ruler_8192, ruler-8192 or .../8192), or ""."""
    if "ruler" not in tasks:
        return ""
    found = re.search(r"/([0-9]+)", tasks) or re.search(r"ruler[_-]([0-9]+)", tasks)
    return found.group(1) if found else ""


def build(settings):
    """The run's endpoint settings -> {OUTPUTS...}. `settings` maps the INPUTS names to strings."""
    get = lambda name, default="": settings.get(name) or default
    model, url = endpoint(settings)
    api_model = get("API_MODEL_NAME", get("MODEL"))
    effort = get("REASONING_EFFORT")
    think = thinking_args(settings)
    template_args = chat_template_args(effort)
    # The backend default for a request without a max_gen_toks; --gen_kwargs is what replaces the task budgets.
    default_budget = ""
    if get("ENABLE_THINKING") == "True" or effort:
        default_budget = ",max_gen_toks=" + get("MAX_NEW_TOKENS")
    # No model is loaded in the job, so dtype and the like do not apply.
    args = "model=%s,base_url=%s,num_concurrent=%s,max_retries=%s,timeout=%s" % (
        api_model, url, get("API_NUM_CONCURRENT"), get("API_MAX_RETRIES"), get("API_REQUEST_TIMEOUT"))
    if get("API_REQUESTS_PER_MINUTE"):
        args += ",requests_per_minute=" + get("API_REQUESTS_PER_MINUTE")
    explicit = get("ENABLE_THINKING_EXPLICIT", "0") not in ("0", "")
    if model == APERTUS2:
        # The model reads the effort itself (no chat template); the context goes in as max_model_len (below).
        args += ",tokenizer=%s,prefill=%s,reasoning_effort=%s%s" % (
            get("TOKENIZER"), get("PREFILL_MODE"), effort, default_budget)
        if get("ANSWER_FRAME", "reply") == "claim":
            args += ",answer_frame=claim"  # only when asked: default runs keep their model arguments
        template_args = ""
    elif model == COMPLETIONS:
        # tokenizer_backend/tokenizer is local bookkeeping (context length, decoding, token counts); lm-eval
        # still needs it for log-likelihood scoring. tokenized_requests is the separate knob for what goes
        # over the wire: left True, lm-eval ships token ids in the prompt field, which the CSCS gateway's
        # /v1/completions does not parse, so it is forced False. The template is rendered client-side, so
        # its arguments apply here; enable_thinking goes only when set (unset keeps the template default).
        args += ",tokenizer_backend=huggingface,tokenizer=%s,tokenized_requests=False" % get("TOKENIZER")
        args += default_budget + think
        if explicit:
            args += ",enable_thinking=" + get("ENABLE_THINKING")
    else:
        # chat/completions leaves the template to the server: its arguments travel with each request as
        # chat_template_kwargs, and the server's reasoning parser returns the reasoning in its own field,
        # which the harness puts back between the think tokens so it is stripped and measured.
        args += ",tokenized_requests=False" + default_budget + think
        if explicit:
            args += ",enable_thinking=" + get("ENABLE_THINKING")
    # Opt-in, for models whose tokenizer is custom code (e.g. Kimi-K2).
    if get("TRUST_REMOTE_CODE") == "true":
        args += ",trust_remote_code=True"
    context_len = ruler_context(get("TASKS"))
    limit = get("MAX_LENGTH")
    if context_len:
        limit = str(int(context_len) + 600)  # RULER needs a little headroom beyond the requested length
    if limit:
        if get("CHAT_FORMAT") == "apertus2":
            # apertus2-completions clamps each request's budget to the server's context, which ruler's
            # detected length does not change.
            args += ",max_model_len=" + get("MAX_LENGTH")
        else:
            # The template path never counts tokens: a prompt longer than the context minus the budget gets
            # the server's 400. clamp_gen_toks lowers that request's budget to what is left. The clamp counts
            # the prompt with a client tokenizer; without one it clamps nothing. local-completions has one
            # already; chat-completions needs it only for counting (the server renders the template).
            args += ",max_length=%s,clamp_gen_toks=True" % limit
            if model == CHAT:
                args += ",tokenizer_backend=huggingface,tokenizer=" + get("TOKENIZER")
    return {
        "API_LM_EVAL_MODEL": model, "COMMON_MODEL_ARGS": args, "CHAT_TEMPLATE_ARGS": template_args,
        "API_BASE_URL": url, "API_MODEL_NAME": api_model, "API_CHAT_ENDPOINT": "true" if model == CHAT else "false",
        "CONTEXT_LEN": context_len, "CONTEXT_LIMIT": limit,
    }


def main():
    settings = dict((name, os.environ.get(name, "")) for name in INPUTS)
    try:
        built = build(settings)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    print("%s against %s (model=%s)" % (built["API_LM_EVAL_MODEL"], built["API_BASE_URL"], built["API_MODEL_NAME"]),
          file=sys.stderr)
    if settings["API_REQUESTS_PER_MINUTE"]:
        print("rate limit: %s requests/minute" % settings["API_REQUESTS_PER_MINUTE"], file=sys.stderr)
    if built["CONTEXT_LEN"]:
        print("Detected ruler context length: %s (context limit=%s and lm-eval metadata)"
              % (built["CONTEXT_LEN"], built["CONTEXT_LIMIT"]), file=sys.stderr)
    if built["CONTEXT_LIMIT"]:
        print("Using context limit: " + built["CONTEXT_LIMIT"], file=sys.stderr)
    for name in OUTPUTS:
        print("%s=%s" % (name, shlex.quote(built[name])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
