"""What an lm-eval job runs with: its settings, resolved and checked, and the strings built from them.

run_lm_eval.sbatch used to normalise and check these in shell. `resolve()` is that logic as a function of the
environment (the variables run_lm_eval.sbatch documents), so it can be called and tested without a Slurm job:

    resolve(env)      defaults, booleans, the Apertus 2 format's rules, the thinking budget, the tokenizer
    gen_kwargs(...)   lm-eval's --gen_kwargs: the sampling and the thinking budget
    model_args.build  the harness model and its --model_args (the settings above, as strings)

The job runs it as

    SETTINGS_OUT=$(python3 -m scripts.lm_eval_harness.job_settings) || die "..."
    eval "$SETTINGS_OUT"

(captured first: `eval "$(...)"` would hide the exit code), which reads the environment (INPUTS lists what it
reads) and prints shell assignments for everything the job needs afterwards (OUTPUTS). What it has to say about
the run (the raised budget, the tokenizer chosen, the sampling) comes back as SETTINGS_NOTES, for the job to
code 1.

Plain Python 3.6 syntax, and scripts.common.sampling (Python 3.8) is imported only by a run that samples: the
job's own `python3` outside the container can be that old, and a run that does not sample used to need no
newer one.
"""

import os
import re
import shlex
import sys

from scripts.lm_eval_harness import model_args

SAMPLERS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty")  # scripts.common.sampling.SAMPLERS

INPUTS = (
    "MODEL", "TOKENIZER", "BOS", "BOS_BY_CLIENT", "MAX_LENGTH", "MAX_NEW_TOKENS", "ENABLE_THINKING",
    "AUTODETECT_THINK_TOKENS", "THINK_START_TOKEN", "THINK_END_TOKEN", "TRACK_THINKING_METRICS", "LOG_LENGTH_METRICS",
    "REASONING_EFFORT", "CHAT_FORMAT", "PREFILL_MODE", "ANSWER_FRAME", "APPLY_CHAT_TEMPLATE", "API_BASE_URL",
    "API_MODEL_NAME", "API_CHAT_ENDPOINT", "API_NUM_CONCURRENT", "API_MAX_RETRIES", "API_REQUEST_TIMEOUT",
    "API_REQUESTS_PER_MINUTE", "TRUST_REMOTE_CODE", "TASKS", "NOTHINK_TEMPERATURE", "NOTHINK_TOP_P",
) + tuple("THINK_" + name.upper() for name in SAMPLERS)
OUTPUTS = (
    "TOKENIZER", "MAX_LENGTH", "MAX_NEW_TOKENS", "ENABLE_THINKING",
    "ENABLE_THINKING_EXPLICIT", "AUTODETECT_THINK_TOKENS", "TRACK_THINKING_METRICS", "LOG_LENGTH_METRICS",
    "REASONING_EFFORT", "CHAT_FORMAT", "PREFILL_MODE", "ANSWER_FRAME", "GEN_KWARGS", "SETTINGS_NOTES",
) + model_args.OUTPUTS

THINKING_BUDGET = "32768"  # the largest task-local budget (AIME), so raising to it lowers no task


def norm_bool(name, value):
    """True/False for a boolean setting, ValueError otherwise: a typo'd 'ture' read as false would waste a job."""
    text = value.strip()
    if text in ("true", "True", "TRUE", "1", "yes"):
        return "True"
    if text in ("false", "False", "FALSE", "0", "no"):
        return "False"
    raise ValueError("%s must be true or false (got '%s')." % (name, value))


def default_tokenizer(model):
    """The tokenizer of a model whose tokenizer was not named, and the note saying so.

    Catalog ids from an inference gateway are prefixed (<catalog-namespace>/<org>/<repo>, e.g.
    CSCS-Inference/swiss-ai/Apertus-v1.5-70B): not a Hugging Face repo id (org/repo, exactly one slash), so
    AutoTokenizer rejects them. The catalog prefix is stripped to recover the real tokenizer repo; a plain
    org/repo id (one slash) is left alone, and so are local paths (always absolute here): stripping up to
    the first slash would mangle them."""
    if not model.startswith("/") and model.count("/") > 1:
        tokenizer = model.split("/", 1)[1]
        return tokenizer, "TOKENIZER not set, using TOKENIZER=%s (catalog prefix stripped from MODEL=%s)" % (
            tokenizer, model)
    return model, "TOKENIZER not set, using TOKENIZER=" + model


def resolve(env, notes=None):
    """The environment -> the job's settings, defaulted, normalised and checked ({OUTPUTS...} plus the
    API_* and TASKS settings model_args.build reads). Raises ValueError for a setting that cannot work."""
    notes = notes if notes is not None else []
    get = lambda name, default="": env.get(name) or default  # ${NAME:-default}: unset and empty alike
    settings = dict((name, get(name)) for name in INPUTS)

    max_length = get("MAX_LENGTH")
    if max_length and not re.match(r"^[1-9][0-9]*$", max_length):
        raise ValueError("MAX_LENGTH must be a positive integer when set (got '%s')." % max_length)
    # Explicitness is captured before defaulting: thinking raises the default budget, but a caller's value wins.
    max_new_tokens_explicit = bool(get("MAX_NEW_TOKENS"))
    max_new_tokens = get("MAX_NEW_TOKENS", "2048")
    # enable_thinking is forwarded only when set (unset lets the template default apply).
    enable_explicit = bool(get("ENABLE_THINKING"))
    enable_thinking = norm_bool("ENABLE_THINKING", get("ENABLE_THINKING", "false"))
    autodetect = norm_bool("AUTODETECT_THINK_TOKENS", get("AUTODETECT_THINK_TOKENS", "false"))
    log_lengths = norm_bool("LOG_LENGTH_METRICS", get("LOG_LENGTH_METRICS", "false"))
    track = norm_bool("TRACK_THINKING_METRICS", get("TRACK_THINKING_METRICS")) if get("TRACK_THINKING_METRICS") else ""
    effort = get("REASONING_EFFORT")
    think_start, think_end = get("THINK_START_TOKEN"), get("THINK_END_TOKEN")

    # CHAT_FORMAT=apertus2: the Apertus 2 native format, rendered and parsed client-side by apertus2-completions;
    # a thinking run (effort, budget and sampling below).
    chat_format = get("CHAT_FORMAT", "template")
    prefill, frame = get("PREFILL_MODE"), get("ANSWER_FRAME")
    apply_template = get("APPLY_CHAT_TEMPLATE", "true")
    if chat_format == "apertus2":
        prefill, frame = prefill or "after_think", frame or "reply"
        if not get("TOKENIZER"):
            raise ValueError("CHAT_FORMAT=apertus2 needs TOKENIZER, the checkpoint's tokenizer.")
        if get("BOS", "auto") == "true":
            raise ValueError("CHAT_FORMAT=apertus2 renders the BOS token itself: unset BOS.")
        if get("API_CHAT_ENDPOINT", "false") == "true" or apply_template != "true":
            raise ValueError("CHAT_FORMAT=apertus2 renders every task as chat over /v1/completions "
                             "(API_CHAT_ENDPOINT=false, APPLY_CHAT_TEMPLATE=true).")
        if not max_length:
            raise ValueError("CHAT_FORMAT=apertus2 needs MAX_LENGTH, the server's --max-model-len.")
        if prefill not in ("after_think", "empty_think"):
            raise ValueError("PREFILL_MODE must be after_think or empty_think (got '%s')." % prefill)
        if frame not in ("reply", "claim"):
            raise ValueError("ANSWER_FRAME must be reply or claim (got '%s')." % frame)
        if enable_explicit or think_end or think_start or track or autodetect == "True":
            raise ValueError("CHAT_FORMAT=apertus2 always reasons and separates the reasoning itself: unset "
                             "ENABLE_THINKING, THINK_END_TOKEN, THINK_START_TOKEN, AUTODETECT_THINK_TOKENS and "
                             "TRACK_THINKING_METRICS.")
        effort = effort or "medium"
    elif chat_format != "template":
        raise ValueError("CHAT_FORMAT must be template or apertus2 (got '%s')." % chat_format)

    # Thinking emits a trace before the answer; raise the budget unless the caller set it. A model given a
    # reasoning effort (gpt-oss) reasons at every level, so it gets the same budget.
    if (enable_thinking == "True" or effort) and not max_new_tokens_explicit:
        # Every generation request is clamped to the window minus its prompt (clamp_gen_toks; apertus2 clamps
        # itself), so the budget is the whole window: a model served with more than 32768 is not cut there.
        max_new_tokens = max_length or THINKING_BUDGET
        notes.append("ENABLE_THINKING=True or REASONING_EFFORT set: raising default MAX_NEW_TOKENS to " + max_new_tokens)

    # lm_eval splits --model_args on commas, so a comma in a value would inject an extra key.
    for name in ("THINK_END_TOKEN", "THINK_START_TOKEN", "API_BASE_URL", "API_MODEL_NAME"):
        if "," in get(name):
            raise ValueError("%s must not contain a comma (got '%s'): it would inject an extra model_args key."
                             % (name, get(name)))

    if "TOKENIZER" in env:
        tokenizer = env["TOKENIZER"]
    else:
        tokenizer, note = default_tokenizer(get("MODEL"))
        notes.append(note)

    by_client = get("BOS_BY_CLIENT", "auto")
    if by_client not in ("auto", "true", "false"):
        raise ValueError("BOS_BY_CLIENT must be auto, true or false (got '%s')" % by_client)

    settings.update({
        "TOKENIZER": tokenizer, "MAX_LENGTH": max_length, "MAX_NEW_TOKENS": max_new_tokens,
        "ENABLE_THINKING": enable_thinking,
        "ENABLE_THINKING_EXPLICIT": "1" if enable_explicit else "0", "AUTODETECT_THINK_TOKENS": autodetect,
        "TRACK_THINKING_METRICS": track, "LOG_LENGTH_METRICS": log_lengths, "REASONING_EFFORT": effort,
        "CHAT_FORMAT": chat_format, "PREFILL_MODE": prefill, "ANSWER_FRAME": frame, "APPLY_CHAT_TEMPLATE": apply_template,
    })
    return settings


def sampling_values(env, settings):
    """The sampling settings of a run: thinking and Apertus 2 runs sample (the defaults of
    configs/sampling_defaults.json, overridden one by one by THINK_*; "none" leaves one out); a no-think run
    samples only when NOTHINK_TEMPERATURE is set (a greedy-vs-sampling ablation, top_p 0.95 unless
    NOTHINK_TOP_P). Returns (values, label) or (None, "")."""
    def overrides(skip=()):
        from scripts.common import sampling  # Python 3.8: only a run that samples needs it
        found = {}
        for name in SAMPLERS:
            text = env.get("THINK_" + name.upper()) or None
            if text is not None and name not in skip:
                found[name] = sampling.parse_value(name, text)
        return found

    if settings["ENABLE_THINKING"] == "True" or settings["CHAT_FORMAT"] == "apertus2":
        from scripts.common import sampling
        try:
            values = sampling.resolve("lm_eval", config=sampling.read_config(), overrides=overrides())
        except ValueError as error:
            raise ValueError("Invalid sampling settings: THINK_* or configs/sampling_defaults.json: %s" % error)
        return values, "Thinking sampling"
    if env.get("NOTHINK_TEMPERATURE"):
        from scripts.common import sampling
        # Optional no-think sampling (inert unless NOTHINK_TEMPERATURE is set): top_p defaults to 0.95 ("none"
        # omits it); THINK_TOP_K/MIN_P/REPETITION_PENALTY apply too.
        try:
            explicit = {"temperature": sampling.parse_value("temperature", env["NOTHINK_TEMPERATURE"]),
                        "top_p": sampling.parse_value("top_p", env.get("NOTHINK_TOP_P") or "0.95")}
            found = overrides(skip=tuple(explicit))
            found.update(explicit)
            values = sampling.resolve("lm_eval", config=sampling.read_config(), overrides=found, use_default=False)
        except ValueError as error:
            raise ValueError("Invalid sampling settings: NOTHINK_* / THINK_*: %s" % error)
        return values, "No-think sampling"
    return None, ""


def gen_kwargs(env, settings, notes=None):
    """lm-eval's --gen_kwargs: the sampling, then the thinking budget, which replaces every task's own
    max_gen_toks (task budgets such as mmlu_pro 2048 and ifeval 1280 were sized for a direct answer, and the
    trace counts against them)."""
    notes = notes if notes is not None else []
    kwargs = ""
    values, label = sampling_values(env, settings)
    if values is not None:
        kwargs = "do_sample=true"
        shown = []
        for name in SAMPLERS:
            if name in values:
                kwargs += ",%s=%s" % (name, values[name])
            shown.append("%s=%s" % (name, values.get(name, "<unset>")))
        budget = " max_gen_toks=" + settings["MAX_NEW_TOKENS"] if label == "Thinking sampling" else ""
        notes.append("%s: do_sample=true %s%s" % (label, " ".join(shown), budget))
    if settings["ENABLE_THINKING"] == "True" or settings["REASONING_EFFORT"]:
        kwargs = (kwargs + "," if kwargs else "") + "max_gen_toks=" + settings["MAX_NEW_TOKENS"]
        notes.append("Thinking/reasoning-effort run: max_gen_toks=%s for every task (replaces task-local budgets)"
                     % settings["MAX_NEW_TOKENS"])
    return kwargs


def build(env):
    """The environment -> ({OUTPUTS...}, notes), or ValueError for a setting that cannot work."""
    notes = []
    settings = resolve(env, notes)
    built = model_args.build(settings)
    notes += model_args.notes(settings, built)  # in the order the job printed them: endpoint and context, then sampling
    outputs = dict(settings)
    outputs.update(built)
    outputs["GEN_KWARGS"] = gen_kwargs(env, settings, notes)
    outputs["SETTINGS_NOTES"] = "\n".join(notes)
    return outputs, notes


def main():
    env = dict((name, os.environ[name]) for name in INPUTS if name in os.environ)
    try:
        outputs, _ = build(env)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    for name in OUTPUTS:
        print("%s=%s" % (name, shlex.quote(outputs[name])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
