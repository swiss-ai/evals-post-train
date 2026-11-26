"""What the launcher's flags mean together: resolved, checked, and turned into the environment the jobs read.

launch_evaluations.sh parses its flags into shell variables; `resolve()` is what used to follow in shell, as a
function of those variables and the ambient environment: the thinking flags, the repeats policy, the Apertus 2
format's rules, the reasoning effort, the endpoint checks, the sampling flags and the repeat/greedy check. The
launcher runs it as

    LAUNCH_OUT=$(python3 -m scripts.lm_eval_harness.launch_args) || exit 1
    eval "$LAUNCH_OUT"

(captured first: `eval "$(...)"` would hide the exit code). It reads the variables from the environment (the
launcher passes its FLAGS for that one command; AMBIENT are read as they are) and prints shell statements: `NAME=value` for the variables the
launcher goes on to use, `export NAME=value` for what the jobs inherit, `unset NAME`. Warnings go to stderr; a
flag combination that cannot work prints "Error: ..." to stderr and exits 1.
"""

import os
import re
import shlex
import subprocess
import sys
import tempfile

# What the checks read. FLAGS are the launcher's shell variables (its parsed flags): it hands them over for this one
# command only, so they do not reach the jobs' environment. AMBIENT are the environment variables the flags fall back to.
FLAGS = (
    "THINKING_UMBRELLA", "ENABLE_THINKING_OVERRIDE", "THINK_END_TOKEN", "THINK_START_TOKEN", "AUTODETECT_THINK_TOKENS",
    "TRACK_THINKING_METRICS", "LOG_LENGTH_METRICS", "REASONING_EFFORT", "REPEATS_FLAG", "REPEAT_METRIC_FLAG",
    "CHAT_FORMAT_FLAG", "PREFILL_MODE_FLAG", "ANSWER_FRAME_FLAG", "CUSTOM_TOKENIZER", "MAX_LENGTH", "BOS_FLAG",
    "CHAT_TEMPLATE_OVERRIDE", "API_ENDPOINT_FLAG", "API_BASE_URL_FLAG", "API_MODEL_NAME_FLAG",
    "API_MAX_RETRIES_FLAG", "API_NUM_CONCURRENT_FLAG", "API_REQUEST_TIMEOUT_FLAG", "API_REQUESTS_PER_MINUTE_FLAG",
    "JUDGE_REQUESTS_PER_MINUTE_FLAG", "SAMPLING_TEMPERATURE_FLAG", "SAMPLING_TOP_P_FLAG", "SAMPLING_TOP_K_FLAG",
    "SAMPLING_MIN_P_FLAG", "SAMPLING_REPETITION_PENALTY_FLAG", "SAMPLING_PROFILE_FLAG", "SAMPLING_CONFIG_FLAG",
    "TRUST_REMOTE_CODE_FLAG", "MODEL_PATH",
)
AMBIENT = (
    "TOKENIZER", "BOS", "ENABLE_THINKING", "API_CHAT_ENDPOINT", "CHAT_FORMAT", "PREFILL_MODE", "ANSWER_FRAME",
    "EVAL_REPEAT_CONFIG", "API_BASE_URL", "API_MODEL_NAME", "API_REQUESTS_PER_MINUTE", "API_MAX_RETRIES",
    "API_NUM_CONCURRENT", "API_REQUEST_TIMEOUT", "JUDGE_REQUESTS_PER_MINUTE", "JUDGE_MODEL_PREFIX", "USER",
)
INPUTS = FLAGS + AMBIENT

GREEDY = re.compile(r"^0(\.0*)?$")  # a temperature that is greedy decoding
SAMPLERS = ("TEMPERATURE", "TOP_P", "TOP_K", "MIN_P", "REPETITION_PENALTY")  # the sampling flags, as in sampling.py


class Refused(Exception):
    """A flag combination that cannot work: its message is what the launcher prints after "Error: "."""


class Resolution(object):
    """What `resolve()` decided: shell variables, exported variables, unset variables and warnings, in order."""

    def __init__(self, warn=None):
        self.variables, self.exports, self.unset, self.warnings = {}, {}, [], []
        self._warn = warn  # told of each warning as it arises, so one is heard even if a later check refuses the run

    def warn(self, *lines):
        for line in lines:
            self.warnings.append(line)
            if self._warn:
                self._warn(line)

    def statements(self):
        lines = ["%s=%s" % (name, shlex.quote(value)) for name, value in self.variables.items()]
        lines += ["export %s=%s" % (name, shlex.quote(value)) for name, value in self.exports.items()]
        lines += ["unset " + name for name in self.unset]
        return lines


class Flags(object):
    """The thinking-related flags, as the steps below read and change them (one attribute each)."""

    def __init__(self, get):
        self.umbrella, self.enable = get("THINKING_UMBRELLA"), get("ENABLE_THINKING_OVERRIDE")
        self.end, self.start = get("THINK_END_TOKEN"), get("THINK_START_TOKEN")
        self.autodetect, self.track = get("AUTODETECT_THINK_TOKENS"), get("TRACK_THINKING_METRICS")
        self.lengths, self.effort = get("LOG_LENGTH_METRICS"), get("REASONING_EFFORT")
        self.template = get("CHAT_TEMPLATE_OVERRIDE")  # "", "true", "false", "by-type"
        self.repeats = get("REPEATS_FLAG")
        self.prefill, self.frame = get("PREFILL_MODE_FLAG"), get("ANSWER_FRAME_FLAG")
        self.apertus2 = False
        self.sampled = {}  # the resolved sampling flags, by name ("TEMPERATURE", ...)

    def thinks(self):
        """A run that reasons: --enable-thinking, or the Apertus 2 format, which always does."""
        return self.enable == "true" or self.apertus2

    def render_template(self):
        """These flags are chat-template arguments: the template has to be rendered ("by-type" keeps it on the
        generative tasks, the only ones they affect)."""
        if self.template != "by-type":
            self.template = "true"


def resolve(env, run=subprocess.run, warn=None):
    """The launcher's variables (`env`) -> Resolution, or Refused. `run` runs the helper commands (the repeats
    policy, the sampling resolution): as subprocesses, like the shell code this replaces. `warn` hears each warning
    when it arises.

    The steps run in the order the launcher always checked them, so a combination that breaks two rules is refused
    for the same one."""
    get = lambda name, default="": env.get(name) or default  # ${NAME:-default}
    out = Resolution(warn)
    flags = Flags(get)
    check_thinking_flags(flags)
    # Two distinct questions (conflating them rejected every "off" switch): THINKING_TOUCHED, any thinking/length
    # flag was passed (it drives the summary line); THINKING_METRICS_ASKED, the user asked to RECORD metrics, which
    # needs a close token and the chat template. --log-length-metrics alone is neither: response_length_* is
    # recorded for any generative response.
    touched = bool(flags.umbrella == "true" or flags.enable or flags.end or flags.start
                   or flags.autodetect == "true" or flags.track or flags.lengths == "true")
    layer_repeats(flags, get, out, run)
    choose_chat_format(flags, get, out)
    asked = apply_template_switches(flags)
    check_endpoint(flags, get, out)
    model_path = resolve_ambient(get, out)
    check_thinking_metrics(flags, asked, out)
    resolve_sampling(flags, get, out, run)
    refuse_greedy_repeats(flags)
    export_what_was_set(flags, get, out)
    out.variables.update({
        "ENABLE_THINKING_OVERRIDE": flags.enable, "AUTODETECT_THINK_TOKENS": flags.autodetect,
        "TRACK_THINKING_METRICS": flags.track, "LOG_LENGTH_METRICS": flags.lengths,
        "REASONING_EFFORT": flags.effort, "CHAT_TEMPLATE_OVERRIDE": flags.template,
        "PREFILL_MODE_FLAG": flags.prefill, "ANSWER_FRAME_FLAG": flags.frame, "MODEL_PATH": model_path,
        "THINKING_TOUCHED": "true" if touched else "false", "THINKING_METRICS_ASKED": "true" if asked else "false",
    })
    return out


def check_thinking_flags(flags):
    """The thinking flags on their own: valid values, and what --thinking implies. Before dispatch, so the forced
    chat template reaches every mode."""
    if flags.track and flags.track not in ("true", "false"):
        raise Refused("--track-thinking-metrics expects 'true' or 'false' (got '%s')" % flags.track)
    # lm_eval splits --model_args on commas, so a comma in a token would inject an extra key.
    for name, value in (("THINK_END_TOKEN", flags.end), ("THINK_START_TOKEN", flags.start)):
        if "," in value:
            raise Refused("%s must not contain a comma (got '%s')" % (name, value))
    if flags.umbrella == "true":
        flags.enable = flags.enable or "true"
        flags.track = flags.track or "true"
        flags.lengths = flags.lengths or "true"
        # A close token must come from somewhere; prefer the user's if they named one.
        if not flags.end and not flags.autodetect:
            flags.autodetect = "true"


def layer_repeats(flags, get, out, run):
    """--repeats / --repeat-metric: layered over --repeat-config (flags win) into one policy file."""
    metric = get("REPEAT_METRIC_FLAG")
    pass_at = ""
    if flags.repeats and not re.match(r"^[1-9][0-9]*$", flags.repeats):
        raise Refused("--repeats takes a positive integer (got '%s')" % flags.repeats)
    if metric:
        if not re.match(r"^pass@[0-9]+(,[0-9]+)*$", metric):
            raise Refused("--repeat-metric takes pass@K or pass@K1,K2 (got '%s'); the mean@N is always reported"
                          % metric)
        pass_at = metric[len("pass@"):]
    if flags.repeats or pass_at:
        policy = os.path.join(tempfile.mkdtemp(), "repeat_flags.json")
        args = [sys.executable, "-m", "scripts.lm_eval_harness.repeat_config", "--output", policy]
        if get("EVAL_REPEAT_CONFIG"):
            args += ["--policy", get("EVAL_REPEAT_CONFIG")]
        if flags.repeats:
            args += ["--repeats", flags.repeats]
        if pass_at:
            args += ["--pass-at", pass_at]
        if run(args, stdout=subprocess.DEVNULL).returncode:
            raise Refused(None)  # the helper has said why
        out.exports["EVAL_REPEAT_CONFIG"] = policy


def choose_chat_format(flags, get, out):
    """--chat-format: the template (the model's own) or apertus2, where the harness renders and parses the native
    format and always reasons, so the template's thinking switches do not apply and the effort sets the level."""
    chat_format = get("CHAT_FORMAT_FLAG", "template")
    if chat_format == "template":
        if flags.prefill:
            raise Refused("--prefill-mode only applies with --chat-format apertus2")
        if flags.frame:
            raise Refused("--answer-frame only applies with --chat-format apertus2")
        out.unset += ["CHAT_FORMAT", "PREFILL_MODE", "ANSWER_FRAME"]
    elif chat_format == "apertus2":
        check_apertus2(flags, get)
        flags.apertus2 = True
        flags.lengths = "true"
        out.exports.update({"CHAT_FORMAT": "apertus2", "PREFILL_MODE": flags.prefill, "ANSWER_FRAME": flags.frame})
    else:
        raise Refused("--chat-format must be 'template' or 'apertus2' (got '%s')" % chat_format)


def check_apertus2(flags, get):
    """What --chat-format apertus2 needs and cannot be combined with; defaults its prefill, frame and effort."""
    if not (get("CUSTOM_TOKENIZER") or get("TOKENIZER")) or not get("MAX_LENGTH"):
        raise Refused("--chat-format apertus2 needs --tokenizer <the checkpoint's tokenizer> and "
                      "--max-length <the server's --max-model-len>")
    if get("BOS_FLAG") == "true" or get("BOS") == "true":
        raise Refused("--chat-format apertus2 renders the BOS token itself; drop --bos (or BOS=true)")
    endpoint = get("API_ENDPOINT_FLAG")
    if (flags.template in ("false", "by-type") or endpoint == "chat-completions"
            or (not endpoint and get("API_CHAT_ENDPOINT", "false") == "true")):
        raise Refused("--chat-format apertus2 renders every task as chat over /v1/completions; drop "
                      "--no-chat-template, --chat-template-by-type and --api-endpoint chat-completions "
                      "(or API_CHAT_ENDPOINT=true)")
    if (flags.umbrella == "true" or flags.enable or get("ENABLE_THINKING") or flags.end or flags.start
            or flags.autodetect == "true" or flags.track):
        raise Refused("--chat-format apertus2 always reasons and separates the reasoning itself; drop "
                      "--thinking, --enable-thinking/--no-enable-thinking (or ENABLE_THINKING), the think "
                      "tokens and --track-thinking-metrics")
    flags.prefill = flags.prefill or get("PREFILL_MODE", "after_think")
    if flags.prefill not in ("after_think", "empty_think"):
        raise Refused("--prefill-mode must be 'after_think' or 'empty_think' (got '%s')" % flags.prefill)
    flags.frame = flags.frame or get("ANSWER_FRAME", "reply")
    if flags.frame not in ("reply", "claim"):
        raise Refused("--answer-frame must be 'reply' or 'claim' (got '%s')" % flags.frame)
    flags.effort = flags.effort or "medium"
    if flags.effort not in ("low", "medium", "high"):
        raise Refused("--chat-format apertus2 takes --reasoning-effort low, medium or high (got '%s')" % flags.effort)


def apply_template_switches(flags):
    """The thinking switches and the reasoning effort are chat-template arguments, rendered client-side with the
    model's template: they need the template (and turn it on). Returns whether thinking METRICS were asked for:
    --enable-thinking on its own only switches the template; with --thinking or a think token the reasoning is
    stripped and measured."""
    asked = bool(flags.umbrella == "true" or flags.enable == "true" or flags.track == "true" or flags.end
                 or flags.start or flags.autodetect == "true")
    if flags.umbrella != "true" and not flags.end and not flags.start and flags.autodetect != "true" \
            and not flags.track:
        asked = False
    if flags.enable:
        if flags.template == "false":
            raise Refused("--enable-thinking/--no-enable-thinking are chat-template arguments; drop --no-chat-template")
        flags.render_template()
    if flags.effort:
        if not re.match(r"^[A-Za-z0-9_-]+$", flags.effort):
            raise Refused("--reasoning-effort expects a level like low, medium or high (got '%s')" % flags.effort)
        if flags.template == "false":
            raise Refused("--reasoning-effort is a chat-template argument; drop --no-chat-template")
        flags.render_template()
    return asked


def check_endpoint(flags, get, out):
    """The endpoint to evaluate: required before scheduling, its numbers valid, and the chat endpoint only where
    every task can use it."""
    if not (get("API_BASE_URL_FLAG") or get("API_BASE_URL")):
        raise Refused("--api-base-url <url> is required (or an exported API_BASE_URL)")
    # --model defaults API_MODEL_NAME (run_lm_eval.sbatch) to its own, possibly catalog-prefixed value, which the
    # gateway may not recognize -- require the caller to say explicitly what the server should see.
    if not (get("API_MODEL_NAME_FLAG") or get("API_MODEL_NAME")):
        raise Refused("--api-model-name <name> is required (or an exported API_MODEL_NAME)")
    for flag, value in (("--api-max-retries", get("API_MAX_RETRIES_FLAG")),
                        ("--api-num-concurrent", get("API_NUM_CONCURRENT_FLAG")),
                        ("--api-request-timeout", get("API_REQUEST_TIMEOUT_FLAG"))):
        if value and not re.match(r"^[1-9][0-9]*$", value):
            raise Refused("%s must be a positive integer (got '%s')" % (flag, value))
    chat_endpoint = get("API_CHAT_ENDPOINT")
    if get("API_ENDPOINT_FLAG"):
        if get("API_ENDPOINT_FLAG") not in ("completions", "chat-completions"):
            raise Refused("--api-endpoint must be 'completions' or 'chat-completions' (got '%s')"
                          % get("API_ENDPOINT_FLAG"))
        chat_endpoint = "true" if get("API_ENDPOINT_FLAG") == "chat-completions" else "false"
        out.exports["API_CHAT_ENDPOINT"] = chat_endpoint
    # chat/completions only generates: log-likelihood tasks (and their template-free pass) need the completions endpoint.
    if (chat_endpoint or "false") == "true" and flags.template in ("false", "by-type"):
        raise Refused("--api-endpoint chat-completions needs the chat template on every task (generative tasks "
                      "only); drop --no-chat-template/--chat-template-by-type or use --api-endpoint completions")


def resolve_ambient(get, out):
    """The endpoint and judge settings a flag gives or the environment already has (the flag wins), exported for the
    jobs. Returns the model path: --model, else the served name, since no checkpoint is loaded and the run needs a
    name."""
    settings = (("API_REQUESTS_PER_MINUTE", get("API_REQUESTS_PER_MINUTE_FLAG") or get("API_REQUESTS_PER_MINUTE")),
                ("API_MAX_RETRIES", get("API_MAX_RETRIES_FLAG") or get("API_MAX_RETRIES")),
                ("API_NUM_CONCURRENT", get("API_NUM_CONCURRENT_FLAG") or get("API_NUM_CONCURRENT")),
                ("API_REQUEST_TIMEOUT", get("API_REQUEST_TIMEOUT_FLAG") or get("API_REQUEST_TIMEOUT")),
                ("JUDGE_REQUESTS_PER_MINUTE", get("JUDGE_REQUESTS_PER_MINUTE_FLAG") or get("JUDGE_REQUESTS_PER_MINUTE")),
                ("JUDGE_MODEL_PREFIX", get("JUDGE_MODEL_PREFIX") or get("USER")))
    out.variables.update(dict(settings))
    if get("API_BASE_URL_FLAG"):
        out.exports["API_BASE_URL"] = get("API_BASE_URL_FLAG")
    if get("API_MODEL_NAME_FLAG"):
        out.exports["API_MODEL_NAME"] = get("API_MODEL_NAME_FLAG")
    for name, value in settings:
        if value:
            out.exports[name] = value
    return get("MODEL_PATH") or get("API_MODEL_NAME_FLAG") or get("API_MODEL_NAME")


def check_thinking_metrics(flags, asked, out):
    """Thinking metrics need the template and a close token (else the run silently records nothing); two flag
    combinations only deserve a warning."""
    if asked:
        # The reasoning tokens live in the chat template, so it has to be rendered.
        if flags.template == "false":
            raise Refused("thinking metrics require the chat template; drop --no-chat-template")
        flags.render_template()
        # Without a close token nothing is stripped or recorded -- the run silently produces nothing.
        if not flags.end and flags.autodetect != "true":
            raise Refused("thinking metrics requested but no reasoning close token is known.\n"
                          "       Pass --think-end-token '</think>', --autodetect-think-tokens, or use --thinking.")
        if not flags.start and flags.autodetect != "true":
            out.warn("WARNING: no reasoning open token (--think-start-token). thinking_format_has_open",
                     "         will not be recorded and thinking_format_correct degrades to == has_close.")
    if flags.track == "false" and flags.lengths == "true":
        out.warn("WARNING: --track-thinking-metrics false drops thinking_length_*;",
                 "         --log-length-metrics will only aggregate response_length_*.")


def resolve_sampling(flags, get, out, run):
    """The sampling flags (scripts/common/sampling.py). Thinking and Apertus 2 runs sample (run_lm_eval.sbatch reads
    the defaults from the sampling config and THINK_*); a flag, profile or config given here is resolved now and
    exported explicitly, so the run configuration (and its resume signature) records it. Without any, nothing is
    exported and an earlier run's signature is unchanged. Other runs stay greedy unless --temperature is given."""
    if not any([get("SAMPLING_%s_FLAG" % name) for name in SAMPLERS]
               + [get("SAMPLING_PROFILE_FLAG"), get("SAMPLING_CONFIG_FLAG")]):
        return
    args = [sys.executable, "-m", "scripts.common.sampling", "--framework", "lm_eval", "--format", "shell"]
    if get("SAMPLING_CONFIG_FLAG"):
        args += ["--config", get("SAMPLING_CONFIG_FLAG")]
    if get("SAMPLING_PROFILE_FLAG"):
        args += ["--profile", get("SAMPLING_PROFILE_FLAG")]
    for name, option in zip(SAMPLERS, ("--temperature", "--top-p", "--top-k", "--min-p", "--repetition-penalty")):
        if get("SAMPLING_%s_FLAG" % name):
            args += [option, get("SAMPLING_%s_FLAG" % name)]
    args.append("--warn-greedy" if flags.thinks() else "--no-default")
    done = run(args, stdout=subprocess.PIPE, universal_newlines=True)
    if done.returncode:
        raise Refused(None)  # the helper has said why
    for line in done.stdout.splitlines():
        name, _, value = line.partition("=")
        flags.sampled[name[len("SAMPLING_"):]] = shlex.split(value)[0] if value else ""
    sampled = flags.sampled
    if flags.thinks():
        for name in SAMPLERS:
            out.exports["THINK_" + name] = sampled.get(name) or "none"
        return
    if not sampled.get("TEMPERATURE"):
        raise Refused("sampling flags on a run that does not think need --temperature (without it the run is greedy).")
    out.exports["NOTHINK_TEMPERATURE"] = sampled["TEMPERATURE"]
    out.exports["NOTHINK_TOP_P"] = sampled.get("TOP_P") or "none"
    for name in ("TOP_K", "MIN_P", "REPETITION_PENALTY"):
        if sampled.get(name):
            out.exports["THINK_" + name] = sampled[name]


def refuse_greedy_repeats(flags):
    """Repeats of a greedy run are identical. Greedy = temperature 0, or no sampling on a run that does not think."""
    if not (flags.repeats and int(flags.repeats) > 1):
        return
    temperature = flags.sampled.get("TEMPERATURE", "")
    if flags.thinks():
        greedy = bool(GREEDY.match(temperature))
    else:
        greedy = not (temperature and not GREEDY.match(temperature))
    if greedy:
        raise Refused("--repeats %s on a greedy run gives %s identical samples. Sample: --enable-thinking "
                      "(profile thinking) or --temperature." % (flags.repeats, flags.repeats))


def export_what_was_set(flags, get, out):
    """What the jobs read, only when set: an unset ENABLE_THINKING keeps the template default, an unset
    TRACK_THINKING_METRICS lets the harness derive it."""
    for name, value in (("ENABLE_THINKING", flags.enable), ("THINK_END_TOKEN", flags.end),
                        ("THINK_START_TOKEN", flags.start), ("TRACK_THINKING_METRICS", flags.track),
                        ("REASONING_EFFORT", flags.effort)):
        if value:
            out.exports[name] = value
    if flags.autodetect == "true":
        out.exports["AUTODETECT_THINK_TOKENS"] = "true"
    if flags.lengths == "true":
        out.exports["LOG_LENGTH_METRICS"] = "true"
    if get("TRUST_REMOTE_CODE_FLAG") == "true":
        out.exports["TRUST_REMOTE_CODE"] = "true"


def main():
    env = dict((name, os.environ.get(name, "")) for name in INPUTS)
    try:
        resolution = resolve(env, warn=lambda line: print(line, file=sys.stderr))
    except Refused as refusal:
        if refusal.args[0] is not None:
            print("Error: " + refusal.args[0], file=sys.stderr)
        return 1
    for statement in resolution.statements():
        print(statement)
    return 0


if __name__ == "__main__":
    sys.exit(main())
