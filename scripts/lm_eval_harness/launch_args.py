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
    "API_MAX_RETRIES_FLAG", "API_NUM_CONCURRENT_FLAG", "API_REQUESTS_PER_MINUTE_FLAG",
    "JUDGE_REQUESTS_PER_MINUTE_FLAG", "SAMPLING_TEMPERATURE_FLAG", "SAMPLING_TOP_P_FLAG", "SAMPLING_TOP_K_FLAG",
    "SAMPLING_MIN_P_FLAG", "SAMPLING_REPETITION_PENALTY_FLAG", "SAMPLING_PROFILE_FLAG", "SAMPLING_CONFIG_FLAG",
    "TRUST_REMOTE_CODE_FLAG", "MODEL_PATH",
)
AMBIENT = (
    "TOKENIZER", "BOS", "ENABLE_THINKING", "API_CHAT_ENDPOINT", "CHAT_FORMAT", "PREFILL_MODE", "ANSWER_FRAME",
    "EVAL_REPEAT_CONFIG", "API_BASE_URL", "API_MODEL_NAME", "API_REQUESTS_PER_MINUTE", "API_MAX_RETRIES",
    "API_NUM_CONCURRENT", "JUDGE_REQUESTS_PER_MINUTE", "JUDGE_MODEL_PREFIX", "USER",
)
INPUTS = FLAGS + AMBIENT

GREEDY = re.compile(r"^0(\.0*)?$")


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


def resolve(env, run=subprocess.run, warn=None):
    """The launcher's variables (`env`) -> Resolution, or Refused. `run` runs the helper commands (the repeats
    policy, the sampling resolution): as subprocesses, like the shell code this replaces. `warn` hears each warning
    when it arises."""
    get = lambda name, default="": env.get(name) or default  # ${NAME:-default}
    out = Resolution(warn)
    umbrella = get("THINKING_UMBRELLA")
    enable = get("ENABLE_THINKING_OVERRIDE")
    end, start = get("THINK_END_TOKEN"), get("THINK_START_TOKEN")
    autodetect, track, lengths = get("AUTODETECT_THINK_TOKENS"), get("TRACK_THINKING_METRICS"), get("LOG_LENGTH_METRICS")
    effort = get("REASONING_EFFORT")
    template = get("CHAT_TEMPLATE_OVERRIDE")  # "", "true", "false", "by-type"

    # --- thinking / reasoning configuration: before dispatch, so the forced chat template reaches every mode
    if track and track not in ("true", "false"):
        raise Refused("--track-thinking-metrics expects 'true' or 'false' (got '%s')" % track)
    # lm_eval splits --model_args on commas, so a comma in a token would inject an extra key.
    for name, value in (("THINK_END_TOKEN", end), ("THINK_START_TOKEN", start)):
        if "," in value:
            raise Refused("%s must not contain a comma (got '%s')" % (name, value))
    if umbrella == "true":
        enable = enable or "true"
        track = track or "true"
        lengths = lengths or "true"
        # A close token must come from somewhere; prefer the user's if they named one.
        if not end and not autodetect:
            autodetect = "true"

    # Two distinct questions (conflating them rejected every "off" switch): THINKING_TOUCHED, any thinking/length
    # flag was passed (it drives the summary line); THINKING_METRICS_ASKED, the user asked to RECORD metrics,
    # which needs a close token and the chat template. --log-length-metrics alone is neither: response_length_*
    # is recorded for any generative response.
    touched = bool(umbrella == "true" or enable or end or start or autodetect == "true" or track or lengths == "true")

    # --repeats / --repeat-metric: layered over --repeat-config (flags win) into one policy file.
    repeats, metric = get("REPEATS_FLAG"), get("REPEAT_METRIC_FLAG")
    pass_at = ""
    if repeats and not re.match(r"^[1-9][0-9]*$", repeats):
        raise Refused("--repeats takes a positive integer (got '%s')" % repeats)
    if metric:
        if not re.match(r"^pass@[0-9]+(,[0-9]+)*$", metric):
            raise Refused("--repeat-metric takes pass@K or pass@K1,K2 (got '%s'); the mean@N is always reported"
                          % metric)
        pass_at = metric[len("pass@"):]
    if repeats or pass_at:
        policy = os.path.join(tempfile.mkdtemp(), "repeat_flags.json")
        args = [sys.executable, "-m", "scripts.lm_eval_harness.repeat_config", "--output", policy]
        if get("EVAL_REPEAT_CONFIG"):
            args += ["--policy", get("EVAL_REPEAT_CONFIG")]
        if repeats:
            args += ["--repeats", repeats]
        if pass_at:
            args += ["--pass-at", pass_at]
        if run(args, stdout=subprocess.DEVNULL).returncode:
            raise Refused(None)  # the helper has said why
        out.exports["EVAL_REPEAT_CONFIG"] = policy

    # apertus2: the harness renders and parses the native format and always reasons, so the template's thinking
    # switches do not apply; the effort sets the thinking level.
    chat_format = get("CHAT_FORMAT_FLAG", "template")
    prefill, frame = get("PREFILL_MODE_FLAG"), get("ANSWER_FRAME_FLAG")
    if chat_format == "template":
        if prefill:
            raise Refused("--prefill-mode only applies with --chat-format apertus2")
        if frame:
            raise Refused("--answer-frame only applies with --chat-format apertus2")
        out.unset += ["CHAT_FORMAT", "PREFILL_MODE", "ANSWER_FRAME"]
        apertus2 = False
    elif chat_format == "apertus2":
        apertus2 = True
        if not (get("CUSTOM_TOKENIZER") or get("TOKENIZER")) or not get("MAX_LENGTH"):
            raise Refused("--chat-format apertus2 needs --tokenizer <the checkpoint's tokenizer> and "
                          "--max-length <the server's --max-model-len>")
        if get("BOS_FLAG") == "true" or get("BOS") == "true":
            raise Refused("--chat-format apertus2 renders the BOS token itself; drop --bos (or BOS=true)")
        endpoint = get("API_ENDPOINT_FLAG")
        if (template in ("false", "by-type") or endpoint == "chat-completions"
                or (not endpoint and get("API_CHAT_ENDPOINT", "false") == "true")):
            raise Refused("--chat-format apertus2 renders every task as chat over /v1/completions; drop "
                          "--no-chat-template, --chat-template-by-type and --api-endpoint chat-completions "
                          "(or API_CHAT_ENDPOINT=true)")
        if umbrella == "true" or enable or get("ENABLE_THINKING") or end or start or autodetect == "true" or track:
            raise Refused("--chat-format apertus2 always reasons and separates the reasoning itself; drop "
                          "--thinking, --enable-thinking/--no-enable-thinking (or ENABLE_THINKING), the think "
                          "tokens and --track-thinking-metrics")
        prefill = prefill or get("PREFILL_MODE", "after_think")
        if prefill not in ("after_think", "empty_think"):
            raise Refused("--prefill-mode must be 'after_think' or 'empty_think' (got '%s')" % prefill)
        frame = frame or get("ANSWER_FRAME", "reply")
        if frame not in ("reply", "claim"):
            raise Refused("--answer-frame must be 'reply' or 'claim' (got '%s')" % frame)
        effort = effort or "medium"
        if effort not in ("low", "medium", "high"):
            raise Refused("--chat-format apertus2 takes --reasoning-effort low, medium or high (got '%s')" % effort)
        lengths = "true"
        out.exports.update({"CHAT_FORMAT": "apertus2", "PREFILL_MODE": prefill, "ANSWER_FRAME": frame})
    else:
        raise Refused("--chat-format must be 'template' or 'apertus2' (got '%s')" % chat_format)

    asked = bool(umbrella == "true" or enable == "true" or track == "true" or end or start or autodetect == "true")

    # The harness renders the chat template client-side, so the switches reach the model. --enable-thinking on its
    # own only switches the template; with --thinking or a think token the reasoning is stripped and measured.
    if umbrella != "true" and not end and not start and autodetect != "true" and not track:
        asked = False
    if enable:
        if template == "false":
            raise Refused("--enable-thinking/--no-enable-thinking are chat-template arguments; drop --no-chat-template")
        if template != "by-type":  # by-type keeps the template on the generative tasks, the only ones these affect
            template = "true"
    # reasoning_effort is a chat-template argument, rendered client-side with the model's template.
    if effort:
        if not re.match(r"^[A-Za-z0-9_-]+$", effort):
            raise Refused("--reasoning-effort expects a level like low, medium or high (got '%s')" % effort)
        if template == "false":
            raise Refused("--reasoning-effort is a chat-template argument; drop --no-chat-template")
        if template != "by-type":
            template = "true"

    # An endpoint is needed; fail here, not after scheduling.
    if not (get("API_BASE_URL_FLAG") or get("API_BASE_URL")):
        raise Refused("--api-base-url <url> is required (or an exported API_BASE_URL)")
    # --model defaults API_MODEL_NAME (run_lm_eval.sbatch) to its own, possibly catalog-prefixed value, which the
    # gateway may not recognize -- require the caller to say explicitly what the server should see.
    if not (get("API_MODEL_NAME_FLAG") or get("API_MODEL_NAME")):
        raise Refused("--api-model-name <name> is required (or an exported API_MODEL_NAME)")
    for flag, value in (("--api-max-retries", get("API_MAX_RETRIES_FLAG")),
                        ("--api-num-concurrent", get("API_NUM_CONCURRENT_FLAG"))):
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
    if (chat_endpoint or "false") == "true" and template in ("false", "by-type"):
        raise Refused("--api-endpoint chat-completions needs the chat template on every task (generative tasks "
                      "only); drop --no-chat-template/--chat-template-by-type or use --api-endpoint completions")

    rpm = get("API_REQUESTS_PER_MINUTE_FLAG") or get("API_REQUESTS_PER_MINUTE")
    retries = get("API_MAX_RETRIES_FLAG") or get("API_MAX_RETRIES")
    concurrent = get("API_NUM_CONCURRENT_FLAG") or get("API_NUM_CONCURRENT")
    judge_rpm = get("JUDGE_REQUESTS_PER_MINUTE_FLAG") or get("JUDGE_REQUESTS_PER_MINUTE")
    judge_prefix = get("JUDGE_MODEL_PREFIX") or get("USER")
    out.variables.update({"API_REQUESTS_PER_MINUTE": rpm, "API_MAX_RETRIES": retries, "API_NUM_CONCURRENT": concurrent,
                          "JUDGE_REQUESTS_PER_MINUTE": judge_rpm, "JUDGE_MODEL_PREFIX": judge_prefix})
    if get("API_BASE_URL_FLAG"):
        out.exports["API_BASE_URL"] = get("API_BASE_URL_FLAG")
    if get("API_MODEL_NAME_FLAG"):
        out.exports["API_MODEL_NAME"] = get("API_MODEL_NAME_FLAG")
    for name, value in (("API_REQUESTS_PER_MINUTE", rpm), ("API_MAX_RETRIES", retries),
                        ("API_NUM_CONCURRENT", concurrent), ("JUDGE_REQUESTS_PER_MINUTE", judge_rpm),
                        ("JUDGE_MODEL_PREFIX", judge_prefix)):
        if value:
            out.exports[name] = value

    # No checkpoint is loaded, so --api-model-name (flag or ambient API_MODEL_NAME) alone identifies the model.
    # Default --model from it so the run can be named when the caller only passed --api-model-name.
    model_path = get("MODEL_PATH") or get("API_MODEL_NAME_FLAG") or get("API_MODEL_NAME")

    if asked:
        # The reasoning tokens live in the chat template, so it has to be rendered.
        if template == "false":
            raise Refused("thinking metrics require the chat template; drop --no-chat-template")
        if template != "by-type":
            template = "true"
        # Without a close token nothing is stripped or recorded -- the run silently produces nothing.
        if not end and autodetect != "true":
            raise Refused("thinking metrics requested but no reasoning close token is known.\n"
                          "       Pass --think-end-token '</think>', --autodetect-think-tokens, or use --thinking.")
        if not start and autodetect != "true":
            out.warn("WARNING: no reasoning open token (--think-start-token). thinking_format_has_open",
                     "         will not be recorded and thinking_format_correct degrades to == has_close.")
    if track == "false" and lengths == "true":
        out.warn("WARNING: --track-thinking-metrics false drops thinking_length_*;",
                 "         --log-length-metrics will only aggregate response_length_*.")

    # --- sampling flags (scripts/common/sampling.py): thinking and apertus2 runs sample (run_lm_eval.sbatch reads the
    # defaults from the sampling config and THINK_*); a flag, profile or config given here is resolved now and
    # exported explicitly, so the run configuration (and its resume signature) records it. Without any, nothing is
    # exported and an earlier run's signature is unchanged. Other runs stay greedy unless --temperature is given.
    samplers = ("TEMPERATURE", "TOP_P", "TOP_K", "MIN_P", "REPETITION_PENALTY")
    sampling_flags = [get("SAMPLING_%s_FLAG" % name) for name in samplers] + [get("SAMPLING_PROFILE_FLAG"),
                                                                              get("SAMPLING_CONFIG_FLAG")]
    sampled = {}
    thinking_run = enable == "true" or apertus2
    if any(sampling_flags):
        args = [sys.executable, "-m", "scripts.common.sampling", "--framework", "lm_eval", "--format", "shell"]
        if get("SAMPLING_CONFIG_FLAG"):
            args += ["--config", get("SAMPLING_CONFIG_FLAG")]
        if get("SAMPLING_PROFILE_FLAG"):
            args += ["--profile", get("SAMPLING_PROFILE_FLAG")]
        for name, option in zip(samplers, ("--temperature", "--top-p", "--top-k", "--min-p", "--repetition-penalty")):
            if get("SAMPLING_%s_FLAG" % name):
                args += [option, get("SAMPLING_%s_FLAG" % name)]
        args.append("--warn-greedy" if thinking_run else "--no-default")
        done = run(args, stdout=subprocess.PIPE, universal_newlines=True)
        if done.returncode:
            raise Refused(None)
        for line in done.stdout.splitlines():
            name, _, value = line.partition("=")
            sampled[name[len("SAMPLING_"):]] = shlex.split(value)[0] if value else ""
        if thinking_run:
            for name in samplers:
                out.exports["THINK_" + name] = sampled.get(name) or "none"
        else:
            if not sampled.get("TEMPERATURE"):
                raise Refused("sampling flags on a run that does not think need --temperature (without it the run "
                              "is greedy).")
            out.exports["NOTHINK_TEMPERATURE"] = sampled["TEMPERATURE"]
            out.exports["NOTHINK_TOP_P"] = sampled.get("TOP_P") or "none"
            for name in ("TOP_K", "MIN_P", "REPETITION_PENALTY"):
                if sampled.get(name):
                    out.exports["THINK_" + name] = sampled[name]

    # Repeats of a greedy run are identical. Greedy = temperature 0, or no sampling on a run that does not think.
    if repeats and int(repeats) > 1:
        temperature = sampled.get("TEMPERATURE", "")
        if thinking_run:
            greedy = bool(GREEDY.match(temperature))
        else:
            greedy = not (temperature and not GREEDY.match(temperature))
        if greedy:
            raise Refused("--repeats %s on a greedy run gives %s identical samples. Sample: --enable-thinking "
                          "(profile thinking) or --temperature." % (repeats, repeats))

    # Export only what was set: unset ENABLE_THINKING keeps the template default; unset TRACK_THINKING_METRICS lets
    # the harness derive it.
    if enable:
        out.exports["ENABLE_THINKING"] = enable
    if end:
        out.exports["THINK_END_TOKEN"] = end
    if start:
        out.exports["THINK_START_TOKEN"] = start
    if autodetect == "true":
        out.exports["AUTODETECT_THINK_TOKENS"] = "true"
    if track:
        out.exports["TRACK_THINKING_METRICS"] = track
    if lengths == "true":
        out.exports["LOG_LENGTH_METRICS"] = "true"
    if effort:
        out.exports["REASONING_EFFORT"] = effort
    if get("TRUST_REMOTE_CODE_FLAG") == "true":
        out.exports["TRUST_REMOTE_CODE"] = "true"

    out.variables.update({
        "ENABLE_THINKING_OVERRIDE": enable, "AUTODETECT_THINK_TOKENS": autodetect, "TRACK_THINKING_METRICS": track,
        "LOG_LENGTH_METRICS": lengths, "REASONING_EFFORT": effort, "CHAT_TEMPLATE_OVERRIDE": template,
        "PREFILL_MODE_FLAG": prefill, "ANSWER_FRAME_FLAG": frame, "MODEL_PATH": model_path,
        "THINKING_TOUCHED": "true" if touched else "false", "THINKING_METRICS_ASKED": "true" if asked else "false",
    })
    return out


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
