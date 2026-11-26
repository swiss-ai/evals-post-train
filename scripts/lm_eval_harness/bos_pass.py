"""The BOS settings of one lm-eval command (a "pass"): what it adds to the model arguments, and its BOS probe.

run_lm_eval.sbatch used to decide these in shell, for each command it builds: one command when the chat
template applies to every task or to none, two with --chat-template-by-type (the generative tasks with the
template, the log-likelihood ones without). `resolve()` is that logic as a function; the job runs it per command as

    PASS_OUT=$(python3 -m scripts.lm_eval_harness.bos_pass --chat-template true|false) || die "..."
    eval "$PASS_OUT"

(captured first, so a failure stops the job) and gets PASS_BOS, PASS_BOS_BY_CLIENT, PASS_MODEL_ARGS (to append
to --model_args), PASS_PROBE (the probe's command line, "" for none) and PASS_NOTES. The settings come from the
environment: INPUTS.

BOS=auto is resolved for each command (bos_default.py): the rule depends on whether the chat template is
applied. bos_by_client applies to commands with a chat template: a rendered template is a complete prompt, so
the server must add nothing to it. Raw text is what the tokenizer's own BOS is for: the server keeps adding it
there. Unset (auto) it is on where it can apply (/v1/completions); --no-bos-by-client (false) turns it off;
--bos-by-client (true) is refused where it cannot apply (job_settings), and said to be ignored for a command
without a template. The BOS probe asks the server what it received and stops before the first task if that is
not what was meant.

Plain Python 3.6 syntax: the job's own `python3` outside the container can be that old.
"""

import argparse
import os
import shlex
import sys

from scripts.lm_eval_harness import bos_default

INPUTS = ("BOS_REQUESTED", "BOS_BY_CLIENT", "TOKENIZER", "API_LM_EVAL_MODEL", "BOS_CHECK", "API_BASE_URL",
          "API_MODEL_NAME")
OUTPUTS = ("PASS_BOS", "PASS_BOS_BY_CLIENT", "PASS_MODEL_ARGS", "PASS_PROBE", "PASS_NOTES")

COMPLETIONS = "local-completions"


def resolve(settings, chat_template, needs_bos=bos_default.needs_bos, warn=None):
    """The command's BOS settings -> {OUTPUTS...}. `needs_bos(tokenizer, chat_template)` decides BOS=auto; if it
    fails, `warn` hears of it and the BOS is left off (a decision the log shows, not a crash)."""
    get = lambda name, default="": settings.get(name) or default
    model = get("API_LM_EVAL_MODEL")
    notes = []

    pass_bos = get("BOS_REQUESTED", "auto")
    if pass_bos == "auto":
        try:
            pass_bos = "true" if needs_bos(get("TOKENIZER"), chat_template) else "false"
        except Exception as error:  # the helper reads the tokenizer's config: whatever goes wrong, say so
            if warn:
                warn("%s: %s" % (type(error).__name__, error))
                warn("WARNING: BOS=auto could not be resolved (bos_default.py failed above), using false")
            pass_bos = "false"
        if pass_bos == "true":
            notes.append("BOS=auto: prepending the BOS token for %s (a tokenizer that adds it itself is left "
                         "alone); BOS=false / --no-bos turns this off" % get("TOKENIZER"))

    requested = get("BOS_BY_CLIENT", "auto")
    by_client = "false"
    if requested != "false":
        if chat_template:
            if requested == "true" or model == COMPLETIONS:
                by_client = "true"
        elif requested == "true":
            notes.append("--bos-by-client applies to commands with a chat template: ignored for this one "
                         "(no chat template)")

    model_args = ""
    # The server adds no special tokens: the BOS the model gets is the one the client wrote.
    if by_client == "true" and model == COMPLETIONS:
        model_args += ",bos_by_client=True"
    if pass_bos == "true" and model == COMPLETIONS:
        notes.append("Adding BOS token.")
        model_args += ",add_bos_token=True"

    probe = ""
    if model == COMPLETIONS and get("BOS_CHECK", "true") != "false":
        flags = (" --chat-template" if chat_template else "") + (" --bos-by-client" if by_client == "true" else "")
        probe = ("cd %s && python -m scripts.lm_eval_harness.bos_probe --url '%s' --model '%s' --tokenizer '%s' "
                 "--bos '%s' --bos-requested '%s'%s" % (os.environ.get("PWD") or os.getcwd(), get("API_BASE_URL"),
                                                       get("API_MODEL_NAME"), get("TOKENIZER"), pass_bos,
                                                       get("BOS_REQUESTED", "auto"), flags))
    return {"PASS_BOS": pass_bos, "PASS_BOS_BY_CLIENT": by_client, "PASS_MODEL_ARGS": model_args,
            "PASS_PROBE": probe, "PASS_NOTES": "\n".join(notes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chat-template", choices=("true", "false"), required=True,
                        help="whether this command applies the chat template")
    args = parser.parse_args()
    settings = dict((name, os.environ.get(name, "")) for name in INPUTS)
    warn = lambda message: print(message, file=sys.stderr)
    resolved = resolve(settings, args.chat_template == "true", warn=warn)
    for name in OUTPUTS:
        print("%s=%s" % (name, shlex.quote(resolved[name])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
