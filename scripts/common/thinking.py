"""Thinking and reasoning effort of the model under test, one vocabulary for every framework launcher.

The flags are the ones launch_evaluations.sh (lm-eval) already has, so a run is asked for the same
thing wherever it goes:

    --enable-thinking / --no-enable-thinking   the chat template's `enable_thinking` switch
    --reasoning-effort <level>                 the chat template's `reasoning_effort` (and the request's own)

Nothing set means nothing sent: the model keeps its server-side default (a Qwen3 template thinks, an
opt-in template does not). A setting reaches the server twice for the effort, once as the request's
`reasoning_effort` and once as the template kwarg, because servers differ in which one they read; the
on/off switch is a template kwarg only.

This module resolves and validates the settings and renders them for each framework's own request
shape. It does not strip or measure the reasoning in the answer: that is the framework's job.
lm-eval's launcher keeps its own implementation of the same two flags.

    python -m scripts.common.thinking --reasoning-effort low --format litellm
"""

import argparse
import json
import shlex

EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def resolve(enable_thinking=None, reasoning_effort=None):
    """The settings to send, as {name: value}; an unset one is absent."""
    values = {}
    if enable_thinking is not None:
        if not isinstance(enable_thinking, bool):
            raise ValueError(f"enable_thinking must be true or false (got {enable_thinking!r})")
        values["enable_thinking"] = enable_thinking
    if reasoning_effort is not None:
        if reasoning_effort not in EFFORTS:
            raise ValueError(f"reasoning effort must be one of {', '.join(EFFORTS)} (got {reasoning_effort!r})")
        values["reasoning_effort"] = reasoning_effort
    return values


def chat_template_kwargs(values):
    """The chat template's own arguments: {"enable_thinking": ..., "reasoning_effort": ...}."""
    return {key: values[key] for key in ("enable_thinking", "reasoning_effort") if key in values}


def extra_body(values):
    """The request-body fragment an OpenAI-compatible server (vLLM, SGLang) reads."""
    kwargs = chat_template_kwargs(values)
    return {"chat_template_kwargs": kwargs} if kwargs else {}


def litellm_kwargs(values):
    """litellm call kwargs: reasoning_effort is its own, the template switches travel in extra_body."""
    kwargs = {"reasoning_effort": values["reasoning_effort"]} if "reasoning_effort" in values else {}
    body = extra_body(values)
    return kwargs | ({"extra_body": body} if body else {})


def merge_extra_body(base, values):
    """`base` (a request's extra_body or None) with the thinking settings merged in, key by key.

    The settings win over a chat_template_kwargs entry of the same name; the other entries stay."""
    body = dict(base or {})
    kwargs = chat_template_kwargs(values)
    if kwargs:
        body["chat_template_kwargs"] = {**(body.get("chat_template_kwargs") or {}), **kwargs}
    return body


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    switch = parser.add_mutually_exclusive_group()
    switch.add_argument("--enable-thinking", dest="enable_thinking", action="store_true", default=None)
    switch.add_argument("--no-enable-thinking", dest="enable_thinking", action="store_false")
    parser.add_argument("--reasoning-effort", dest="reasoning_effort")
    parser.add_argument("--format", choices=("json", "shell", "summary", "litellm", "extra-body"), default="json")
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    try:
        values = resolve(args.enable_thinking, args.reasoning_effort)
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc
    if args.format == "shell":
        enable = values.get("enable_thinking")
        print(f"THINKING_ENABLE={'' if enable is None else str(enable).lower()}")
        print(f"THINKING_EFFORT={shlex.quote(values.get('reasoning_effort', ''))}")
        # The body without its braces, to sit in a hand-built JSON object next to the sampling keys.
        fragment = json.dumps(extra_body(values))[1:-1] if values else ""
        print(f"THINKING_EXTRA_BODY={shlex.quote(fragment)}")
    elif args.format == "litellm":
        print(json.dumps(litellm_kwargs(values), sort_keys=True))
    elif args.format == "extra-body":
        print(json.dumps(extra_body(values), sort_keys=True))
    elif args.format == "summary":
        print(", ".join(f"{key}={values.get(key, '<default>')}" for key in ("enable_thinking", "reasoning_effort")))
    else:
        print(json.dumps(values, sort_keys=True))


if __name__ == "__main__":
    main()
