"""Sampling settings of the model under test, one policy for every framework launcher.

The defaults live in configs/sampling_defaults.json (profiles, the profile of each thinking mode,
and the profile each framework uses when no mode is given). Precedence, strongest first: explicit
flags (or --env-prefix variables), --profile, the profile of the thinking mode the thinking flags
ask for, the framework's default profile. A sampler that is not set stays out of the request, so
the server's own default applies. "none" removes one explicitly.

The mode comes from the same flags every launcher already takes (scripts/common/thinking.py):
--enable-thinking is "thinking" and --no-enable-thinking is "non_thinking" (the switch wins over an
effort); without a switch, --reasoning-effort none is "non_thinking" and any other level
"thinking"; neither flag means no mode.

    python -m scripts.common.sampling --framework inspect --top-p none --format json
"""

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

SAMPLERS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty")
FRAMEWORKS = ("lm_eval", "inspect", "evalscope", "harbor", "tau2")
MODES = ("thinking", "non_thinking")
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "sampling_defaults.json"


def _number(name, value):
    """Validate one sampler value; None means 'leave it out'."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number or null (got {value!r})")
    if name == "temperature" and value < 0:
        raise ValueError(f"temperature must be >= 0 (got {value})")
    if name == "top_p" and not 0 < value <= 1:
        raise ValueError(f"top_p must be in (0, 1] (got {value})")
    if name == "top_k":
        if int(value) != value or value < 1:
            raise ValueError(f"top_k must be an integer >= 1 (got {value})")
        return int(value)
    if name == "min_p" and not 0 <= value <= 1:
        raise ValueError(f"min_p must be in [0, 1] (got {value})")
    if name == "repetition_penalty" and value <= 0:
        raise ValueError(f"repetition_penalty must be > 0 (got {value})")
    return value


def parse_value(name, text):
    """A command-line or environment value: a number, or 'none' to omit the sampler."""
    if text is None:
        return None
    if str(text).strip().lower() == "none":
        return None
    try:
        number = float(text)
    except ValueError:
        raise ValueError(f"{name} must be a number or 'none' (got {text!r})") from None
    return _number(name, int(number) if name == "top_k" and number == int(number) else number)


def thinking_mode(enable_thinking=None, reasoning_effort=None):
    """The thinking mode the thinking flags ask for: "thinking", "non_thinking", or None (unset).

    The on/off switch wins over the effort; without a switch, an effort of "none" is non-thinking
    and any other level is thinking."""
    if enable_thinking is not None:
        return "thinking" if enable_thinking else "non_thinking"
    if reasoning_effort is not None:
        return "non_thinking" if reasoning_effort == "none" else "thinking"
    return None


def read_config(path=None):
    config = json.loads(Path(path or DEFAULT_CONFIG).read_text())
    if not isinstance(config, dict) or set(config) - {"profiles", "modes", "frameworks"}:
        raise ValueError("Sampling config accepts profiles, modes and frameworks only")
    profiles = config.get("profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError("profiles must be an object")
    for name, profile in profiles.items():
        if not isinstance(profile, dict) or set(profile) - set(SAMPLERS):
            raise ValueError(f"profile {name!r} accepts {', '.join(SAMPLERS)} only")
        for key, value in profile.items():
            _number(key, value)
    modes = config.get("modes", {})
    if not isinstance(modes, dict) or set(modes) - set(MODES):
        raise ValueError(f"modes accepts {', '.join(MODES)} only")
    for name, profile in modes.items():
        if profile not in profiles:
            raise ValueError(f"mode {name!r} names an unknown profile {profile!r}")
    frameworks = config.get("frameworks", {})
    if not isinstance(frameworks, dict) or set(frameworks) - set(FRAMEWORKS):
        raise ValueError(f"frameworks accepts {', '.join(FRAMEWORKS)} only")
    for name, profile in frameworks.items():
        if profile is not None and profile not in profiles:
            raise ValueError(f"framework {name!r} names an unknown profile {profile!r}")
    return config


def resolve(
    framework, *, config=None, profile=None, overrides=None, use_default=True, enable_thinking=None, reasoning_effort=None
):
    """The sampling settings to send, as {sampler: number}; unset samplers are absent.

    Profile precedence: `profile`, the thinking mode the thinking flags ask for, the framework's
    default (only with use_default)."""
    if framework not in FRAMEWORKS:
        raise ValueError(f"unknown framework {framework!r}")
    config = config or read_config()
    if profile is not None and profile not in config.get("profiles", {}):
        raise ValueError(f"unknown sampling profile {profile!r} (have {sorted(config['profiles'])})")
    mode = thinking_mode(enable_thinking, reasoning_effort)
    if profile is None and mode is not None:
        profile = config.get("modes", {}).get(mode)
        if profile is None:
            raise ValueError(f"the sampling config maps no profile to the {mode!r} mode")
    chosen = profile if profile is not None else (config["frameworks"].get(framework) if use_default else None)
    values = dict(config["profiles"][chosen]) if chosen else {}
    for key, value in (overrides or {}).items():
        if key not in SAMPLERS:
            raise ValueError(f"unknown sampler {key!r}")
        values[key] = value
    return {key: value for key, value in values.items() if value is not None}


def litellm_kwargs(values):
    """litellm call kwargs: temperature and top_p are its own; the rest travel in extra_body, which
    an OpenAI-compatible server reads (litellm rejects params the openai provider does not know)."""
    kwargs = {key: values[key] for key in ("temperature", "top_p") if key in values}
    body = {key: values[key] for key in ("top_k", "min_p", "repetition_penalty") if key in values}
    return kwargs | ({"extra_body": body} if body else {})


def greedy_warning(values):
    if values.get("temperature") == 0:
        return (
            "WARNING: temperature 0 is greedy decoding; reasoning models degrade under it "
            "(loops, truncated traces). Use it only on purpose."
        )
    return None


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--framework", required=True, choices=FRAMEWORKS)
    parser.add_argument("--config", help="sampling config (default: configs/sampling_defaults.json)")
    parser.add_argument("--profile", help="profile to use instead of the framework's default")
    parser.add_argument("--no-default", action="store_true", help="only explicit settings, no framework default")
    switch = parser.add_mutually_exclusive_group()
    switch.add_argument("--enable-thinking", dest="enable_thinking", action="store_true", default=None,
                        help="thinking mode: use the profile of the thinking mode")
    switch.add_argument("--no-enable-thinking", dest="enable_thinking", action="store_false",
                        help="non-thinking mode: use the profile of the non_thinking mode")
    parser.add_argument("--reasoning-effort", dest="reasoning_effort",
                        help="'none' is the non_thinking mode, any other level the thinking mode (unless a switch is given)")
    parser.add_argument("--warn-greedy", action="store_true", help="warn on stderr when temperature is 0")
    parser.add_argument("--env-prefix", help="also read overrides from <PREFIX>TEMPERATURE, <PREFIX>TOP_P, ...")
    for name in SAMPLERS:
        parser.add_argument("--" + name.replace("_", "-"), dest=name, help="number, or 'none' to omit")
    parser.add_argument("--format", choices=("json", "shell", "summary", "litellm"), default="json")
    parser.add_argument("--output", type=Path, help="also write the resolved settings here (JSON)")
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    try:
        overrides = {}
        for name in SAMPLERS:
            text = getattr(args, name)
            if text is None and args.env_prefix:
                text = os.environ.get(args.env_prefix + name.upper()) or None
            if text is not None:
                overrides[name] = parse_value(name, text)
        values = resolve(
            args.framework,
            config=read_config(args.config),
            profile=args.profile,
            overrides=overrides,
            use_default=not args.no_default,
            enable_thinking=args.enable_thinking,
            reasoning_effort=args.reasoning_effort,
        )
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc
    if args.warn_greedy and (warning := greedy_warning(values)):
        print(warning, file=sys.stderr)
    serialized = json.dumps(values, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    if args.format == "shell":
        for name in SAMPLERS:
            print(f"SAMPLING_{name.upper()}={shlex.quote(str(values[name])) if name in values else ''}")
    elif args.format == "litellm":
        print(json.dumps(litellm_kwargs(values), sort_keys=True))
    elif args.format == "summary":
        print(", ".join(f"{name}={values.get(name, '<unset>')}" for name in SAMPLERS))
    else:
        print(serialized)


if __name__ == "__main__":
    main()
