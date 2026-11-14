#!/usr/bin/env python3
"""Check that the served model receives the BOS token the client meant it to (openai backend).

The harness decides about the BOS from its own (client) tokenizer, but the server tokenizes the text it
is sent with its own copy. When the two disagree the prompt reaches the model with no BOS, or with two
(a chat template or the harness writes `<s>` into the text and the server adds another), and every score
changes without an error. So before the evaluation this sends the prompts the harness would send, built by
the harness's own model class (with the chat template rendered when the run applies it), and reads back the
prompt tokens the server reports (echo + logprobs, the mechanism log-likelihood tasks use). Two prompts are
checked, a log-likelihood and a generation request, because the harness writes the BOS differently for each.

Outcomes: exactly one BOS is ok. None fails where one was expected (the tokenizer adds one, the run or the
chat template writes one, or BOS=auto should have: see expected_bos) and only warns otherwise: some models
were trained without one (GPT-2 style), and a config can claim a BOS that nothing adds (DeepSeek-V3). Two or
more fail, except in a run with a chat template that does not use --bos-by-client: that doubling (the
template's `<s>` plus the server's) is how such runs have always gone, so it only warns and names the fix.
With --bos-by-client the server is told not to add special tokens, so what it receives must be what the
client wrote. The server's tokens decide, not the tokenizer's add_bos_token flag.
Exit 0: ok, a warning, or a skip (no BOS token, --no-bos without a template, or an endpoint that returns no
prompt tokens). Exit 1: a failure. BOS_CHECK=false skips the probe.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

CONTEXT = "The capital of France is"
CONTINUATION = " Paris"


def leading_bos(tokens: list[str], bos_token: str) -> int:
    """How many BOS tokens the prompt starts with."""
    count = 0
    for token in tokens:
        if token != bos_token:
            break
        count += 1
    return count


def prompt_tokens(
    url: str, model: str, prompt: str, api_key: str = "", timeout: float = 60, add_special_tokens=None
) -> list[str]:
    """The prompt's tokens as the server tokenized it."""
    payload = {"model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0, "echo": True, "logprobs": 1}
    if add_special_tokens is not None:  # --bos-by-client: the requests the harness sends carry it
        payload["add_special_tokens"] = add_special_tokens
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        answer = json.loads(response.read())
    tokens = answer["choices"][0]["logprobs"]["tokens"]
    # echo returns the prompt tokens, then the generated one
    return tokens[:-1] if len(tokens) > 1 else tokens


def probe(
    url: str,
    model: str,
    bos_token: str,
    prompts: dict[str, str],
    api_key: str = "",
    expected: bool = True,
    double_is_error: bool = True,
    add_special_tokens=None,
) -> tuple[bool, str]:
    """(ok, message): each named prompt, as sent, checked on the server. `expected`: a BOS should arrive
    (none is then an error; otherwise a warning). `double_is_error`: two or more is an error (otherwise a
    warning)."""
    lines, ok = [], True
    for name, prompt in prompts.items():
        try:
            tokens = prompt_tokens(url, model, prompt, api_key, add_special_tokens=add_special_tokens)
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            return True, f"BOS check skipped: the endpoint did not return prompt tokens ({exc!r})"
        count = leading_bos(tokens, bos_token)
        shown = " ".join(tokens[:4])
        if count == 1:
            lines.append(f"  {name}: ok, the server received one {bos_token} ({shown} ...)")
        elif count == 0 and not expected:
            lines.append(
                f"  {name}: WARNING, the server received no {bos_token} ({shown} ...). The tokenizer defines one "
                "but nothing adds it; if the model was trained with a start token, pass --bos"
            )
        elif count == 0:
            ok = False
            lines.append(
                f"  {name}: the server received NO {bos_token} ({shown} ...): the model reads the prompt "
                "without its start token. Pass --bos, or make --tokenizer match the served model's"
            )
        elif not double_is_error:
            lines.append(
                f"  {name}: WARNING, the server received {count} {bos_token} tokens ({shown} ...): the chat "
                "template or the harness wrote one and the server added another. --bos-by-client tells the "
                "server to add none, so the model gets one"
            )
        else:
            ok = False
            lines.append(
                f"  {name}: the server received {count} {bos_token} tokens ({shown} ...): the client wrote "
                "one and the server added another. Drop --bos, use --bos-by-client, or make --tokenizer "
                "match the served model's"
            )
    head = "BOS check ok" if ok else "BOS check FAILED (BOS_CHECK=false skips it)"
    return ok, "\n".join([head, *lines])


def harness_prompts(
    model, tokenizer: str, base_url: str, bos: bool, chat_template: bool = False, bos_by_client: bool = False
) -> dict[str, str]:
    """The texts the harness sends: a log-likelihood request (ids decoded back to text) and a generation
    request, built by the harness's own model class so they cannot drift from it. With a chat template the
    context is the rendered template of one user message."""
    from lm_eval.models.openai_completions import LocalCompletionsAPI

    harness = LocalCompletionsAPI(
        model=model,
        base_url=base_url,
        tokenizer_backend="huggingface",
        tokenizer=tokenizer,
        tokenized_requests=False,
        add_bos_token=bos,
        # an older harness ignores it: the probe then reports what the server received
        **({"bos_by_client": True} if bos_by_client else {}),
    )
    if bos_by_client and getattr(harness, "bos_by_client", None) is not True:
        # an older harness swallows the argument: its requests would still let the server add a BOS,
        # while this probe's own request (add_special_tokens=false) would pass
        raise SystemExit(
            "BOS check FAILED: --bos-by-client needs a lm-evaluation-harness with bos_by_client "
            "(this one ignores it): bump requirements/lm-eval-harness.txt, or drop --bos-by-client"
        )
    context = CONTEXT
    if chat_template:
        context = harness.apply_chat_template([{"role": "user", "content": CONTEXT}], add_generation_prompt=True)
    context_enc, continuation_enc = harness._encode_pair(context, CONTINUATION)
    inputs, _, _ = harness.batch_loglikelihood_requests([[(None, context_enc, continuation_enc)]])
    generation = harness._with_bos_text(context) if harness.add_bos_token else context
    return {"log-likelihood": harness.decode_batch(inputs)[0], "generation": generation}


def configured_add_bos_token(tokenizer: str):
    """add_bos_token as the tokenizer's own tokenizer_config.json says it (None: unset or unreadable).
    The loaded tokenizer object does not keep it reliably."""
    from transformers.utils import cached_file

    try:
        path = cached_file(tokenizer, "tokenizer_config.json", _raise_exceptions_for_missing_entries=False)
        return json.loads(open(path).read()).get("add_bos_token") if path else None
    except (OSError, ValueError):
        return None


def expected_bos(
    bos_token: str | None, adds_itself: bool, bos_resolved: bool, bos_requested: str, add_bos_token
) -> bool:
    """Whether the run should deliver a BOS (none then fails; otherwise it only warns).

    Yes when the tokenizer adds one itself, when the run prepends one, or when BOS=auto would have
    (the tokenizer defines a BOS token and says add_bos_token=false) but the host-side lookup could not
    tell: a failed lookup must not hide a missing BOS. A tokenizer that defines a BOS token and says
    nothing is not expected to deliver one (GPT-2 style models were trained without): it is still probed,
    and warned about."""
    if not bos_token:
        return False
    auto_would_prepend = bos_requested == "auto" and add_bos_token is False
    return adds_itself or bos_resolved or auto_would_prepend


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", required=True, help="the /v1/completions endpoint")
    parser.add_argument("--model", required=True, help="the model name sent in requests")
    parser.add_argument("--tokenizer", required=True, help="the client tokenizer (repo id or directory)")
    parser.add_argument("--bos", choices=("true", "false"), default="false", help="BOS as resolved for the harness")
    parser.add_argument(
        "--bos-requested", choices=("auto", "true", "false"), default="auto", help="BOS as the run asked for it"
    )
    parser.add_argument("--chat-template", action="store_true", help="the run applies the chat template")
    parser.add_argument("--bos-by-client", action="store_true", help="the run tells the server to add no BOS")
    parser.add_argument(
        "--api-key", default=os.environ.get("OPENAI_API_KEY", ""), help="bearer token (default: $OPENAI_API_KEY)"
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    ids = tokenizer("a", add_special_tokens=True).input_ids
    adds_itself = bool(ids) and ids[0] == tokenizer.bos_token_id
    add_bos_token = configured_add_bos_token(args.tokenizer)
    if not tokenizer.bos_token:
        print(f"BOS check skipped: {args.tokenizer} defines no BOS token")
        return 0
    expected = expected_bos(
        tokenizer.bos_token, adds_itself, args.bos == "true", args.bos_requested, add_bos_token
    )
    if args.bos_requested == "false" and not adds_itself and not args.chat_template:
        print("BOS check skipped: --no-bos (BOS=false) and the tokenizer adds none")
        return 0
    prompts = harness_prompts(
        args.model, args.tokenizer, args.url, args.bos == "true", args.chat_template, args.bos_by_client
    )
    # a BOS the client wrote (a chat template that starts with one) is expected to arrive
    expected = expected or prompts["generation"].startswith(tokenizer.bos_token)
    ok, message = probe(
        args.url,
        args.model,
        tokenizer.bos_token,
        prompts,
        args.api_key,
        expected,
        double_is_error=args.bos_by_client or not args.chat_template,
        add_special_tokens=False if args.bos_by_client else None,
    )
    print(message, file=sys.stdout if ok else sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
