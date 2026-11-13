#!/usr/bin/env python3
"""Check that the served model receives exactly one BOS token (openai backend, no chat template).

The harness decides about the BOS from its own (client) tokenizer, but the server
tokenizes the text it is sent with its own copy. When the two disagree the prompt reaches
the model with no BOS, or with two (the harness writes `<s>` into the text and the server adds
another), and every score changes without an error. So before the evaluation this sends the
prompts the harness would send, built by the harness's own model class, and reads back the prompt
tokens the server reports (echo + logprobs, the mechanism log-likelihood tasks use): the first
must be the BOS token, the second must not. Two prompts are checked, the way a log-likelihood
request and a generation request are built, because the harness writes the BOS differently for each.

A double BOS always fails. No BOS fails when one was expected (the tokenizer adds one itself, the run
prepends one, or BOS=auto should have: see expected_bos) and only warns otherwise, because a model that
defines a BOS token but was trained without one (GPT-2 style) legitimately gets none; so does a
tokenizer whose config claims it adds one when nothing does (DeepSeek-V3). The server's tokens decide, not
the config flag. Exit 0: one BOS, a warning, a skip (no BOS token, --no-bos, or an endpoint that returns
no prompt tokens). Exit 1: a doubled BOS, or none where one was expected. BOS_CHECK=false skips it.
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


def prompt_tokens(url: str, model: str, prompt: str, api_key: str = "", timeout: float = 60) -> list[str]:
    """The prompt's tokens as the server tokenized it."""
    body = json.dumps(
        {"model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0, "echo": True, "logprobs": 1}
    ).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    tokens = payload["choices"][0]["logprobs"]["tokens"]
    # echo returns the prompt tokens, then the generated one
    return tokens[:-1] if len(tokens) > 1 else tokens


def probe(
    url: str, model: str, bos_token: str, prompts: dict[str, str], api_key: str = "", expected: bool = True
) -> tuple[bool, str]:
    """(ok, message): each named prompt, as sent, checked on the server. `expected`: a BOS should arrive
    (none is then an error; otherwise a warning)."""
    lines, ok = [], True
    for name, prompt in prompts.items():
        try:
            tokens = prompt_tokens(url, model, prompt, api_key)
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
        else:
            ok = False
            lines.append(
                f"  {name}: the server received {count} {bos_token} tokens ({shown} ...): the harness wrote "
                "one and the server added another. Drop --bos, or make --tokenizer match the served model's"
            )
    head = "BOS check ok" if ok else "BOS check FAILED (BOS_CHECK=false skips it)"
    return ok, "\n".join([head, *lines])


def harness_prompts(model, tokenizer: str, base_url: str, bos: bool) -> dict[str, str]:
    """The texts the harness sends: a log-likelihood request (ids decoded back to text) and a generation
    request, built by the harness's own model class so they cannot drift from it."""
    from lm_eval.models.openai_completions import LocalCompletionsAPI

    harness = LocalCompletionsAPI(
        model=model,
        base_url=base_url,
        tokenizer_backend="huggingface",
        tokenizer=tokenizer,
        tokenized_requests=False,
        add_bos_token=bos,
    )
    context_enc, continuation_enc = harness._encode_pair(CONTEXT, CONTINUATION)
    inputs, _, _ = harness.batch_loglikelihood_requests([[(None, context_enc, continuation_enc)]])
    generation = harness._with_bos_text(CONTEXT) if bos else CONTEXT
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
    if args.bos_requested == "false" and not adds_itself:
        print("BOS check skipped: --no-bos (BOS=false) and the tokenizer adds none")
        return 0
    prompts = harness_prompts(args.model, args.tokenizer, args.url, args.bos == "true")
    ok, message = probe(args.url, args.model, tokenizer.bos_token, prompts, args.api_key, expected)
    print(message, file=sys.stdout if ok else sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
