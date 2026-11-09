# Thinking and reasoning effort

Every launcher takes the same two settings for the model under test, resolved and validated by
`scripts/common/thinking.py`:

| Flag | Meaning |
|---|---|
| `--enable-thinking` / `--no-enable-thinking` | The chat template's `enable_thinking` switch |
| `--reasoning-effort L` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh` or `max` |

Nothing set means nothing sent: the model keeps its server-side default (a Qwen3 template thinks,
an opt-in template does not). The settings apply to the model under test only, not to judges,
graders or user simulators.

## By framework

| Framework | How the settings reach the model |
|---|---|
| lm-eval | Its own `--enable-thinking`, `--no-enable-thinking`, `--reasoning-effort` and `--thinking` ([lm-eval-harness.md](lm-eval-harness.md)) |
| Inspect | `--reasoning-effort L` for Inspect, plus `chat_template_kwargs` in `--extra-body` (merged with the sampling keys; a literal `-- --extra-body` replaces them) |
| EvalScope | `reasoning_effort` and `chat_template_kwargs` in `extra_body` of the generation config; they win over `--generation-config`. Not with `--completions` |
| Harbor | terminus-2 only: its `reasoning_effort` kwarg and `chat_template_kwargs` in `llm_kwargs.extra_body`; they win over your `--agent-kwarg` |
| tau2 | The agent model's litellm args (`reasoning_effort`, `extra_body`), merged over `--agent-llm-args`; the user simulator never gets them |
| GDPval | The client's `reasoning_effort` and `extra_body` for the model under test; the judges never get them |

The effort is sent twice, as the request's `reasoning_effort` and as the template kwarg, because
servers differ in which one they read. The on/off switch is a template kwarg only.

## Apertus 2 on `/v1/chat/completions` (temporary)

lm-eval renders Apertus 2 itself (`--chat-format apertus2`). The other frameworks send ordinary chat
requests, so for them the **server** has to render the Apertus 2 format and split the reasoning from
the reply. That is the vLLM plugin of `apertus-common` PR #5, which is **not merged yet**:
`requirements/apertus2-vllm-plugin.txt` pins the PR head we tested. When the PR merges, replace
that commit with the merged one (or drop the file if the server image ships the plugin).

1. **Build the wheel** where your git access is (the repository is private):
   `scripts/lm_eval_harness/prepare_apertus_common.sh requirements/apertus2-vllm-plugin.txt`
   prints the directory it keeps the wheel in, under `$EVAL_ENV_CACHE_ROOT`.
2. **Serve the model with the plugin.** The `vllm-apertus2-cuda` image does not contain it, so every
   node installs the wheel before `vllm serve`, and the server gets the four `apertus2` options.
   With model-launch's `sml advanced`, for a launch like `examples/clariden/cli/swiss-ai/apertus2/chonk/`:

   ```bash
   sml advanced ... \
     --pre-launch-cmds "pip install -q --no-deps --force-reinstall --ignore-requires-python <wheel dir>/apertus_common-*.whl" \
     --framework-args "--model <checkpoint> ... --tokenizer-mode apertus2 --reasoning-parser apertus2 \
       --enable-auto-tool-choice --tool-call-parser apertus2"
   ```

   The wheel directory must be one the job's container mounts. Without the install the server exits
   at startup with `invalid tool call parser: apertus2`. `sml advanced` does not set a QoS: for
   `highprio`, render the job with `--output-script`, add the `#SBATCH --qos` line, and `sbatch` it.
3. **Point the framework at it** like any served model: `--model <served name> --api-base-url ...`.

The server answers with the reasoning in a separate `reasoning` field. Notes:

- Apertus 2 always reasons: `--no-enable-thinking` did not change the output length, so the thinking
  flags only pick the sampling profile ([sampling](sampling.md)) for this model.
- It thinks a lot even on trivial prompts, so give a large token budget. The Chonk launch has a
  context of 16384; an empty `content` with `finish_reason: length` means the budget ran out while
  it was still thinking.

Tested: Inspect `gsm8k --limit 5` against Chonk (5 of 5). Not tested: EvalScope, Harbor, tau2, and
whether the scores match lm-eval's `--chat-format apertus2` path.

## What this does not do

It does not strip or measure the reasoning in the answer. Without a server-side reasoning parser
(`reasoning_content`), `<think>...</think>` stays in the text the framework scores; lm-eval's
`--thinking` is the only launcher option that strips it. `--chat-format apertus2` (lm-eval only)
renders thinking itself and refuses these flags.
