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

## What this does not do

It does not strip or measure the reasoning in the answer. Without a server-side reasoning parser
(`reasoning_content`), `<think>...</think>` stays in the text the framework scores; lm-eval's
`--thinking` is the only launcher option that strips it. `--chat-format apertus2` (lm-eval only)
renders thinking itself and refuses these flags.
