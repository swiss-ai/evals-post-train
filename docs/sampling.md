# Sampling

One policy sets how the model under test is sampled, in every framework launcher:
`configs/sampling_defaults.json`, read by `scripts/common/sampling.py`. Flags override it.

```json
{
  "profiles": {"thinking": {"temperature": 0.6, "top_p": 0.95}, "greedy": {"temperature": 0.0}},
  "frameworks": {"lm_eval": "thinking", "inspect": "thinking", "evalscope": "thinking",
                 "harbor": "thinking", "tau2": null}
}
```

A sampler that is not set stays out of the request, so the server's own default applies (top-k,
min-p and the repetition penalty are unset by default). The sampling applies to the model under
test only, not to judges, graders or user simulators.

## Flags

Every launcher takes the same flags:

| Flag | Meaning |
|---|---|
| `--temperature V` | Sampling temperature (`0` is greedy) |
| `--top-p V` | Nucleus sampling, in (0, 1] |
| `--top-k V` | Keep the k most likely tokens, an integer >= 1 |
| `--min-p V` | Drop tokens below `V` times the best token's probability |
| `--repetition-penalty V` | Divide (positive) or multiply (negative) the logit of tokens already seen, `V` > 1 penalises |
| `--sampling-profile P` | A profile of the config (`thinking`, `greedy`) instead of the framework's default |
| `--sampling-config F` | Another config file instead of `configs/sampling_defaults.json` |

`none` leaves one out (`--top-p none`). Precedence, strongest first: the flags, `--sampling-profile`,
the framework's default profile. The resolved values are printed in the run's log. Temperature 0
on a thinking run prints a warning: reasoning models loop and truncate under greedy decoding.

## By framework

| Framework | Default | How the values reach the model |
|---|---|---|
| lm-eval | Thinking runs (`--thinking`, `--enable-thinking`, `--chat-format apertus2`): `thinking`. Other runs stay greedy (task yaml), so the base-model and pretrain suites do not change. `--temperature` turns sampling on for them. | `--gen_kwargs`; flags are exported as `THINK_*` (`NOTHINK_*`) and recorded in the run configuration |
| Inspect | `thinking` | `--temperature`, `--top-p`, `--top-k`; min-p and repetition penalty in `--extra-body` (a literal `-- --temperature ...` still wins) |
| EvalScope | `thinking` (EvalScope's own default is greedy) | `generation_config`: `--generation-config` over the defaults, the flags over both; min-p in `extra_body` |
| Harbor | `thinking`, terminus-2 only | `--ak temperature=...` and `--ak llm_kwargs=...` (`extra_body`); your own `--agent-kwarg` wins over the defaults, the flags win over both. Other agents get no defaults and refuse the flags |
| tau2 | none: no setting is sent unless you pass one (tau2-bench's own default is temperature 0, but the script's `--agent-llm-args` replaces it) | merged over `--agent-llm-args`: `temperature`, `top_p`, the rest in `extra_body` |

Not covered: the GDPval runner (`scripts/aaii/run_gdpval.sh`) fixes its own `max_tokens`.

## Other settings

`MAX_NEW_TOKENS` (32768 for thinking and reasoning-effort runs), `API_NUM_CONCURRENT` and
`API_MAX_RETRIES` of the lm-eval `openai` backend stay environment variables
([details](lm-eval-harness.md)). Repeated sampling (mean@k, pass@k) is a separate policy.
