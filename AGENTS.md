# Agent guide

Eval infrastructure for LLMs on CSCS Alps, results in W&B. Usage: [README.md](README.md). **Read [CONTRIBUTING.md](CONTRIBUTING.md) first**: tests, adding benchmarks and frameworks, dependencies, design principles.

## Rules

- **Don't submit cluster jobs or write to W&B** unless asked.
- **Chat template:** for post-trained models, generative tasks with it and log-likelihood tasks (multiple choice) without it: `--chat-template-by-type` sorts them. Base models never get it: `--no-chat-template`. lm-eval applies it to every task by default, so a post-trained model's log-likelihood tasks need `--chat-template-by-type` (scored through the template, they measure how likely the model is to start its reply with the answer, not the raw continuation).
- **Thinking models:** no deterministic settings (keep the thinking profile in `configs/sampling_defaults.json`), and generally report repeated scores (`--repeat-config`, [docs/repeated_sampling.md](docs/repeated_sampling.md)).
- **Post-trained Apertus 2** has no Jinja template: `--chat-format apertus2 --tokenizer <dir or name> --max-length <max-model-len>` (lm-eval only; [details](docs/lm-eval-harness.md#apertus-2-native-format)).
- **Judges:** no self-grading (biased); use a separate, fixed judge model.
- **Sampling and thinking flags:** one policy for all launchers, changed in `scripts/common/`.
- **Secrets:** from the environment or `scripts/secrets.env` ([README](README.md#credentials)); never print or commit them.
