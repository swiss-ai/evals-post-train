# Agent guide

Eval infrastructure for LLMs on CSCS Alps, results in W&B. See [README.md](README.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

## Tests

```bash
uv run --all-extras python -m unittest discover -s tests
```

Offline only. Bug fixes come with a failing-first test; behavior changes update `docs/`. Run `bash -n` on edited shell scripts.

## Rules

- **Chat template:** always for post-trained models, never for base models. lm-eval applies it by default, so base models need `--no-chat-template`.
- **Thinking models:** no deterministic settings (keep the thinking profile in `configs/sampling_defaults.json`), and generally report repeated scores (`--repeat-config`, [docs/repeated_sampling.md](docs/repeated_sampling.md)).
- **Sampling and thinking flags:** one policy for all launchers, changed in `scripts/common/`.
- **Secrets:** from the environment or `scripts/secrets.env`; never print or commit them.
- **Stand-alone and generic:** no coupling to downstream consumers, no per-model special cases.
- **Simple:** one code path over extra modes; small PRs; pin what you add.
- **Shell:** works on macOS bash 3.2 (`${ARR[@]+"${ARR[@]}"}` under `set -u`); quote paths; never `rm -rf` a variable path.

## Trying a change

Use `--limit 5 --no-wandb`; `--gold` for Harbor and EvalScope. Don't submit cluster jobs or write to W&B unless asked.
