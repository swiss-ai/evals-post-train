# Contributing

Keep this repo stand-alone (not coupled to a downstream consumer), and pin what you add (datasets, harness commits, packages) so runs can be repeated.

## Before you open a PR

```bash
uv run --all-extras python -m unittest discover -s tests
```

CI also runs syntax checks, ShellCheck and the Harbor tests ([ci.yml](.github/workflows/ci.yml)). Harbor's tests are skipped locally unless Harbor is installed (Python >= 3.12, `pip install -r requirements/harbor-runtime.lock`).

Smoke-test your change with a few samples (`--limit`) and `--no-wandb`, so nothing reaches the shared W&B project. For Harbor and EvalScope, `--gold` checks environments and scoring without a model.

**Bug fixes** come with a test in `tests/` that fails without the fix (offline `unittest`: no cluster, network or model), and update the docs if documented behavior changes.

## Adding a benchmark to an existing framework

| The benchmark... | Framework |
|---|---|
| is a standard task with a built-in metric | lm-evaluation-harness |
| needs its own scorer, an LLM judge or a simulated user | Inspect AI |
| is agentic coding or terminal work, one container per task | Harbor |

EvalScope covers only SWE-bench Verified and Pro. Results are uploaded by the framework's existing W&B uploader.

**lm-evaluation-harness:** add the task to a task list in `configs/` and its `task/metric` to the matching `*_main_table.txt`. A task missing from [swiss-ai/lm-evaluation-harness](https://github.com/swiss-ai/lm-evaluation-harness) goes there first; then bump the commit in `requirements/lm-eval-harness.txt`. Details: [Adding a New Task](docs/lm-eval-harness.md#adding-a-new-task).

**Inspect AI:** a task in [inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals) needs no code (`--task <name>`); add an example to [docs/inspect-ai.md](docs/inspect-ai.md). Otherwise write a `@task` file, run it by path (`--task path/to/task.py`), and test it. `scripts/aaii/aa_lcr.py` and its `tests/test_aa_lcr.py` are an example with a judge.

**Harbor:** add a pinned entry to `BENCHMARKS` in `scripts/harbor/run.py` (`name@1.0` or `@sha256:...`), list it in `run_harbor_eval.sh`'s header and [docs/harbor.md](docs/harbor.md), and check that `--gold --limit 2` scores 1 on each sandbox it supports. Tasks that need fixing after download get a patch in `scripts/harbor/task_patches.py`, with a test.

## Adding a framework

Follow the layout of the existing four:

- [ ] `scripts/<framework>/run_<framework>_eval.sh` and `.sbatch` (Clariden), taking `--model`, `--api-base-url`, `--name`, `--limit`, reading `TARGET_API_KEY` (default `CSCS_SERVING_API`; keys come from the environment or `scripts/secrets.env` via `scripts/common/secrets.sh`), and honoring `EVAL_PREBUILT_ENV=1` (no install at run time).
- [ ] W&B through `scripts/common/wandb.sh`, with an uploader `scripts/results/update_wandb_<framework>.py`.
- [ ] `requirements/<framework>-runtime.txt`, locked by `scripts/common/lock_requirements.sh`, installed through `scripts/common/locked_venv.sh` and in `containers/Dockerfile.eval`.
- [ ] `scripts/common/k8s_sandbox.py` for `--sandbox k8s`, if tasks run in containers.
- [ ] Tests, plus a CI job if they need their own Python or packages.
- [ ] `docs/<framework>.md`, and the README's tables, Quick Start and Repository Structure.

## Changing a dependency

Edit the `requirements/*.txt`, run `scripts/common/lock_requirements.sh` (needs [uv](https://docs.astral.sh/uv/); `--upgrade-package <name>` moves one package), and commit both files. CI fails on a stale lock.

| File | Installed as |
|---|---|
| `inspect-runtime`, `evalscope-runtime`, `harbor-runtime`, `harbor-adapters` | the `.lock` |
| `lm-eval-runtime` | the `.lock` in the prebuilt image (`openai` backend); the `.txt` by in-job backends, whose containers provide torch |
| `lm-eval-harness.txt` | no lock: the harness commit, installed `--no-deps` (its dependencies go in `lm-eval-runtime.txt`) |
| `apertus2.txt` | no lock: the `apertus-common` commit, installed `--no-deps` for `--chat-format apertus2` runs: into the harness overlay of built environments, and from a wheel the launcher builds (`prepare_apertus_common.sh`) on the prebuilt image (not in the image: private repository); its dependencies are in `lm-eval-runtime.lock` |
