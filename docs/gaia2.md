# Omnilingual GAIA2

[facebook/omnilingual-gaia2](https://huggingface.co/datasets/facebook/omnilingual-gaia2) is Meta's GAIA2 agent benchmark machine-translated into 10 languages (cmn_Hans, deu_Latn, fra_Latn, hin_Deva, ind_Latn, ita_Latn, jpn_Jpan, por_Latn, spa_Latn, tur_Latn), each with 4 subsets (execution, search, ambiguity, adaptability) of 160 scenarios: 6,400 in all. Licence cc-by-nc-4.0: fine for evaluation, not for training data.

Each scenario is a simulated phone of apps (email, calendar, chats, cabs, shopping, files, ...). The agent finishes a user's request by calling the apps' command-line tools while events arrive; an LLM judge then compares the agent's actions and the apps' final state with the reference.

It runs through Meta's own runner, gaia2-runner ([meta-agents-research-environments](https://github.com/facebookresearch/meta-agents-research-environments)/gaia2-cli), with its OpenClaw agent: `scripts/gaia2/run_gaia2_eval.sh` (`--help` lists every option).

## How a run works

- **One container per scenario** holds the app simulator, the agent and the judge. gaia2-runner starts them with podman; here each is a **Kata microVM on an OpenSandbox server** instead (`scripts/gaia2/opensandbox_launcher.py`, the same server as Harbor's `--sandbox opensandbox`): no Docker, no Kubernetes access, only `OPENSANDBOX_API_KEY`. The runner reaches the in-container adapter (port 8090) through a local forward over the server's proxy.
- **The agent and the judge call their models from inside the sandbox.** So `--api-base-url` and `--judge-base-url` must be reachable from there: the SwissAI gateway (`https://api.swissai.svc.cscs.ch/v1`) is; a model served inside a Clariden job is not. The sandbox reaches only those two hosts (`--sandbox-network allowlist`, the default).
- **OpenClaw needs native tool calls**: the endpoint must serve them for the model (vLLM `--enable-auto-tool-choice --tool-call-parser ...`). Chonk on SwissAI-Research has the apertus2 parser.
- **One gaia2-runner config per language and subset**, under `<work-dir>/runs/<language>/<subset>/`. The dataset is downloaded at a pinned revision (`DATASET_REVISION` in `scripts/gaia2/run.py`) into `$GAIA2_HF_CACHE` (default `~/.cache/gaia2/hf_datasets`).
- **`--limit n`**: a random n scenarios of each language and subset (seed 42, the same on every run).

## Results

`<work-dir>/summary.json`, per cell (`cells["deu_Latn/execution"]`), per language, per subset and overall:

| Field | Meaning |
|---|---|
| `success_rate` | passed / attempts: gaia2-runner's own pass rate (avg@k with `--pass-at k`) |
| `pass_at_k` | with `--pass-at k > 1`: scenarios passed in any attempt / scenarios (Meta's headline is pass@3) |
| `n_scenarios`, `n_attempts`, `n_passed`, `n_failed` | counts |
| `n_errors`, `errors` | attempts the judge gave no verdict on (sandbox or adapter failure, agent never ended its turn), with their message |

An error counts as not passed, as in gaia2-runner; the run exits non-zero when there are any, so the score is a lower bound. A scenario that ran out of time is a failure, not an error. gaia2-runner's per-scenario artifacts (events, agent response, judgment, trace, its HTML viewer) are under each cell's `results/`.

W&B (`scripts/results/update_wandb_gaia2.py`): tasks `omnilingual_gaia2` (overall), `omnilingual_gaia2_<language>`, `omnilingual_gaia2_<subset>` and `omnilingual_gaia2_<language>_<subset>`; the main table has the overall and per-language success rates. Not for `--gold` runs.

## The judge

Default: `CSCS-Inference/zai-org/GLM-5.3` on the gateway (`--judge-model`, `--judge-base-url`; key `JUDGE_API_KEY`, default the model's), always with gaia2's multilingual judge prompts (`prompt_version = "omnilingual-gaia2"`: the English ones score correct non-English answers as misses). **Meta's published numbers use gpt-oss-120b with reasoning effort low**; scores with another judge are not directly comparable to theirs.

## Rate limits

Every scenario makes many agent calls and several judge calls. CSCS-Inference models (the default judge among them) are limited to about 15 requests a minute per user; gaia2 has no client-side limiter, and litellm (the judge) retries 429s with backoff. Keep `--concurrency` low for rate-limited models.

## Pins and images

| What | Where |
|---|---|
| gaia2-runner, gaia2-cli, gaia2-core | `requirements/gaia2-runtime.txt` (git installs of one ARE commit; lock `gaia2-runtime.lock`) |
| the same commit | `ARE_COMMIT` in `scripts/gaia2/run.py`, `.github/workflows/gaia2-images.yml` (checked by `tests/test_requirement_locks.py`) |
| the dataset | `DATASET_REVISION` in `scripts/gaia2/run.py` |
| the images | `ghcr.io/swiss-ai/gaia2-oc:<commit>`, `gaia2-oracle:<commit>` (and their base `gaia2-cli:<commit>`), linux/amd64 |

The images are built from gaia2-cli's own Dockerfiles by the "GAIA2 images" workflow (on a change to it, or by hand). The sandboxes pull them without credentials, so the ghcr packages must be public. gaia2-runner picks its container profile from the image name: keep `gaia2-oc` / `gaia2-oracle` in any other `--agent-image` / `--oracle-image`. Two things Meta's Dockerfiles leave unpinned: the base image downloads the `gaia2_filesystem` dataset at build time and installs `litellm` unpinned, so a rebuild at the same commit can differ; the tag is the build that ran.

## Examples

```bash
# The oracle (reference actions replayed, no model): checks the sandboxes, apps and judge; all should pass.
scripts/gaia2/run_gaia2_eval.sh --gold --language deu_Latn --limit 2 --no-wandb

# A smoke test with Chonk on the gateway.
scripts/gaia2/run_gaia2_eval.sh --model SwissAI-Research/swiss-ai/chonk-lc-16k-v0.2-sft \
  --api-base-url https://api.swissai.svc.cscs.ch/v1 --language deu_Latn --limit 5 --no-wandb

# German, French, Italian, all four subsets, pass@1, on Clariden (a CPU job).
sbatch scripts/gaia2/run_gaia2_eval.sbatch --model SwissAI-Research/swiss-ai/chonk-lc-16k-v0.2-sft \
  --api-base-url https://api.swissai.svc.cscs.ch/v1 --language deu_Latn,fra_Latn,ita_Latn
```

The prebuilt image (`containers/Dockerfile.eval`) has gaia2-runner in `GAIA2_VENV`; with `EVAL_PREBUILT_ENV=1` nothing is installed at run time.
