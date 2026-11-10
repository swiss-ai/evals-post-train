# Queue acceptance checks on Clariden

SSH is enough: run the commands below in your Clariden SSH terminal. A VS Code
tunnel is optional. These are submission templates, not already submitted jobs.

The batch script runs API clients and scorers; it does not load checkpoint weights
or start GPU servers. Hosted inference/judges must be reachable from compute
nodes, not just through `localhost` on your laptop/login node. An allocation on
Clariden may reserve a whole GPU node even for these CPU-side clients; confirm
the site's resource policy rather than assuming free CPU-only capacity.

## Stage a verified runtime and evidence

Apply the complete implementation patch to your authorized cluster checkout.
Keep the source unchanged while jobs using it are queued/running. Prepare the
existing evaluation environment using `scripts/lm_eval_harness/prepare_eval_env.sbatch`
and the exact harness pin in `requirements/lm-eval-harness.txt`, or use an
equivalent already prepared evaluation image. Follow [the environment guide](lm-eval-harness.md).
The generic handbook PyTorch image alone is not this evaluator runtime.

Clariden is ARM64. Do not upload an x86/Windows virtualenv or blindly install the
local CPU validation lock. Select ARM-compatible pinned packages and create
plans in the actual cluster runtime. Exact Python patch/package/source hashes
are checked; cluster plans and judge locks will differ from local test plans.

Supply an approved EDF file/environment. Its identity mounts must expose the
checkout, expert configuration, frozen inputs/oracle, scorer assets, environment
manifest/base/overlay, HF cache and output at their actual cluster paths. Home
mounts alone do not expose capstor or scratch. Prefer a team container over
building a new one; NGC access does not prove that the evaluation image exists.

Keep immutable evidence/results on permanent storage, for example an approved
project directory or `/capstor/store/cscs/swissai/infra01/users/$USER/`.
Use scratch only for regenerable caches; it can be purged. NLTK assets require a
private Linux directory. Do not apply broad group/world permission commands to
credentials, private scorer assets or directories you did not create.

Prepare licensed data/cache/assets separately using the existing `prepare`
command. For each benchmark create independent acceptance configs/oracles as
described in [helpfulness.md](helpfulness.md#separate-acceptance-checks).
The example TOML still has intentionally invalid placeholders. Original per-case
labels, immutable model metadata and justified tolerances are not downloaded or
invented by the batch script.

## Submit one benchmark/scope per job

### Prepare the missing runtime and first genuine regression fixture

`scripts/helpfulness/prepare_regression.sbatch` runs the existing immutable
environment builder inside the verified image, with
`requirements/helpfulness-runtime.txt` and the repository's exact harness commit.
It installs into a new private venv/overlay, never the shared container image.
It explicitly requests `EVAL_ENV_ISOLATED=true`: no system/user site-packages,
no inherited `PYTHONPATH`/`PYTHONHOME`, and no container Torch library paths.
The scorer dependency profile pins CPU-only Torch and excludes vLLM, CUDA
kernels and other in-process serving packages. The shared builder's default for
existing GPU-serving callers is unchanged; isolated mode uses a different cache
identity. `runtime-isolation.json` verifies the actual venv configuration,
package search paths, absence of serving packages and CPU-only Torch location.

The job retains the image SHA-256, full resolved `requirements.frozen.txt`,
`pip check` output, package/source fingerprint and implementation-test results.
The Python interpreter/OS still come from the image, so reproduction requires
that exact image, not just its mutable file name. No package changes occur after
freezing. An incompatible package or skipped native integration fails preparation.

The same job acquires genuine IFEval prompts and the released November 2023
GPT-4 responses at Google Research commit
`b7f222394aeb06af2f8f8511b910bf8a413f9339`. It preserves license notices and source
hashes, then runs the **original 2023 Google CLI in a separate process**.
Its raw strict/loose outputs and printed aggregates become the independent oracle;
our scorer is not used to generate expected decisions or aggregate scores.
All 541 released cases are required. Scorer Python RNG and langdetect seeds are
fixed at zero on both sides and recorded; this is a frozen-runtime scorer
regression, not a reconstruction of every original paper environment detail.
The released literal inputs do not establish GPT-4's original chat template.

```bash
# Create a fresh private permanent run directory and logs before submission.
export EVAL_REGRESSION_ROOT=/ABSOLUTE/FRESH/PERMANENT/regression-001
export EVAL_REPO_ROOT=/ABSOLUTE/STAGED/evals-post-train
export EVAL_CONTAINER_CONFIG=/ABSOLUTE/APPROVED/evaluation.toml
export EVAL_SUITE_FILE=/ABSOLUTE/EXPERT/evals/helpfulness.toml
mkdir -p "$EVAL_REGRESSION_ROOT/logs"
cd "$EVAL_REGRESSION_ROOT"
sbatch --parsable --account=infra01 --partition=normal \
  "$EVAL_REPO_ROOT/scripts/helpfulness/prepare_regression.sbatch"
```

`workflow-status.json` distinguishes runtime installation, scorer-asset staging,
implementation checks, original-oracle preparation and actual scorer replay.
The final comparison is in `ifeval/report/report.json`. A scorer mismatch is a
failed scientific gate, not permission to change expected answers or tolerances.
Original/reference files stay outside git.

The preparation preflight validates the staged six-family expert suite before
installing anything. TruthfulQA generation and SimpleQA are excluded by request;
optional library adapters/tests are retained but are not scheduled evaluations.
Harness revisions are read using whitespace-safe Python parsing, and the shared
builder strips CR/LF from inherited revision values. Linux staging must normalize
shell scripts **and requirement text files**, not just `.sh` files.

This bootstraps **IFEval replay only**, not six completed benchmark approvals.
Other source corpora, published-model deployment metadata, independently sourced
judge labels and justified regeneration/calibration policies remain separate
prerequisites. No checkpoint server is launched. Missing proprietary baseline
weights/judges cannot be manufactured by installing more packages.

### Submit an already prepared benchmark

Replace all placeholders; export actual endpoint credentials through your site's
approved protected mechanism, never paste them into these scripts or git.

```bash
cd /ABSOLUTE/CLUSTER/evals-post-train
export EVAL_REPO_ROOT="$PWD"
export EVAL_CONTAINER_CONFIG=/ABSOLUTE/APPROVED/evaluation.toml
# Trusted output of the existing prepare_eval_env.sbatch job:
export EVAL_ENV_MANIFEST=/ABSOLUTE/PREPARED/environment.sh
# Alternatively omit EVAL_ENV_MANIFEST and set an installed container Python:
# export EVAL_PYTHON=/ABSOLUTE/PREPARED/venv/bin/python
export HF_HOME=/ABSOLUTE/STAGED/huggingface-cache
RUN_ROOT=/ABSOLUTE/PERMANENT/helpfulness-acceptance/run-001
mkdir -p logs

sbatch --parsable --account=infra01 --partition=normal --time=02:00:00 \
  scripts/helpfulness/run_acceptance.sbatch \
  replay /ABSOLUTE/CONFIG/ifeval-replay.toml "$RUN_ROOT/ifeval-replay"

sbatch --parsable --account=infra01 --partition=normal --time=02:00:00 \
  scripts/helpfulness/run_acceptance.sbatch \
  regenerate /ABSOLUTE/CONFIG/ifeval-regeneration.toml \
  "$RUN_ROOT/ifeval-regeneration" /ABSOLUTE/CONFIG/reference-endpoint.toml

sbatch --parsable --account=infra01 --partition=normal --time=02:00:00 \
  scripts/helpfulness/run_acceptance.sbatch \
  calibrate /ABSOLUTE/CONFIG/simpleqa-calibration.toml \
  "$RUN_ROOT/simpleqa-calibration" /ABSOLUTE/CONFIG/judge-endpoint.toml
```

Account/partition are deliberately not hardcoded in the batch file. Verify them
with `sacctmgr show associations user=$USER` and `sinfo` before submission.
The times/resources above are examples, not runtime estimates; LC bootstrapping
can be expensive. Use a short `debug` allocation only for smoke checks within
its current limit. Limited data is diagnostic, not published-score acceptance.

Each job creates `plan.json` in its runtime, optionally produces `candidate.json`,
then writes `report/report.json` and per-case evidence. Calibration freezes
`judge-lock.json` only after a complete passing report. A wrong lane, missing
input, failed producer or failed gate exits nonzero. Existing output directories
are rejected atomically; choose a new experiment directory rather than overwrite.
No installs, dataset downloads, model launches or retries happen in this script.

Repeat for the available benchmark configs; do not silently skip unavailable
benchmarks. Launch only as many concurrent clients as your endpoint quota permits.
These lanes are scientifically distinct: calibration is not required for
mechanical replay, and matching a score is not proof of original equivalence.

For a dependent regeneration job, capture the corresponding replay's ID and
add `--dependency=afterok:JOB_ID`. Use this only for a real prerequisite, not an
arbitrary chain through all benchmarks. A failed gate leaves dependent jobs
waiting; inspect them and explicitly cancel the affected IDs when appropriate.

Original likelihood/encoder-decoder producers are not replaced by chat calls.
For these, submit `command plan --config ... --output ...`, run the appropriate
pinned original/native producer under its own verified GPU environment, then
submit `command run --plan ... --candidate ... --output ...`. The `command` lane
passes arguments to the existing acceptance CLI without generating another plan.
Verify that original producer outputs identify that exact sealed plan.

## Inspect results and evaluate teachers

```bash
squeue --me
scontrol show job JOB_ID
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed
tail -n 60 logs/helpfulness-acceptance_JOB_ID.err
```

Slurm `COMPLETED` plus exit code zero is necessary, not the scientific report.
Inspect the report's status, per-case differences, denominators, protocols,
coverage and uncertainty. Failed/incomplete jobs retain evidence for diagnosis.
Retain inputs/oracle/config/plan/candidate/report/lock together.

Once the needed judges pass, configure `judge.require_calibration` and the
per-benchmark lock paths in the expert suite. Run the existing checkpoint CLI
inside the same verified container/runtime:

```bash
python -m scripts.helpfulness.cli run \
  --suite /ABSOLUTE/EXPERT/evals/helpfulness.toml \
  --bundle /ABSOLUTE/PREPARED/helpfulness-bundle \
  --output /ABSOLUTE/PERMANENT/teacher-run
```

This command is for an allocated compute session/job, not a long-running login
node experiment. Actual served teacher/judge IDs and credentials are still
required. Serving compatibility for the custom Apertus2KDA checkpoints has not
been established. Never assume a generic container or successful SSH connection
proves it. The original proprietary judges/checkpoints are not reproducible
simply by substituting Swiss AI models.

The reusable agent procedure is
`.copilot/skills/clariden-helpfulness-evals/SKILL.md`.
