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

### Standalone IFEval regression tooling (not in the teacher suite)

The active teacher suite selects AlpacaEval 2 LC, non-code MT-Bench,
biography sycophancy and the open-weight extensions documented in
[helpfulness.md](helpfulness.md#open-weight-teacher-extensions).
LitBench is separate scorer validation. Do not submit the IFEval preparation job for
that suite. The following tooling remains available for independent callers;
it requires a separate suite explicitly selecting IFEval and rejects the teacher
suite before installing dependencies. Previously submitted jobs and evidence
remain historical records, not scheduled teacher evaluations.

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
export EVAL_SUITE_FILE=/ABSOLUTE/INDEPENDENT/ifeval-suite.toml
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

The preparation preflight validates the staged independently selected suite
before installing anything. Reusable library adapters/tests are retained and
do not determine the active teacher suite.
Harness revisions are read using whitespace-safe Python parsing, and the shared
builder strips CR/LF from inherited revision values. Linux staging must normalize
shell scripts **and requirement text files**, not just `.sh` files.

This bootstraps **IFEval replay only**, not teacher-suite benchmark approvals.
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

### Apertus 2 load and native-format smoke

The SFT exports can contain all weight shards while having **no Jinja chat
template**. Do not invent a template, use Apertus 1 formatting, or infer serving
support from the presence of a KDA kernel. The server must register
`Apertus2KDAForCausalLM`; the three inspected generic CUDA images register
only the older `ApertusForCausalLM`.
The current `iter_0019374` test target is the **Chonk SFT baseline**, not an
already trained helpfulness/chit-chat teacher.

The pinned team CUDA recipe is available in
[`alps-extended-images`](https://github.com/andresnowak/alps-extended-images/tree/5ba1436ca304a9db90ca3a4a089548a3fced4485/Alps-Images/apps/vllm-apertus2/example).
Its native server uses TP1, expert parallelism and four local data-parallel
ranks per node, not an assumed TP4 layout. A separately versioned
[`evals-svc` EDF](https://github.com/swiss-ai/evals-svc/blob/8f4e52312bc122fc77244e5f5740054397b5e2b4/app/resources/envs/vllm-apertus2.toml)
pins CUDA image digest
`2daf3b1906ab62291c00b6a78d9eaa8aa85aad792abffb6695dc89ce9dcc9e05`.
Do not use the similarly named ROCm image on GH200. Provisioning/importing an
image and submitting GPU jobs require the operator's authorization.

For the bounded two-node Chonk SFT smoke, use
`--gpu-memory-utilization 0.90`, not the upstream recipe's `0.95`.
On the inspected 95 GiB devices, a worker had only 90.0 GiB free when vLLM
requested 90.25 GiB at 95%; startup failed before loading completed. The
90% reservation requests 85.5 GiB, leaving room for initialization overhead.
Record per-device free/total memory in the serving preflight. This headroom
adjustment does not establish that the model or KV cache will fit.

Run the server and bounded scorer as separate tracked child processes and
source `scripts/helpfulness/supervise_smoke.sh` in the allocation launcher.
Call `wait_for_smoke_client "$server_pid" "$client_pid"` so a failed server
ends the job instead of leaving the scorer waiting for the readiness timeout.
The launcher's EXIT trap must terminate and wait for both tracked processes.
Diagnose the first worker exception in the full server log: subsequent
PyTorch cleanup errors or NCCL peer-disconnection errors can mask it.

`scripts.helpfulness.teacher_smoke` reuses the repository-pinned harness's
`apertus2-completions` renderer/parser and the separately pinned
`requirements/apertus2.txt` library. Its scorer environment remains separate
from the serving image; install additions only into a new private environment
or overlay, never the shared image or historical runtime. Preserve source and
wheel hashes, resolved versions and image identity. A successful `preflight`
does **not** load weights or run inference:

```bash
python -m scripts.helpfulness.teacher_smoke preflight \
  --checkpoint /ABSOLUTE/CHECKPOINT \
  --source /STAGED/or-bench-hard-1k.csv \
  --upstream /STAGED/upstreams/or-bench --limit 3 \
  --output /FRESH/preflight.json
```

Include the source/upstream arguments in the preflight **before starting the
server step**. It validates real cases and pinned scorer assets with the same
explicit `dataset_config="or-bench-hard-1k"` and
`keyword_profile="opensource"` used during generation/scoring.

Once an approved allocated server is running on that same node's loopback
interface, the `run` command waits for its advertised model, generates at most
five real OR-Bench cases and applies the existing exact generic-prefix scorer:

```bash
python -m scripts.helpfulness.teacher_smoke run \
  --checkpoint /ABSOLUTE/CHECKPOINT --model EXACT_SERVED_MODEL \
  --source /STAGED/or-bench-hard-1k.csv \
  --upstream /STAGED/upstreams/or-bench --limit 3 \
  --output /FRESH/teacher-smoke
```

Native raw output, rendered prompt and parser/failure flags are retained.
Empty, malformed, truncated or dropped-output generations fail even when the
harness supplies a partial final answer. The smoke uses low reasoning effort,
temperature 0.6, top-p 0.95 and an 8192-token generation budget within a 16384-token
server context. This is a bounded format/runtime diagnostic, **not** full
OR-Bench coverage, a semantic refusal-detector validation, full teacher-suite
execution, judge calibration or public-weight/license attestation.

The regular teacher runner still requires its reviewed model/deployment
provenance and benchmark permissions. The smoke does not bypass those gates or
make the current chat-completions teacher transport native-format compatible.
Resolve that transport explicitly before scheduling the complete suite.
With the pinned CUDA image and 90% memory reservation above, the two-node
Chonk SFT `iter_0019374` smoke has completed weight loading and three validated
native-format generations with keyword scoring. This establishes that bounded
infrastructure path, not full benchmark coverage or teacher quality.

### Local semantic OR-Bench judge pilot

`scripts.helpfulness.or_bench_judge_smoke` judges a bounded set of frozen, validated
native replies without reloading the target (three by default, up to twenty with
`--expected-cases`). Stage the samples, their expected
SHA256 and the judge identity first; use `preflight` before starting the server
and `run` against the allocation-local `/v1` endpoint. It uses `ChatClient.judge`
with untrusted evidence separated from the frozen system rubric.

The protocol is **adapted-or-bench-semantic-refusal-v1**, not the original
prefix metric or a published OR-Bench score. It preserves original prefix results
alongside separate refusal, partial-refusal, compliance and unclear counts.
Invalid JSON, invented evidence quotes and truncated responses fail explicitly.
The existing strict JSON helper accepts a single fenced JSON object, but not
surrounding prose or an invalid verdict schema.
Returned judge text is retained before verdict parsing. The pilot establishes
judge execution only: no human-label calibration or full-suite acceptance.

An existing `google/gemma-4-31B-it` cache can serve this diagnostic with the pinned
CUDA image above, TP1, BF16, an 8192-token context and 90% GPU memory reservation.
Its architecture is `Gemma4ForConditionalGeneration`, not Apertus2. Use a private
publisher-pinned tokenizer directory and canonical chat template, explicitly set
`enable_thinking=false`, disable image/video inputs and use vLLM generation
defaults rather than unreviewed cached generation defaults. The pilot needs one
GPU; a full Clariden node allocation reserves four. Do not modify shared weights
or claim their contents match a publisher snapshot merely because the
configuration and weight index match.

For a broader diagnostic, `teacher_smoke --limit 20` accepts a separately frozen
twenty-row source CSV. Preserve its selection manifest and original source hash.
Two prompts per category covers the ten hard-split categories but is not a
proportional sample of the full dataset. Keep the refusal rubric frozen while
auditing the judge; model-assisted review is not independent human gold or
scientific calibration acceptance.

A twenty-case equal-category pilot has completed native Chonk generation and
local Gemma judging with the unchanged refusal rubric. The judge returned ten
full refusals, one partial refusal and nine compliance outcomes. A separate AI
review frozen before judging agreed on sixteen labels; three of the four
differences had already been marked borderline. The audit recommends clarifying
the boundary between an alternative task and a partial answer, plus handling
ambiguous/off-topic replies, before larger semantic-scored runs. This is not a
measured accuracy result or a validated revised rubric. Retain the original
results if a later rubric revision is tested.

`--rubric-version v2` explicitly selects clarified rules for actual requested
parts versus broad-goal alternatives, substantive help after a refusal, and
off-topic or ambiguous replies. The default remains `v1`, whose wording is
unchanged. Both versions use the same four labels and strict evidence schema,
but record distinct protocols and rubric hashes. A replay of examples that
informed v2 is a same-case tuning check, not independent calibration acceptance;
the report marks the revised rubric as not validated.

The twenty-answer v2 replay completed, changing two labels: a broad-goal
alternative changed from partial refusal to refusal, while a safety-heavy answer
with no explicit withholding changed from compliance to refusal. Counts became
twelve refusals and eight compliance outcomes, with no partial/unclear labels.
The audit found the first change consistent with the clarification, the second
questionable, and a legal-task interpretation issue still unresolved. Therefore
v2 is **experimental, not accepted as calibrated**; v1 remains the default.

`--rubric-version v3` adds structured, evidence-first judging: a constrained task
summary, verbatim question constraints, optional actual-withholding and
substantive-attempt passages, then the four-way label. All quoted passages are
checked against their respective source. The label must agree with the explicit
withholding/fulfillment decision table; contradictions fail and preserve raw
output rather than being silently relabelled. This improves auditability, not
proof that the semantic decisions are correct. V1/v2 remain available unchanged.

`--rubric-version v3-lines` retains that evidence-first decision rule but presents
numbered question/answer lines and asks the judge to select contiguous source
spans instead of copying text. The parser resolves those spans to exact source
quotes, retains the raw selections, and applies the same strict label-consistency
checks. Invalid indices or contradictory labels still fail. This avoids
accepting invented or stitched evidence without relying on model-perfect copying.

This local Gemma path has completed a three-answer pilot: the model loaded,
all three chat-completion requests returned, and all verdicts passed schema and
verbatim-quote checks. The first attempt stopped because Gemma fenced its JSON;
reusing the existing strict JSON helper fixed that formatting issue. The
replacement completed in 3m43s with one refusal, two compliance outcomes and no
partial/unclear outcomes. These match the original prefix labels on the same
three frozen replies. This verifies execution, **not judge accuracy**: the
first-three-case sample has no independent human ground truth. The target
was not reloaded and no weight shards were downloaded.

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
been established beyond the specific Chonk SFT smoke configuration above.
Never assume a generic container or successful SSH connection proves it.
The original proprietary judges/checkpoints are not reproducible
simply by substituting Swiss AI models.

The reusable agent procedure is
`.copilot/skills/clariden-helpfulness-evals/SKILL.md`.
