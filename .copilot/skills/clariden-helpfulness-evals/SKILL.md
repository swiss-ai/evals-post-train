---
name: clariden-helpfulness-evals
description: Queue and diagnose helpfulness benchmark acceptance checks on Clariden using Slurm, EDF containers, frozen upstream evidence and independent Swiss AI judges.
---

# Clariden helpfulness evaluation jobs

Use this skill for scorer regression replay, pinned reference regeneration, judge
calibration and Slurm diagnosis. Read `docs/helpfulness.md` and
`docs/helpfulness-cluster.md` before acting. This file lives in the user-requested
`.copilot/skills` directory; agents can read it directly. If a client's skill
discovery requires another location, configure that client rather than claiming
this repository-local path is automatically discovered everywhere.

## Preconditions and authorization

- SSH is sufficient. VS Code tunnels are optional development tools, not Slurm
  prerequisites. Use the user's configured SSH alias; never request key
  passphrases, NGC keys or endpoint tokens in chat.
- Read-only discovery: `uname -m`, `sacctmgr show associations user=$USER`,
  `sinfo -h -o '%P %l %a'`, and names of EDF files in `$HOME/.edf`.
  Do not inspect credential files or dump the environment.
- Confirm the account, partition, runtime, EDF identity mounts, permanent result
  directory, independent evidence, and desired scope before submitting. Ask for
  permission before allocating nodes, building images, installing packages or
  launching models. Do not submit merely because a template is available.
- Clariden uses ARM64. Never copy a Windows/x86 venv or assume the CPU validation
  lock/image works on ARM. Prefer the team's existing evaluation container and
  prepared harness overlay. NGC setup is separate; do not rotate existing keys.
- Generic handbook PyTorch/flash-attn versions are examples, not validated eval
  requirements. Build required images on allocated compute nodes, not login
  nodes. Do not change ACLs or recursively widen permissions on existing storage.

## Scientific lanes

1. Replay frozen responses and judgments against an independently captured
   upstream oracle. Compare every decision and denominator, not just totals.
2. Regenerate a pinned original open-model reference, preserving tokenizer,
   template, split, decoding and inference. Score predefined uncertainty gates.
3. Judge identical frozen answers against original judgments/human annotations;
   freeze a passing judge/rubric and require its lock before teacher evaluation.

No original proprietary checkpoint/judge is silently replaced. Missing published
pins, labels or source articles are blockers. Filtered MT-Bench and replacement
judges remain adapted even when a gate passes. Keep IFBench test/RL data separate.
Do not create scientific oracles or approved tolerances from synthetic tests.
Endpoints are operator-asserted deployments, not proven server state.

## Queue and monitor

Use `scripts/helpfulness/run_acceptance.sbatch` for one kind/benchmark/scope per
job. Create `logs/` before submitting, pass account/partition explicitly, and use
unique output directories. The script makes its plan inside the compute runtime:
sealed local plans contain absolute paths and exact Python/package fingerprints
and cannot simply be copied from Windows to ARM.

The job launches clients/scorers, not a model server. Hosting reference or teacher
weights locally needs a separately verified backend/allocation. Never assume
`Apertus2KDAForCausalLM` works in a generic vLLM image. Likelihood probes,
encoder-decoder baselines and unsupported original extraction must use their
pinned original/native producer and the external candidate artifact contract.

Use `afterok` dependencies only when the next job should require a passing gate.
Independent benchmark scopes need not depend on each other. If using arrays,
give every element its own output and cap concurrency to endpoint/resource
limits. A failed acceptance test exits nonzero; do not retry it with relaxed
thresholds or overwrite its evidence.

Monitor with `squeue --me`, `scontrol show job JOB_ID`, and
`sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed`. Cancel only a specific
user-approved job ID. Do not cancel all jobs or infer completion from queue
disappearance alone.

## Storage and credentials

Home is quota-limited; scratch is disposable and subject to purge. Keep frozen
evidence, exported manifests, configs, reports and judge locks on approved
permanent project/user storage. Stage HF caches and private NLTK assets separately;
NLTK directories must stay owner-private even when the broader project is shared.
Use dedicated transfer resources for large capstor/iopsstor transfers.

Pass credentials through protected runtime mechanisms/environment variable names,
never through CLI arguments, config values, job names or logs. Slurm environment
export is not a secret vault: follow site policy for authenticated clients.
Do not enable shell tracing or print endpoint request authorization headers.
Inspect only the specific requested storage/job paths, never other users' data.
