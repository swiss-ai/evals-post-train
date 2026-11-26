"""Suites: the benchmarks of a comparison and the setups (window + launcher flags) a model can run under.

configs/suites/<name>.json holds two things:

  groups      optional ordered list of group names (e.g. Math, IF, Knowledge) the results view sections by
  benchmarks  rows in table order: label, optionally group (one of groups), task, framework, repeats {n, via}, metric {label, key},
              optionally sampling (a profile name that overrides the setup's) and task_args
              (inspect only: -T arguments of the task, e.g. {"cot": "true"}) and flags (launcher
              flags of this row under every setup, e.g. a sandbox or judge roles)
  setups      named kinds of model setup: max_length, sampling (a profile name), flags (the
              launcher flags of each framework the suite uses: thinking, chat format, ...) and
              optionally serving (the arguments the server itself needs, e.g. a reasoning parser;
              nothing here runs them; scripts/direct_serving/job.py --suite appends them to the launch
              block's) and launch {model_path, environment, nodes, time, framework_args} (how the
              model is served: the checkpoint, the serving EDF, the nodes of the one replica, the
              wall time and the layout arguments; scripts/direct_serving/job.py --suite/--setup reads it) and
              tokenizer {repo, revision, path} (a tokenizer pinned in a git repository; lm_eval
              gets --tokenizer <local copy>, which `flags` fetches once into the cache)

`run` starts every (model, benchmark) pair as its own launch, so each one is its own Slurm job (lm_eval: the
launcher's `single` mode; inspect: `sbatch run_inspect_eval.sbatch`). --model SETUP=SERVED_NAME names a
setup and the name its model is served under, repeatable. Without --submit it only prints the commands.

A setup's reasoning_effort (optional) becomes --reasoning-effort <level> on every lm_eval and inspect row
run under it.

Profiles are the ones of configs/sampling_defaults.json; a suite names one and holds no temperature.

`repeats.via` says who repeats: "flag" (lm_eval gets --repeats n, inspect gets -- --epochs n) or
"task" (the task already repeats n times, nothing is passed). max_length is the lm_eval --max-length;
inspect has no such flag, the server's --max-model-len is the window. The code only reads the file and knows no model.

    python -m scripts.common.suites list            # names, e.g. apertus2/chonk_sft
    python -m scripts.common.suites show apertus2/chonk_sft    # title and setups
    python -m scripts.common.suites plan apertus2/chonk_sft --setup apertus-1.5-70b-thinking
    python -m scripts.common.suites flags apertus2/chonk_sft --setup apertus-1.5-70b-thinking --task ifbench
    python -m scripts.common.suites run apertus2/chonk_sft --api-base-url <url> --helper-url <url> \\
        --model chonk-lc-16k-v0.2-sft=<served name> --model qwen3-32b-thinking=<served name> [--submit]
"""

# Split by concern: schema.py reads and validates the file, flags.py says what a setup means for one
# benchmark, runners.py turns a row into a runner's command line, cli.py is the command above. This module
# re-exports what the tests and scripts/direct_serving/job.py use.
from scripts.common.suites.cli import main, plan
from scripts.common.suites.flags import fetch_tokenizer, flags_for, sampling_for, tokenizer_dir
from scripts.common.suites.runners import RUNNERS, Runner, commands
from scripts.common.suites.schema import (
    FRAMEWORKS, REPO_ROOT, SUITES_DIR, benchmark, launch_for, load, setup, validate,
)

__all__ = [
    "FRAMEWORKS", "REPO_ROOT", "RUNNERS", "Runner", "SUITES_DIR", "benchmark", "commands", "fetch_tokenizer",
    "flags_for", "launch_for", "load", "main", "plan", "sampling_for", "setup", "tokenizer_dir", "validate",
]
