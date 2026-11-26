"""How a suite row becomes one command of a framework's runner: the table of what differs between them."""

from scripts.common.suites.flags import flags_for, sampling_for

HELPER_URL = "{helper_url}"


class Runner:
    """How a suite row becomes one command of a framework's runner.

    Every runner takes --model, --api-base-url and --limit, and all but tau2 take --wandb-project/--no-wandb.
    What differs is data, kept here and nowhere else:
      script, sbatch  the runner, and whether the row is `sbatch script` (qos and partition as --qos=x before
                      the script) or `bash script <subcommand>` (a login-node launcher that submits its own
                      jobs: qos and partition as --qos x at the end);
      task_flag       how the benchmark is named there, and task_value what it is called;
      out_flag        where results go (lm_eval's logs already sit under the W&B project: none);
      wandb           whether it uploads to W&B (tau2 prints its summary instead);
      sampling_profile, api_model_name   the flags only some runners take."""

    def __init__(self, script, task_flag, task_value=lambda task: task, out_flag="--out-dir", wandb=True,
                 sampling_profile=False, api_model_name=False, sbatch=True, subcommand=()):
        self.script, self.task_flag, self.task_value = script, task_flag, task_value
        self.out_flag, self.wandb = out_flag, wandb
        self.sampling_profile, self.api_model_name = sampling_profile, api_model_name
        self.sbatch, self.subcommand = sbatch, list(subcommand)

    def argv(self, scheduling, task, rest, sbatch_flags=()):
        """The command: the script, then the benchmark, then `rest`."""
        benchmark = [self.task_flag, self.task_value(task)]
        if self.sbatch:
            return ["sbatch", *scheduling["equals"], *sbatch_flags, self.script, *benchmark, *rest]
        return ["bash", self.script, *self.subcommand, *benchmark, *rest, *scheduling["flags"]]


RUNNERS = {
    "lm_eval": Runner("scripts/lm_eval_harness/launch_evaluations.sh", "--task", out_flag=None, api_model_name=True,
                      sbatch=False, subcommand=["single"]),
    "inspect": Runner("scripts/inspect_ai/run_inspect_eval.sbatch", "--task"),
    "harbor": Runner("scripts/harbor/run_harbor_eval.sbatch", "--benchmark"),
    "evalscope": Runner("scripts/evalscope/run_evalscope_eval.sbatch", "--benchmark"),
    "bfcl": Runner("scripts/custom/run_bfcl.sbatch", "--version", task_value=lambda task: task.removeprefix("bfcl_"),
                   sampling_profile=True),
    "tau2": Runner("scripts/custom/run_tau2_bench.sbatch", "--domain", wandb=False, sampling_profile=True),
}


def commands(suite, models, api_base_url=None, api_model_name=None, limit=None, no_wandb=False,
             qos=None, partition=None, fetch=False, helper_url=None):
    """One launch per (model, benchmark), models in the order given, benchmarks in table order.

    `models` maps a setup name to the name the model is served under. Every launch is its own Slurm job (lm_eval:
    one `single` launch of the login-node launcher; the other frameworks: one sbatch), described by its Runner.
    api_model_name only applies to lm_eval, which takes the served name separately from the model; qos and
    partition go to the launcher or to sbatch. Every launch but tau2's uploads to the W&B project named after the
    suite (the tau2 runner has no upload), and all but lm_eval's write to logs/<framework>/<suite>/<served name>/<task>
    (lm_eval's logs already sit under the project). If the suite has a `results_dir`, results go to
    <results_dir>/<framework>/<served name>/<task> (lm_eval: --logs-root <results_dir>/lm_eval) and the sbatch
    runners' Slurm files to <results_dir>/slurm/%x_%j.{out,err}; the caller creates that directory.
    A flag may carry `{helper_url}`: the endpoint of a helper model launched alongside (tau2's simulated user
    and judge), which the caller names with helper_url.
    Returns [{"setup", "label", "framework", "argv"}]."""
    scheduling = {"flags": (["--qos", qos] if qos else []) + (["--partition", partition] if partition else []),
                  "equals": ([f"--qos={qos}"] if qos else []) + ([f"--partition={partition}"] if partition else [])}
    rows = []
    for setup_name, served in models.items():
        for bench in suite["benchmarks"]:
            framework = bench["framework"]
            if framework not in RUNNERS:
                raise ValueError(f"{bench['task']}: no launcher for the {framework} framework yet")
            runner = RUNNERS[framework]
            flags = flags_for(suite, setup_name, bench["task"], fetch=fetch)
            if any(HELPER_URL in f for f in flags):
                if not helper_url:
                    raise ValueError(f"{bench['label']} needs the endpoint of its helper model (user and judge): pass --helper-url")
                flags = [f.replace(HELPER_URL, helper_url) for f in flags]
            rest = ["--model", served]
            if api_base_url:
                rest += ["--api-base-url", api_base_url]
            if limit:
                rest += ["--limit", str(limit)]
            if runner.wandb:
                rest += ["--no-wandb"] if no_wandb else ["--wandb-project", suite["name"]]
            if runner.api_model_name and api_base_url:
                rest += ["--api-model-name", api_model_name or served]
            if runner.sampling_profile:
                rest += ["--sampling-profile", sampling_for(suite, setup_name, bench["task"])]
            results_dir = suite.get("results_dir")
            sbatch_flags = []
            if results_dir:
                if runner.out_flag:
                    rest += [runner.out_flag, f"{results_dir}/{framework}/{served}/{bench['task']}"]
                else:
                    rest += ["--logs-root", f"{results_dir}/{framework}"]
                sbatch_flags = [f"--output={results_dir}/slurm/%x_%j.out", f"--error={results_dir}/slurm/%x_%j.err"]
            elif runner.out_flag:
                rest += [runner.out_flag, f"logs/{framework}/{suite['name']}/{served}/{bench['task']}"]
            argv = runner.argv(scheduling, bench["task"], [*rest, *flags], sbatch_flags)
            rows.append({"setup": setup_name, "label": bench["label"], "framework": framework, "argv": argv})
    return rows
