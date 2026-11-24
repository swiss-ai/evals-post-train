"""The score summary of a tau2 run's results.json (scripts/custom/run_tau2_bench.sh prints it, scripts/results/
tau2_wandb_utils.py uploads it).

tau2 gives up on a task after 4 attempts (an infrastructure error, no reward) and leaves it out of its own metrics.
The cause is in the task's info (error, error_type, error_traceback): when the model under test is to blame (it
replied with neither content nor tool calls, or its conversation overflowed its context window) the task counts as
failed, reward 0 (n_agent_failures); any other cause (the user simulator, the judge, the server, a missing key) is
not the model's and is not counted: those tasks are listed, with the ones missing altogether, and the run is not
complete, so that a task that is not the model's fault is rerun and never passes as a score.

    python -m scripts.custom.tau2_summary <results.json> <num trials> [<expected simulations>]
"""

import json
import math
import sys
from collections import defaultdict

# What the model under test is to blame for: tau2 rejects its reply as an AssistantMessage without content or tool
# calls, or its conversation outgrows its context window. A context overflow counts only when the failing call was the
# agent's (the traceback passes through tau2/agent/): the user simulator and the judge are other models with windows of
# their own.
AGENT_ERRORS = ("AssistantMessage must have either content or tool_calls",)


def agent_failure_cause(sim):
    """Why the model under test is to blame for a task without reward: "empty_reply" or "context_overflow"; else None."""
    if sim.get("termination_reason") != "infrastructure_error":
        return None
    info = sim.get("info") or {}
    if str(info.get("error", "")).startswith(AGENT_ERRORS):
        return "empty_reply"
    if info.get("error_type") == "ContextWindowExceededError" and "/tau2/agent/" in str(info.get("error_traceback", "")):
        return "context_overflow"
    return None


def is_success(reward):
    return (1 - 1e-6) <= reward <= (1 + 1e-6)


def summarize(rows, num_trials):
    """avg_reward and pass^k of `rows`, and their simulations per task."""
    per_task = defaultdict(list)
    for s in rows:
        per_task[s["task_id"]].append(s)
    rewards = [s["reward_info"]["reward"] for s in rows]
    out = {"avg_reward": sum(rewards) / len(rewards)}
    for k in range(1, num_trials + 1):
        vals = []
        for runs in per_task.values():
            n = len(runs)
            c = sum(is_success(s["reward_info"]["reward"]) for s in runs)
            if n >= k:
                vals.append(math.comb(c, k) / math.comb(n, k))
        if vals:
            out[f"pass^{k}"] = sum(vals) / len(vals)
    return out, per_task


def summarize_results(data, num_trials, expected=None):
    """(metrics, problems) of a results.json already loaded: the figures of the run, and why it is not complete."""
    all_sims = data.get("simulations") or []
    scored = [s for s in all_sims if (s.get("reward_info") or {}).get("reward") is not None]
    unscored = [s for s in all_sims if (s.get("reward_info") or {}).get("reward") is None]
    agent_failures = [s for s in unscored if agent_failure_cause(s)]
    unscored = [s for s in unscored if not agent_failure_cause(s)]
    sims = scored + [{**s, "reward_info": {"reward": 0.0}} for s in agent_failures]
    if not sims:
        return None, ["no scored simulations"]

    metrics, per_task = summarize(sims, num_trials)
    metrics["n_tasks"] = len(per_task)
    metrics["n_simulations"] = len(sims)
    metrics["n_agent_failures"] = len(agent_failures)
    metrics["agent_failures_by_cause"] = {cause: sum(agent_failure_cause(s) == cause for s in agent_failures)
                                          for cause in ("empty_reply", "context_overflow")}
    if agent_failures and scored:
        # What tau2 itself prints: the failed tasks left out, which flatters a model that fails often.
        metrics["excluding_agent_failures"] = summarize(scored, num_trials)[0]
    metrics["n_unscored"] = len(unscored)
    if expected is not None:
        metrics["n_missing"] = max(expected - len(all_sims), 0)
    problems = ["task %s: no reward (%s)" % (s.get("task_id"), s.get("termination_reason")) for s in unscored]
    if expected is not None and len(all_sims) < expected:
        problems.append("%d of %d simulations are not in results.json" % (expected - len(all_sims), expected))
    return metrics, problems


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    with open(argv[0]) as f:
        data = json.load(f)
    num_trials = int(argv[1])
    expected = int(argv[2]) if len(argv) > 2 and argv[2] else None
    metrics, problems = summarize_results(data, num_trials, expected)
    if metrics is None:
        print(problems[0])
        return 1
    print(json.dumps(metrics, indent=2))
    if problems:
        print("NOT A COMPLETE RESULT: " + "; ".join(problems), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
