import gzip
import json
import tempfile
import unittest
from pathlib import Path

from scripts.custom.tau2_summary import summarize_results
from scripts.results.tau2_wandb_utils import create_model_evaluation_from_tau2_run
from scripts.results.trace_upload import compress_trace, upload_trace

EMPTY = "AssistantMessage must have either content or tool_calls. Got AssistantMessage"


def sim(task, reward, why="user_stop", error=None, start="2026-11-24T10:00:00", end="2026-11-24T10:05:00"):
    out = {"task_id": task, "termination_reason": why, "start_time": start, "end_time": end,
           "reward_info": {"reward": reward} if reward is not None else None}
    if error:
        out["info"] = {"error": error}
    return out


def results(*sims, domain="airline"):
    return {"info": {"environment_info": {"domain_name": domain}}, "simulations": list(sims)}


def evaluation(data, trials=1):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp, "results.json")
        path.write_text(json.dumps(data))
        return create_model_evaluation_from_tau2_run("m", path, trials)


class Tau2WandbTests(unittest.TestCase):
    def test_a_run_becomes_one_task_with_its_scores_and_failure_counts(self):
        data = results(sim("0", 1.0), sim("1", 0.0), sim("2", None, "infrastructure_error", EMPTY, "2026-11-24T10:01:00", "2026-11-24T10:20:00"))
        model_eval, duration, problems = evaluation(data)
        flat = model_eval.get_flattened_metrics()
        self.assertEqual(model_eval.tasks[0].task_name, "tau2_airline")
        self.assertEqual((flat["tau2_airline/pass^1"], flat["tau2_airline/n_tasks"]), (1 / 3, 3))  # the empty reply counts as failed
        self.assertEqual((flat["tau2_airline/n_agent_failures"], flat["tau2_airline/n_empty_replies"],
                          flat["tau2_airline/n_context_overflows"], flat["tau2_airline/n_unscored"]), (1, 1, 0, 0))
        self.assertEqual(flat["tau2_airline/pass^1_excluding_agent_failures"], 0.5)  # what tau2 itself prints
        self.assertEqual((duration, problems), (20 * 60, []))  # from the first start to the last end

    def test_a_task_that_failed_for_another_reason_is_not_counted_and_is_reported(self):
        judge = "litellm.AuthenticationError: Incorrect API key provided: None"
        model_eval, _, problems = evaluation(results(sim("0", 1.0), sim("1", None, "infrastructure_error", judge), domain="retail"))
        flat = model_eval.get_flattened_metrics()
        self.assertEqual((flat["tau2_retail/pass^1"], flat["tau2_retail/n_unscored"]), (1.0, 1))
        self.assertNotIn("tau2_retail/pass^1_excluding_agent_failures", flat)  # nothing was counted as the model's
        self.assertEqual(len(problems), 1)

    def test_pass_k_comes_with_every_trial_count(self):
        data = results(*[sim("0", r) for r in (1.0, 1.0)], *[sim("1", r) for r in (1.0, 0.0)])
        flat = evaluation(data, trials=2)[0].get_flattened_metrics()
        self.assertEqual((flat["tau2_airline/pass^1"], flat["tau2_airline/pass^2"]), (0.75, 0.5))

    def test_a_file_without_a_scored_simulation_is_refused(self):
        with self.assertRaisesRegex(ValueError, "no scored simulations"):
            evaluation(results(sim("0", None, "infrastructure_error", "boom")))

    def test_the_summary_module_is_what_the_launcher_prints(self):
        metrics, problems = summarize_results(results(sim("0", 1.0), sim("1", 0.0)), 1, expected=3)
        self.assertEqual((metrics["pass^1"], metrics["n_missing"]), (0.5, 1))
        self.assertIn("1 of 3 simulations are not in results.json", problems[0])


class TraceUploadTests(unittest.TestCase):
    def test_a_trace_is_compressed_and_reads_back_the_same(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp, "results.json")
            source.write_text(json.dumps({"messages": ["hello"] * 5000}))
            packed = compress_trace(source, Path(tmp))
            self.assertEqual(packed.name, "results.json.gz")
            self.assertLess(packed.stat().st_size, source.stat().st_size / 10)
            self.assertEqual(gzip.open(packed).read(), source.read_bytes())

    def test_a_trace_over_the_limit_is_not_uploaded_and_wandb_is_not_needed_to_say_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp, "results.json")
            source.write_text("x" * 100000)
            self.assertIsNone(upload_trace("e", "p", "m", "tau2_airline", source, max_mb=0))


if __name__ == "__main__":
    unittest.main()
