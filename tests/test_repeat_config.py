import json
import tempfile
import unittest
from pathlib import Path

from scripts.lm_eval_harness.repeat_config import select_policy
from scripts.lm_eval_harness.task_output_types import Classifier
from scripts.results.merge_split_results import merge_split_results
from scripts.results.repeat_metrics import mean_key, score_label
from tests.test_task_output_types import _TaskManager
from tests.test_launch_flow import _launch, _run_config


class RepeatPolicyTests(unittest.TestCase):
    def test_group_chunking_preserves_overrides_and_rejects_global_typos(self):
        classifier = Classifier(_TaskManager(), None)
        policy = {"repeats": 3, "repeat_scoring": {"enabled": True},
                  "task_overrides": {"gsm8k": {"repeats": 5}, "arc_easy": {"repeats": 1}}}
        chunk = select_policy(policy, ["mixed_group"], ["gsm8k"], classifier)
        self.assertEqual(chunk["task_overrides"], {"gsm8k": {"repeats": 5}})
        policy["task_overrides"]["typo"] = {"repeats": 2}
        with self.assertRaisesRegex(ValueError, "typo"):
            select_policy(policy, ["mixed_group"], ["gsm8k"], classifier)

    def test_launch_snapshots_policy_content_in_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = root / "policy.json"
            configs = []
            for repeats in [3, 5]:
                policy.write_text(json.dumps({"repeats": repeats, "repeat_scoring": {"enabled": True}}))
                logs = root / str(repeats)
                run = _launch("single", "--task", "gsm8k_cot", "--model", "test/model",
                              "--repeat-config", str(policy), "--logs-root", str(logs))
                self.assertEqual(run.returncode, 0, run.stderr)
                config = _run_config(str(logs))
                configs.append(config)
                snapshot = next(logs.rglob("repeat_policy.json"))
                self.assertEqual(json.loads(snapshot.read_text())["repeats"], repeats)
            self.assertNotEqual(configs[0], configs[1])

    def test_merge_replaces_all_repeat_scores_on_shorter_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dirs = []
            for name, task, count in [("old", "a", 3), ("other", "b", 2), ("new", "a", 1)]:
                directory = root / name
                directory.mkdir()
                (directory / "results_test.json").write_text(json.dumps({
                    "results": {task: {"acc_mean_repeats,none": count}},
                    "repeat_results": [{"repeat_index": i, "results": {task: {"acc,none": i}}}
                                       for i in range(count)]}))
                dirs.append(directory)
            output = root / "merged"
            output.mkdir()
            merge_split_results(dirs, output)
            result = json.loads((output / "results_test.json").read_text())
            self.assertEqual(result["repeat_results"], [
                {"repeat_index": 0, "results": {"a": {"acc,none": 0}, "b": {"acc,none": 0}}},
                {"repeat_index": 1, "results": {"b": {"acc,none": 1}}}])

    def test_mean_resolution_including_code_metrics(self):
        summary = {"code/pass@1_mean_repeats,none": .25, "code/pass@1_pass@2,none": .5,
                   "repeat_protocols": {"code": {"repeats": 4}}}
        key = mean_key(summary, "code/pass@1")
        self.assertEqual(summary[key], .25)
        self.assertEqual(score_label(summary, "code/pass@1"), "mean@4")
        self.assertEqual(score_label(summary, "code/pass@1_pass@2"), "pass@2 (m=4)")
