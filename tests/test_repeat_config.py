import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.lm_eval_harness.repeat_config import select_policy, with_flags
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

    def test_globs_resolve_and_unindexed_tasks_pass_through(self):
        classifier = Classifier(_TaskManager(), None)
        policy = {"repeat_scoring": {"enabled": True}, "task_overrides": {"gsm8k": {"repeats": 5}}}
        chunk = select_policy(policy, ["mixed_group", "/tasks/custom.yaml"], ["/tasks/custom.yaml"], classifier)
        self.assertEqual(chunk["task_overrides"], {})
        globbed = select_policy(policy, ["gsm8*"], ["gsm8*"], classifier)
        self.assertEqual(globbed["task_overrides"], {"gsm8k": {"repeats": 5}})

    def test_flags_build_a_policy_and_win_over_the_config(self):
        policy = with_flags(None, 8, ["8"])
        self.assertEqual(policy["repeats"], 8)
        self.assertTrue(policy["repeat_scoring"]["mean"])
        self.assertEqual(policy["repeat_scoring"]["pass_at_k"]["exact_match"], [8])
        base = {"repeats": 5, "repeat_scoring": {"enabled": True, "pass_at_k": {"exact_match": [2]}},
                "task_overrides": {"ifeval": {"repeats": 1}}}
        merged = with_flags(base, 4, [])
        self.assertEqual(merged["repeats"], 4)
        self.assertEqual(merged["task_overrides"], base["task_overrides"])
        self.assertEqual(base["repeats"], 5)
        self.assertEqual(with_flags(base, None, [4])["repeat_scoring"]["pass_at_k"]["exact_match"], [2])
        with self.assertRaisesRegex(ValueError, "pass@8 needs"):
            with_flags(None, 4, [8])

    def test_launcher_checks_repeat_flags(self):
        base = ("single", "--task", "gsm8k_cot", "--model", "test/model", "--backend", "openai",
                "--api-base-url", "http://localhost:8000", "--api-model-name", "m")
        with tempfile.TemporaryDirectory() as tmp:
            for flags, message in [
                (["--repeats", "4"], "identical samples"),
                (["--repeats", "4", "--enable-thinking", "--temperature", "0"], "identical samples"),
                (["--repeats", "x"], "positive integer"),
                (["--repeats", "4", "--temperature", "0.6", "--repeat-metric", "pass@8"], "needs at least 8"),
                (["--repeat-metric", "mean"], "pass@K"),
            ]:
                run = _launch(*base, *flags, "--logs-root", tmp)
                self.assertNotEqual(run.returncode, 0, flags)
                self.assertIn(message, run.stdout + run.stderr)
            ok = _launch(*base, "--repeats", "4", "--repeat-metric", "pass@4", "--temperature", "0.6",
                         "--logs-root", tmp)
            self.assertEqual(ok.returncode, 0, ok.stdout + ok.stderr)
            thinking = _launch(*base, "--repeats", "4", "--enable-thinking", "--logs-root", tmp)
            self.assertEqual(thinking.returncode, 0, thinking.stdout + thinking.stderr)
            local = _launch("single", "--task", "gsm8k_cot", "--model", "test/model", "--repeats", "4",
                            "--temperature", "0.6", "--logs-root", tmp)
            self.assertIn("need --backend openai", local.stdout + local.stderr)

    def test_launcher_refuses_repeat_config_for_local_backends(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy = Path(tmp) / "policy.json"
            policy.write_text(json.dumps({"repeat_scoring": {"enabled": True}}))
            run = _launch("single", "--task", "gsm8k_cot", "--model", "test/model",
                          "--repeat-config", str(policy), "--logs-root", tmp)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("need --backend openai", run.stdout + run.stderr)

    def test_job_runs_repeat_config_from_the_repository(self):
        sbatch = (Path(__file__).resolve().parents[1] / "scripts/lm_eval_harness/run_lm_eval.sbatch").read_text()
        make_cmd = sbatch[sbatch.index("make_cmd() {"):]
        make_cmd = make_cmd[:make_cmd.index("\n}\n") + 3]
        script = (make_cmd + "COMMON_EVAL_ARGS=(); LM_EVAL_BACKEND=openai API_LM_EVAL_MODEL=x\n"
                  "EVAL_REPEAT_CONFIG=/p.json EVAL_REPEAT_TASKS=/t.txt\n"
                  'cd /tmp && make_cmd gsm8k /out true && printf "%s" "$CMD"')
        run = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)
        self.assertTrue(run.stdout.startswith("(cd /tmp && python -m scripts.lm_eval_harness.repeat_config "),
                        run.stdout)
        self.assertIn("--config /out/repeat_config.json", run.stdout)
        with tempfile.TemporaryDirectory() as tmp:
            python = Path(tmp) / "python"
            python.write_text("#!/bin/bash\nexit 3\n")
            python.chmod(0o755)
            failed = subprocess.run(["bash", "-c", f"set -euo pipefail\n{run.stdout}\necho continued"],
                                    capture_output=True, text=True, env={"PATH": f"{tmp}:/usr/bin:/bin"})
        self.assertEqual(failed.returncode, 3)
        self.assertNotIn("continued", failed.stdout)

    def test_environment_build_checks_the_policy_against_the_suite(self):
        script = (Path(__file__).resolve().parents[1] / "scripts/lm_eval_harness/build_eval_env.sh").read_text()
        check = script[script.index("if [[ -n \"${EVAL_REPEAT_CONFIG:-}\""):]
        check = check[:check.index("\nfi\n") + 4]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bin").mkdir()
            python = root / "bin/python"
            python.write_text(f'#!/bin/bash\necho "$PWD $*" > {root}/args\n')
            python.chmod(0o755)
            (root / "tasks.txt").write_text("aime25\ngsm8k\n")
            env = {"EVAL_REPEAT_CONFIG": "/p.json", "EVAL_REPEAT_TASKS": str(root / "tasks.txt"),
                   "EVAL_REPO_ROOT": tmp, "BASE_ENV": tmp, "HARNESS_OVERLAY": "/overlay", "PATH": "/usr/bin:/bin"}
            subprocess.run(["bash", "-c", "set -euo pipefail\n" + check], env=env, check=True)
            args = (root / "args").read_text()
        self.assertIn(f"{tmp} -m scripts.lm_eval_harness.repeat_config --policy /p.json", args)
        self.assertIn("--tasks aime25,gsm8k", args)

    def test_upload_clears_old_protocols_and_maps_repeat_headlines(self):
        from unittest import mock
        from scripts.results import wandb_alignment_utils as upload
        from scripts.results.data_structures import Metric, ModelEvaluation, Task

        def uploaded(protocols, metric):
            run = mock.MagicMock(summary={})
            fake = mock.MagicMock()
            fake.init.return_value.__enter__.return_value = run
            evaluation = ModelEvaluation("m", [Task("aime25", [Metric(metric, 0.5)], [])], protocols)
            with mock.patch.object(upload, "wandb", fake), \
                    mock.patch.object(upload, "create_wandb_table", lambda name, data: data):
                upload._upload_to_wandb_with_model_eval("e", "p", evaluation, ["aime25/exact_match"], 1)
            return run.summary["repeat_protocols"], run.log.call_args_list[0].args[0]["main_results"]

        protocols, main = uploaded({"aime25": {"repeats": 8}}, "exact_match_mean_repeats,none")
        self.assertEqual(protocols, {"aime25": {"repeats": 8}})
        self.assertEqual(main, {"aime25/exact_match": 0.5})
        protocols, _ = uploaded({}, "exact_match,none")
        self.assertEqual(protocols, {})

    def test_launch_snapshots_policy_content_in_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = root / "policy.json"
            configs = []
            for repeats in [3, 5]:
                policy.write_text(json.dumps({"repeats": repeats, "repeat_scoring": {"enabled": True}}))
                logs = root / str(repeats)
                run = _launch("single", "--task", "gsm8k_cot", "--model", "test/model",
                              "--backend", "openai", "--api-base-url", "http://localhost:8000",
                              "--api-model-name", "test-model",
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
            merge_split_results(dirs, output, excluded_tasks={"b"})
            result = json.loads((output / "results_test.json").read_text())
            self.assertEqual(result["repeat_results"], [{"repeat_index": 0, "results": {"a": {"acc,none": 0}}}])
            merge_split_results(dirs, output, excluded_tasks={"a", "b"})
            self.assertNotIn("repeat_results", json.loads((output / "results_test.json").read_text()))

    def test_mgsm_average_and_sample_classes_use_repeat_scores(self):
        from scripts.results import make_html_table, make_table
        from scripts.results import wandb_alignment_utils as upload

        summary = {"repeat_protocols": {"mgsm_direct_de": {}}, "mgsm_direct_de/exact_match": .9,
                   "mgsm_direct_de/exact_match_mean_repeats": .5, "mgsm_direct_fr/exact_match": .7}
        for module in (make_html_table, make_table):
            self.assertAlmostEqual(module.compute_mgsm_avg(summary), .6)
        samples = [{"metrics": ["exact_match"], "exact_match": scores} for scores in ([1, 1], [1, 0])]
        selected = upload._select_stratified_samples(samples, n_positive=1, n_negative=1)
        self.assertEqual(sorted(sample["is_correct"] for sample in selected), [False, True])

    def test_group_labels_resolve_filterless_metrics(self):
        summary = {"g/exact_match_mean_repeats": .5, "repeat_protocols": {
            "g": {"repeats": 5, "per_task_repeats": {"a": 5, "b": 5},
                  "metric_tasks": {"exact_match_mean_repeats,extract": ["a", "b"]}},
            "a": {"sampling_enabled": False}, "b": {"sampling_enabled": False}}}
        self.assertEqual(score_label(summary, "g/exact_match"), "mean@5 (greedy)")

    def test_mean_resolution_including_code_metrics(self):
        summary = {"code/pass@1_mean_repeats,none": .25, "code/pass@1_pass@2,none": .5,
                   "repeat_protocols": {"code": {"repeats": 4}}}
        key = mean_key(summary, "code/pass@1")
        self.assertEqual(summary[key], .25)
        stale = {"code/pass@1_mean_repeats,none": .25, "code/pass@1,none": .75, "repeat_protocols": {}}
        self.assertIsNone(mean_key(stale, "code/pass@1"))
        self.assertEqual(score_label(summary, "code/pass@1"), "mean@4")
        self.assertEqual(score_label(summary, "code/pass@1_pass@2"), "pass@2 (m=4)")
        summary["repeat_protocols"]["code"]["sampling_enabled"] = False
        self.assertEqual(score_label(summary, "code/pass@1"), "mean@4 (greedy)")
        summary["group/pass@1_pass@2,none"] = .5
        summary["repeat_protocols"]["group"] = {
            "repeats": 4, "per_task_repeats": {"code": 4},
            "metric_tasks": {"pass@1_pass@2,none": ["code"]}}
        self.assertEqual(score_label(summary, "group/pass@1_pass@2"), "pass@2 (m=4) (greedy)")
