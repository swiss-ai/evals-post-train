import copy
import io
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.common import suites

SUITE = "apertus2/chonk_sft"
THINK = "apertus-1.5-70b-thinking"


class SuiteTests(unittest.TestCase):
    def test_every_shipped_suite_is_valid(self):
        names = [str(p.relative_to(suites.SUITES_DIR).with_suffix("")) for p in suites.SUITES_DIR.rglob("*.json")]
        self.assertIn("apertus2/chonk_sft", names)
        for name in names:
            suites.load(name)

    def test_repeats_flag_only_where_the_task_has_no_repeats_of_its_own(self):
        suite = suites.load(SUITE)
        self.assertEqual(suites.flags_for(suite, THINK, "aime24_think"),
                         ["--thinking", "--think-start-token", "<|inner_prefix|>", "--think-end-token", "<|inner_suffix|>",
                          "--api-num-concurrent", "256", "--max-length", "262144", "--repeats", "4"])
        self.assertNotIn("--repeats", suites.flags_for(suite, THINK, "minerva_math500_think"))

    def test_apertus_15_70b_runs_with_high_client_concurrency_on_every_lm_eval_row(self):
        # Its two replicas sit idle at the launcher's default of 8.
        suite = suites.load(SUITE)
        flags = suites.flags_for(suite, THINK, "ifbench_think")
        self.assertEqual(flags[flags.index("--api-num-concurrent") + 1], "256")
        self.assertNotIn("--api-num-concurrent", suites.flags_for(suite, "apertus-1.5-8b-thinking", "ifbench_think"))

    def test_setup_sets_window_and_format(self):
        suite = suites.load("apertus2/chonk_sft")
        flags = suites.flags_for(suite, "chonk-lc-16k-v0.2-sft", "ifbench_think")
        self.assertEqual(flags[:4], ["--backend", "openai", "--chat-format", "apertus2"])
        self.assertIn("apertus2", suite["setups"]["chonk-lc-16k-v0.2-sft"]["serving"])
        self.assertEqual(flags[flags.index("--max-length") + 1], "32768")

    def test_plan_has_one_row_per_benchmark_in_table_order(self):
        suite = suites.load(SUITE)
        rows = suites.plan(suite, THINK)
        self.assertEqual([r["task"] for r in rows], [b["task"] for b in suite["benchmarks"]])
        self.assertEqual(rows[0]["metric"], "mean@4")

    def test_a_benchmark_samples_with_its_setups_profile_unless_it_names_one(self):
        suite = suites.load(SUITE)
        self.assertEqual(suites.sampling_for(suite, THINK, "aime24_think"), "thinking")
        suite["benchmarks"][5]["sampling"] = "thinking"
        self.assertEqual(suites.plan(suite, THINK)[5]["sampling"], "thinking")

    def test_unknown_task_or_setup_is_refused(self):
        suite = suites.load(SUITE)
        with self.assertRaisesRegex(ValueError, "not in suite"):
            suites.flags_for(suite, THINK, "nope")
        with self.assertRaisesRegex(ValueError, "unknown setup"):
            suites.flags_for(suite, "nope", "ifbench_think")

    def test_malformed_suites_name_the_field(self):
        base = suites.load(SUITE)
        for mutate, message in (
            (lambda s: s["benchmarks"][5].update(framework="x"), "framework"),
            (lambda s: s["benchmarks"][5]["repeats"].update(n=0), "repeats.n"),
            (lambda s: s["benchmarks"][5]["repeats"].update(via="both"), "repeats.via"),
            (lambda s: s["benchmarks"][5]["metric"].update(label="mean@8"), "does not match"),
            (lambda s: s["benchmarks"][1].update(task=s["benchmarks"][0]["task"]), "unique"),
            (lambda s: s["benchmarks"][0].pop("metric"), "missing 'metric'"),
            (lambda s: s["benchmarks"][0].update(repeats=4), "repeats must be an object"),
            (lambda s: s.pop("description"), "description"),
            (lambda s: s["setups"][THINK].update(max_length="32k"), "max_length"),
            (lambda s: s["benchmarks"][5].update(sampling="warm"), "unknown sampling profile"),
            (lambda s: s["setups"][THINK].update(sampling="greedy"), "not deterministic"),
            (lambda s: s["benchmarks"][5].update(sampling="non_thinking"), "not deterministic"),
            (lambda s: s["setups"][THINK].pop("sampling"), "sampling"),
            (lambda s: s["setups"][THINK]["flags"]["lm_eval"].extend(["--repeats", "2"]), "--repeats"),
            (lambda s: s["setups"][THINK].update(flags=["--thinking"]), "keyed by framework"),
            (lambda s: s["setups"][THINK].update(serving="--reasoning-parser x"), "serving must be a list"),
            (lambda s: s["setups"][THINK]["flags"].pop("lm_eval"), "no entry for 'lm_eval'"),
            (lambda s: s["benchmarks"][1].update(task_args={"cot": "true"}), "inspect framework only"),
        ):
            suite = copy.deepcopy(base)
            mutate(suite)
            with self.assertRaisesRegex(ValueError, message):
                suites.validate(suite)

    def test_a_setups_launch_block_becomes_the_serving_the_launcher_needs(self):
        suite = suites.load(SUITE)
        launch = suites.launch_for(suite, "chonk-lc-16k-v0.2-sft")
        self.assertEqual((launch["nodes"], launch["time"]), (2, "24:00:00"))
        self.assertTrue(Path(launch["environment"]).is_file())  # the EDF it names ships in this repo
        self.assertIn("--data-parallel-backend ray", launch["framework_args"])
        self.assertTrue(launch["framework_args"].endswith("--tool-call-parser apertus2"))  # the setup's serving
        with self.assertRaisesRegex(ValueError, "no launch block"):
            suites.launch_for(suite, "apertus-1.5-8b-thinking")
        launch = suites.launch_for(suite, THINK)
        self.assertTrue(Path(launch["environment"]).is_file())
        self.assertIn("--max-model-len 262144", launch["framework_args"])  # served with the setup's window

    def test_malformed_launch_blocks_name_the_field(self):
        base = suites.load(SUITE)
        for mutate, message in (
            (lambda launch: launch.pop("nodes"), "launch must be an object"),
            (lambda launch: launch.update(nodes=0), "launch.nodes"),
            (lambda launch: launch.update(time="1 day"), "launch.time"),
            (lambda launch: launch.update(environment=""), "launch.environment"),
            (lambda launch: launch.update(framework_args="--x"), "list of strings"),
            (lambda launch: launch["framework_args"].extend(["--model", "x"]), "the launcher sets it"),
            (lambda launch: launch.update(env={"A B": "1"}), "launch.env"),
            (lambda launch: launch.update(env={"A": 1}), "launch.env"),
            (lambda launch: launch.update(extra=1), "launch must be an object"),
        ):
            suite = copy.deepcopy(base)
            mutate(suite["setups"]["chonk-lc-16k-v0.2-sft"]["launch"])
            with self.assertRaisesRegex(ValueError, message):
                suites.validate(suite)

    def test_inspect_rows_get_their_own_flags_task_args_and_epochs(self):
        suite = suites.load("apertus2/chonk_sft")
        # No shipped inspect row repeats via the flag at the moment: the epochs path is exercised with a row set here.
        next(b for b in suite["benchmarks"] if b["task"] == "gpqa_diamond")["repeats"]["via"] = "flag"
        self.assertEqual(suites.flags_for(suite, "qwen3-32b-thinking", "gpqa_diamond"),
                         ["--enable-thinking", "--", "--epochs", "4"])
        with_args = copy.deepcopy(suite)
        next(b for b in with_args["benchmarks"] if b["task"] == "gpqa_diamond")["task_args"] = {"cot": "true"}
        self.assertEqual(suites.flags_for(with_args, "qwen3-32b-thinking", "gpqa_diamond"),
                         ["--enable-thinking", "--task-arg", "cot=true", "--", "--epochs", "4"])
        self.assertEqual(suites.flags_for(suite, "chonk-lc-16k-v0.2-sft", "gpqa_diamond"), ["--reasoning-effort", "high", "--", "--epochs", "4"])
        lm_eval = suites.flags_for(suite, "qwen3-32b-thinking", "ifbench_think")
        self.assertIn("--max-length", lm_eval)
        self.assertNotIn("--epochs", lm_eval)

    def test_inspect_flags_may_not_hold_what_the_suite_produces(self):
        suite = suites.load("apertus2/chonk_sft")
        suite["setups"]["qwen3-32b-thinking"]["flags"]["inspect"].append("--epochs")
        with self.assertRaisesRegex(ValueError, "--epochs"):
            suites.validate(suite)

    def test_a_pinned_tokenizer_becomes_the_lm_eval_tokenizer_flag(self):
        suite = suites.load(SUITE)
        spec = suite["setups"]["chonk-lc-16k-v0.2-sft"]["tokenizer"]
        with mock.patch.dict(os.environ, {"EVAL_ENV_CACHE_ROOT": "/cache"}):
            flags = suites.flags_for(suite, "chonk-lc-16k-v0.2-sft", "ifbench_think")
            self.assertEqual(flags[flags.index("--tokenizer") + 1], f"/cache/tokenizers/{spec['revision']}/{spec['path']}")
            self.assertNotIn("--tokenizer", suites.flags_for(suite, "chonk-lc-16k-v0.2-sft", "gpqa_diamond"))
        suite["setups"]["chonk-lc-16k-v0.2-sft"]["flags"]["lm_eval"] += ["--tokenizer", "x"]
        with self.assertRaisesRegex(ValueError, "--tokenizer"):
            suites.validate(suite)

    def test_malformed_tokenizer_pins_are_refused(self):
        base = suites.load(SUITE)
        for mutate, message in (
            (lambda t: t.update(revision="main"), "full commit sha"),
            (lambda t: t.update(path="../x"), "relative directory"),
            (lambda t: t.pop("repo"), "repo, revision and path"),
        ):
            suite = copy.deepcopy(base)
            mutate(suite["setups"]["chonk-lc-16k-v0.2-sft"]["tokenizer"])
            with self.assertRaisesRegex(ValueError, message):
                suites.validate(suite)

    @unittest.skipUnless(shutil.which("git"), "needs git")
    def test_fetch_tokenizer_copies_the_pinned_directory_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, cache = Path(tmp) / "repo", Path(tmp) / "cache"
            (repo / "tok" / "a").mkdir(parents=True)
            (repo / "tok" / "a" / "tokenizer.json").write_text("{}")
            git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run([*git, "add", "-A"], check=True)
            subprocess.run([*git, "commit", "-q", "-m", "x"], check=True)
            sha = subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
            spec = {"repo": str(repo), "revision": sha, "path": "tok/a"}
            with mock.patch.dict(os.environ, {"EVAL_ENV_CACHE_ROOT": str(cache)}):
                path = suites.fetch_tokenizer(spec)
                self.assertTrue((path / "tokenizer.json").is_file())
                with mock.patch.object(suites.subprocess, "run", side_effect=AssertionError("fetched twice")):
                    self.assertEqual(suites.fetch_tokenizer(spec), path)
                with self.assertRaisesRegex(ValueError, "could not fetch"):
                    suites.fetch_tokenizer({**spec, "revision": "0" * 40})

    def test_one_launch_per_model_and_benchmark(self):
        suite = suites.load(SUITE)
        models = {"chonk-lc-16k-v0.2-sft": "chonk", "qwen3-32b-thinking": "qwen"}
        rows = suites.commands(suite, models, api_base_url="http://x/v1", limit=5, no_wandb=True,
                               qos="highprio", partition="highprio")
        self.assertEqual(len(rows), len(models) * len(suite["benchmarks"]))
        self.assertEqual([(r["setup"], r["label"]) for r in rows[:2]], [("chonk-lc-16k-v0.2-sft", "GPQA Diamond"), ("chonk-lc-16k-v0.2-sft", "IFBench")])
        inspect_row, lm_eval_row = rows[0]["argv"], rows[1]["argv"]
        self.assertEqual(inspect_row[:3], ["sbatch", "--qos=highprio", "--partition=highprio"])
        self.assertEqual(lm_eval_row[:3], ["bash", "scripts/lm_eval_harness/launch_evaluations.sh", "single"])
        self.assertEqual(lm_eval_row[lm_eval_row.index("--task") + 1], "ifbench_think")
        self.assertEqual(lm_eval_row.count("--backend"), 1)  # the Chonk setup already names it
        qwen_lm_eval = next(r["argv"] for r in rows if r["setup"] == "qwen3-32b-thinking" and r["framework"] == "lm_eval")
        self.assertEqual(qwen_lm_eval[qwen_lm_eval.index("--backend") + 1], "openai")
        self.assertEqual(qwen_lm_eval[qwen_lm_eval.index("--api-model-name") + 1], "qwen")

    def test_tau2_rows_are_one_sbatch_per_domain_with_the_agents_thinking_flags(self):
        suite = suites.load(SUITE)
        rows = {(r["setup"], r["label"]): r["argv"] for r in suites.commands(
            suite, {"chonk-lc-16k-v0.2-sft": "chonk", "qwen3-32b-thinking": "qwen"}, api_base_url="http://x/v1",
            limit=2, qos="highprio", partition="highprio")}
        domains = ["airline", "retail", "telecom", "banking_knowledge"]
        for domain in domains:
            argv = rows["chonk-lc-16k-v0.2-sft", f"tau2-bench {domain}"]
            self.assertEqual(argv, ["sbatch", "--qos=highprio", "--partition=highprio", "scripts/custom/run_tau2_bench.sbatch",
                                    "--domain", domain, "--model", "chonk", "--api-base-url", "http://x/v1",
                                    "--num-tasks", "2", "--workdir", f"logs/tau2/{suite['name']}/chonk/{domain}",
                                    "--reasoning-effort", "high"])
        self.assertEqual(rows["qwen3-32b-thinking", "tau2-bench retail"][-1], "--enable-thinking")

    def test_every_launch_uploads_to_the_suites_wandb_project(self):
        suite = suites.load(SUITE)
        rows = suites.commands(suite, {"chonk-lc-16k-v0.2-sft": "chonk"}, api_base_url="http://x/v1")
        for row in rows:
            argv = row["argv"]
            if row["framework"] == "tau2":
                self.assertNotIn("--wandb-project", argv)  # its runner has no W&B upload
                continue
            self.assertEqual(argv[argv.index("--wandb-project") + 1], suite["name"], row["label"])
        for row in suites.commands(suite, {"chonk-lc-16k-v0.2-sft": "chonk"}, no_wandb=True):
            self.assertNotIn("--wandb-project", row["argv"])

    def test_sbatch_rows_write_to_their_own_log_dir(self):
        suite = suites.load(SUITE)
        flag = {"inspect": "--logs-dir", "evalscope": "--work-dir", "harbor": "--work-dir", "bfcl": "--workdir", "tau2": "--workdir"}
        seen = set()
        for row in suites.commands(suite, {"chonk-lc-16k-v0.2-sft": "chonk"}):
            argv = row["argv"]
            if row["framework"] not in flag:
                for name in flag.values():
                    self.assertNotIn(name, argv)
                continue
            if row["framework"] == "bfcl":
                task = "bfcl_" + argv[argv.index("--version") + 1]
            elif row["framework"] == "tau2":
                task = argv[argv.index("--domain") + 1]
            else:
                task = argv[argv.index("--task") + 1] if "--task" in argv else argv[argv.index("--benchmark") + 1]
            expected = f"logs/{row['framework']}/{suite['name']}/chonk/{task}"
            self.assertEqual(argv[argv.index(flag[row["framework"]]) + 1], expected)
            seen.add(expected)
        self.assertIn(f"logs/bfcl/{suite['name']}/chonk/bfcl_v4", seen)

    def test_bfcl_rows_run_the_custom_runner(self):
        suite = suites.load(SUITE)
        rows = {r["label"]: r["argv"] for r in suites.commands(
            suite, {"chonk-lc-16k-v0.2-sft": "chonk"}, api_base_url="http://x/v1", limit=2, qos="highprio")}
        for label, version in (("BFCL v3", "v3"), ("BFCL v4", "v4")):
            argv = rows[label]
            self.assertEqual(argv[:3], ["sbatch", "--qos=highprio", "scripts/custom/run_bfcl.sbatch"])
            self.assertEqual(argv[argv.index("--version") + 1], version)
            self.assertEqual(argv[argv.index("--model") + 1], "chonk")
            self.assertEqual(argv[argv.index("--api-base-url") + 1], "http://x/v1")
            self.assertEqual(argv[argv.index("--num-tasks") + 1], "2")
            self.assertEqual(argv[argv.index("--sampling-profile") + 1], "thinking")
            self.assertEqual(argv[argv.index("--reasoning-effort") + 1], "high")
            self.assertEqual(argv[argv.index("--wandb-project") + 1], suite["name"])
            self.assertNotIn("--limit", argv)

    def test_run_prints_by_default_and_checks_its_arguments(self):
        with mock.patch.object(suites.subprocess, "run", side_effect=AssertionError("submitted")), \
                mock.patch("sys.stderr", new=io.StringIO()) as err:
            self.assertEqual(suites.main(["run", SUITE, "--model", f"{THINK}=m", "--model", "qwen3-32b-thinking=q"]), 0)
        self.assertIn("32 launches: 2 models x 16 benchmarks", err.getvalue())
        for bad in (["--model", "m"], ["--model", "nope=m"], ["--model", f"{THINK}=a", "--model", f"{THINK}=b"]):
            self.assertEqual(suites.main(["run", SUITE, *bad]), 1)

    def test_a_setups_reasoning_effort_reaches_all_its_lm_eval_and_inspect_rows(self):
        suite = suites.load(SUITE)
        chonk = "chonk-lc-16k-v0.2-sft"
        for name in (chonk, "chonk-lc-16k-v0.1-ep1"):
            self.assertEqual(suite["setups"][name]["reasoning_effort"], "high")
            for bench in suite["benchmarks"]:
                flags = suites.flags_for(suite, name, bench["task"])
                self.assertEqual(flags.count("--reasoning-effort"), 1, (name, bench["label"]))
                self.assertEqual(flags[flags.index("--reasoning-effort") + 1], "high")
                self.assertEqual(suites.sampling_for(suite, name, bench["task"]), "thinking")
        self.assertEqual(suites.flags_for(suite, chonk, "gpqa_diamond"), ["--reasoning-effort", "high"])
        for other in set(suite["setups"]) - {chonk, "chonk-lc-16k-v0.1-ep1"}:
            for bench in suite["benchmarks"]:
                self.assertNotIn("--reasoning-effort", suites.flags_for(suite, other, bench["task"]))

    def test_a_benchmarks_flags_follow_the_setups(self):
        suite = suites.load(SUITE)
        # No shipped row carries flags of its own at the moment: the mechanism is exercised with a row set here.
        next(b for b in suite["benchmarks"] if b["task"] == "gpqa_diamond")["flags"] = ["--sandbox", "opensandbox"]
        suites.validate(suite)
        flags = suites.flags_for(suite, THINK, "gpqa_diamond")
        setup_flags = suite["setups"][THINK]["flags"]["inspect"]
        self.assertEqual(flags[:len(setup_flags)], setup_flags)
        own = flags.index("--sandbox")
        self.assertEqual(flags[own:own + 2], ["--sandbox", "opensandbox"])
        next(b for b in suite["benchmarks"] if b["task"] == "ifbench_think")["flags"] = ["--repeats", "2"]
        with self.assertRaisesRegex(ValueError, "--repeats"):
            suites.validate(suite)

    def test_reasoning_effort_is_validated_and_not_a_flag_of_the_setup(self):
        suite = suites.load(SUITE)
        suite["setups"]["chonk-lc-16k-v0.2-sft"]["reasoning_effort"] = "loud"
        with self.assertRaisesRegex(ValueError, "reasoning_effort"):
            suites.validate(suite)
        suite = suites.load(SUITE)
        suite["setups"]["qwen3-32b-thinking"]["flags"]["inspect"] += ["--reasoning-effort", "low"]
        with self.assertRaisesRegex(ValueError, "--reasoning-effort"):
            suites.validate(suite)

    def test_cli(self):
        for argv in (["list"], ["show", SUITE], ["plan", SUITE, "--setup", THINK],
                     ["flags", SUITE, "--setup", THINK, "--task", "aime25_think"]):
            self.assertEqual(suites.main(argv), 0)
        self.assertEqual(suites.main(["flags", SUITE, "--setup", "x", "--task", "aime25_think"]), 1)


if __name__ == "__main__":
    unittest.main()
