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
        self.assertEqual(suites.flags_for(suite, THINK, "ifbench"),
                         ["--thinking", "--think-start-token", "<|inner_prefix|>", "--think-end-token", "<|inner_suffix|>",
                          "--max-length", "32768", "--repeats", "4"])
        self.assertNotIn("--repeats", suites.flags_for(suite, THINK, "minerva_math500_self_consistency"))

    def test_setup_sets_window_and_format(self):
        suite = suites.load("apertus2/chonk_sft")
        flags = suites.flags_for(suite, "chonk-lc-16k-v0.2-sft", "ifbench")
        self.assertEqual(flags[:4], ["--backend", "openai", "--chat-format", "apertus2"])
        self.assertIn("apertus2", suite["setups"]["chonk-lc-16k-v0.2-sft"]["serving"])
        self.assertEqual(flags[flags.index("--max-length") + 1], "16384")

    def test_plan_has_one_row_per_benchmark_in_table_order(self):
        suite = suites.load(SUITE)
        rows = suites.plan(suite, THINK)
        self.assertEqual([r["task"] for r in rows], [b["task"] for b in suite["benchmarks"]])
        self.assertEqual(rows[0]["metric"], "mean@4")

    def test_a_benchmark_samples_with_its_setups_profile_unless_it_names_one(self):
        suite = suites.load(SUITE)
        self.assertEqual(suites.sampling_for(suite, THINK, "aime2024"), "thinking")
        suite["benchmarks"][6]["sampling"] = "thinking"
        self.assertEqual(suites.plan(suite, THINK)[6]["sampling"], "thinking")

    def test_unknown_task_or_setup_is_refused(self):
        suite = suites.load(SUITE)
        with self.assertRaisesRegex(ValueError, "not in suite"):
            suites.flags_for(suite, THINK, "nope")
        with self.assertRaisesRegex(ValueError, "unknown setup"):
            suites.flags_for(suite, "nope", "ifbench")

    def test_malformed_suites_name_the_field(self):
        base = suites.load(SUITE)
        for mutate, message in (
            (lambda s: s["benchmarks"][6].update(framework="x"), "framework"),
            (lambda s: s["benchmarks"][6]["repeats"].update(n=0), "repeats.n"),
            (lambda s: s["benchmarks"][6]["repeats"].update(via="both"), "repeats.via"),
            (lambda s: s["benchmarks"][6]["metric"].update(label="mean@4"), "does not match"),
            (lambda s: s["benchmarks"][1].update(task=s["benchmarks"][0]["task"]), "unique"),
            (lambda s: s["benchmarks"][0].pop("metric"), "missing 'metric'"),
            (lambda s: s["benchmarks"][0].update(repeats=4), "repeats must be an object"),
            (lambda s: s.pop("description"), "description"),
            (lambda s: s["setups"][THINK].update(max_length="32k"), "max_length"),
            (lambda s: s["benchmarks"][6].update(sampling="warm"), "unknown sampling profile"),
            (lambda s: s["setups"][THINK].update(sampling="greedy"), "not deterministic"),
            (lambda s: s["benchmarks"][6].update(sampling="non_thinking"), "not deterministic"),
            (lambda s: s["setups"][THINK].pop("sampling"), "sampling"),
            (lambda s: s["setups"][THINK]["flags"]["lm_eval"].extend(["--repeats", "2"]), "--repeats"),
            (lambda s: s["setups"][THINK].update(flags=["--thinking"]), "keyed by framework"),
            (lambda s: s["setups"][THINK].update(serving="--reasoning-parser x"), "serving must be a list"),
            (lambda s: s["setups"][THINK]["flags"].pop("lm_eval"), "no entry for 'lm_eval'"),
            (lambda s: s["benchmarks"][2].update(task_args={"cot": "true"}), "inspect framework only"),
        ):
            suite = copy.deepcopy(base)
            mutate(suite)
            with self.assertRaisesRegex(ValueError, message):
                suites.validate(suite)

    def test_inspect_rows_get_their_own_flags_task_args_and_epochs(self):
        suite = suites.load("apertus2/chonk_sft")
        self.assertEqual(suites.flags_for(suite, "qwen3-32b-thinking", "mmlu_0_shot"),
                         ["--enable-thinking", "--task-arg", "cot=true", "--", "--epochs", "4"])
        self.assertEqual(suites.flags_for(suite, "chonk-lc-16k-v0.2-sft", "aime2024"), ["--reasoning-effort", "high", "--", "--epochs", "8"])
        lm_eval = suites.flags_for(suite, "qwen3-32b-thinking", "ifbench")
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
            flags = suites.flags_for(suite, "chonk-lc-16k-v0.2-sft", "ifbench")
            self.assertEqual(flags[flags.index("--tokenizer") + 1], f"/cache/tokenizers/{spec['revision']}/{spec['path']}")
            self.assertNotIn("--tokenizer", suites.flags_for(suite, "chonk-lc-16k-v0.2-sft", "aime2024"))
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
        self.assertEqual([(r["setup"], r["label"]) for r in rows[:2]], [("chonk-lc-16k-v0.2-sft", "MMLU"), ("chonk-lc-16k-v0.2-sft", "GPQA")])
        inspect_row, lm_eval_row = rows[0]["argv"], rows[2]["argv"]
        self.assertEqual(inspect_row[:3], ["sbatch", "--qos=highprio", "--partition=highprio"])
        self.assertEqual(inspect_row[-3:], ["--", "--epochs", "4"])
        self.assertIn("--task-arg", inspect_row)
        self.assertEqual(lm_eval_row[:3], ["bash", "scripts/lm_eval_harness/launch_evaluations.sh", "single"])
        self.assertEqual(lm_eval_row[lm_eval_row.index("--task") + 1], "ifbench")
        self.assertEqual(lm_eval_row.count("--backend"), 1)  # the Chonk setup already names it
        qwen_lm_eval = next(r["argv"] for r in rows if r["setup"] == "qwen3-32b-thinking" and r["framework"] == "lm_eval")
        self.assertEqual(qwen_lm_eval[qwen_lm_eval.index("--backend") + 1], "openai")
        self.assertEqual(qwen_lm_eval[qwen_lm_eval.index("--api-model-name") + 1], "qwen")

    def test_swe_bench_row_tells_the_agent_the_window_of_each_setup(self):
        suite = suites.load(SUITE)
        self.assertEqual(suites.benchmark(suite, "swe_bench")["framework"], "harbor")
        chonk = suites.flags_for(suite, "chonk-lc-16k-v0.2-sft", "swe_bench")
        self.assertEqual(chonk, ["--agent-kwarg", 'model_info={"max_input_tokens": 16384}', "--reasoning-effort", "high",
                                 "--sandbox", "opensandbox"])
        think = suites.flags_for(suite, "apertus-1.5-70b-thinking", "swe_bench")
        self.assertEqual(think, ["--enable-thinking", "--agent-kwarg", 'model_info={"max_input_tokens": 32768}',
                                 "--sandbox", "opensandbox"])
        self.assertNotIn("--reasoning-effort", think)

    def test_run_prints_by_default_and_checks_its_arguments(self):
        with mock.patch.object(suites.subprocess, "run", side_effect=AssertionError("submitted")), \
                mock.patch("sys.stderr", new=io.StringIO()) as err:
            self.assertEqual(suites.main(["run", SUITE, "--model", f"{THINK}=m", "--model", "qwen3-32b-thinking=q"]), 0)
        self.assertIn("26 launches: 2 models x 13 benchmarks", err.getvalue())
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
        self.assertEqual(suites.flags_for(suite, chonk, "aime2024"), ["--reasoning-effort", "high", "--", "--epochs", "8"])
        for other in set(suite["setups"]) - {chonk, "chonk-lc-16k-v0.1-ep1"}:
            for bench in suite["benchmarks"]:
                self.assertNotIn("--reasoning-effort", suites.flags_for(suite, other, bench["task"]))

    def test_task_flags_replace_the_setups_flags_of_that_task_only(self):
        suite = suites.load(SUITE)
        chonk, v1 = "chonk-lc-16k-v0.2-sft", "chonk-lc-16k-v0.1-ep1"
        fc = suites.flags_for(suite, chonk, "bfcl_v3_fc")
        self.assertIn("--api-endpoint", fc)
        self.assertNotIn("--chat-format", fc)
        self.assertEqual(fc.count("--max-length"), 1)
        self.assertIn("--chat-format", suites.flags_for(suite, chonk, "ifbench"))
        for task in (b["task"] for b in suite["benchmarks"]):
            self.assertEqual(suites.flags_for(suite, v1, task), suites.flags_for(suite, chonk, task))
        suite["setups"][chonk]["task_flags"]["bfcl_v3_fc"] += ["--max-length", "1"]
        with self.assertRaisesRegex(ValueError, "task_flags.bfcl_v3_fc"):
            suites.validate(suite)
        suite["setups"][chonk]["task_flags"] = {"nope": []}
        with self.assertRaisesRegex(ValueError, "task_flags"):
            suites.validate(suite)

    def test_a_benchmarks_flags_follow_the_setups_and_come_before_the_epochs(self):
        suite = suites.load(SUITE)
        flags = suites.flags_for(suite, THINK, "hle")
        self.assertEqual(flags[:1], ["--enable-thinking"])
        self.assertEqual(flags.count("--model-role"), 2)
        self.assertEqual(flags[-2:], ["--task-arg", "judge_prompt=grade_c_i"])
        self.assertEqual(suites.flags_for(suite, THINK, "scicode"), ["--enable-thinking", "--sandbox", "opensandbox"])
        next(b for b in suite["benchmarks"] if b["task"] == "bfcl_v3_fc")["flags"] = ["--repeats", "2"]
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
                     ["flags", SUITE, "--setup", THINK, "--task", "aime2025"]):
            self.assertEqual(suites.main(argv), 0)
        self.assertEqual(suites.main(["flags", SUITE, "--setup", "x", "--task", "aime2025"]), 1)


if __name__ == "__main__":
    unittest.main()
