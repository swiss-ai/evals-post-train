import copy
import unittest

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
        flags = suites.flags_for(suite, "apertus-2-chonk-sft", "ifbench")
        self.assertEqual(flags[:4], ["--backend", "openai", "--chat-format", "apertus2"])
        self.assertIn("apertus2", suite["setups"]["apertus-2-chonk-sft"]["serving"])
        self.assertEqual(flags[flags.index("--max-length") + 1], "16384")

    def test_plan_has_one_row_per_benchmark_in_table_order(self):
        suite = suites.load(SUITE)
        rows = suites.plan(suite, THINK)
        self.assertEqual([r["task"] for r in rows], [b["task"] for b in suite["benchmarks"]])
        self.assertEqual(rows[0]["metric"], "mean@4")

    def test_a_benchmark_samples_with_its_setups_profile_unless_it_names_one(self):
        suite = suites.load(SUITE)
        self.assertEqual(suites.sampling_for(suite, THINK, "aime2024"), "thinking")
        suite["benchmarks"][5]["sampling"] = "thinking"
        self.assertEqual(suites.plan(suite, THINK)[5]["sampling"], "thinking")

    def test_unknown_task_or_setup_is_refused(self):
        suite = suites.load(SUITE)
        with self.assertRaisesRegex(ValueError, "not in suite"):
            suites.flags_for(suite, THINK, "nope")
        with self.assertRaisesRegex(ValueError, "unknown setup"):
            suites.flags_for(suite, "nope", "ifbench")

    def test_malformed_suites_name_the_field(self):
        base = suites.load(SUITE)
        for mutate, message in (
            (lambda s: s["benchmarks"][5].update(framework="x"), "framework"),
            (lambda s: s["benchmarks"][5]["repeats"].update(n=0), "repeats.n"),
            (lambda s: s["benchmarks"][5]["repeats"].update(via="both"), "repeats.via"),
            (lambda s: s["benchmarks"][5]["metric"].update(label="mean@4"), "does not match"),
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

    def test_inspect_rows_get_their_own_flags_task_args_and_epochs(self):
        suite = suites.load("apertus2/chonk_sft")
        self.assertEqual(suites.flags_for(suite, "qwen3-32b-thinking", "mmlu_0_shot"),
                         ["--enable-thinking", "--task-arg", "cot=true", "--", "--epochs", "4"])
        self.assertEqual(suites.flags_for(suite, "apertus-2-chonk-sft", "aime2024"), ["--", "--epochs", "8"])
        lm_eval = suites.flags_for(suite, "qwen3-32b-thinking", "ifbench")
        self.assertIn("--max-length", lm_eval)
        self.assertNotIn("--epochs", lm_eval)

    def test_inspect_flags_may_not_hold_what_the_suite_produces(self):
        suite = suites.load("apertus2/chonk_sft")
        suite["setups"]["qwen3-32b-thinking"]["flags"]["inspect"].append("--epochs")
        with self.assertRaisesRegex(ValueError, "--epochs"):
            suites.validate(suite)

    def test_cli(self):
        for argv in (["list"], ["show", SUITE], ["plan", SUITE, "--setup", THINK],
                     ["flags", SUITE, "--setup", THINK, "--task", "aime2025"]):
            self.assertEqual(suites.main(argv), 0)
        self.assertEqual(suites.main(["flags", SUITE, "--setup", "x", "--task", "aime2025"]), 1)


if __name__ == "__main__":
    unittest.main()
