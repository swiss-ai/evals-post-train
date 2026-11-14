import copy
import unittest

from scripts.common import suites

THINK = "apertus-1.5-70b-thinking"


class SuiteTests(unittest.TestCase):
    def test_every_shipped_suite_is_valid(self):
        names = [p.stem for p in suites.SUITES_DIR.glob("*.json")]
        self.assertIn("table25", names)
        for name in names:
            suites.load(name)

    def test_repeats_flag_only_where_the_task_has_no_repeats_of_its_own(self):
        suite = suites.load("table25")
        self.assertEqual(suites.flags_for(suite, THINK, "ifbench"),
                         ["--enable-thinking", "--chat-template", "--max-length", "32768", "--repeats", "4"])
        self.assertNotIn("--repeats", suites.flags_for(suite, THINK, "aime24"))

    def test_setup_sets_window_and_format(self):
        flags = suites.flags_for(suites.load("table25"), "apertus-2-chonk-sft", "gsm8k_cot_zeroshot_self_consistency")
        self.assertEqual(flags[:2], ["--chat-format", "apertus2"])
        self.assertEqual(flags[flags.index("--max-length") + 1], "16384")

    def test_plan_has_one_row_per_benchmark_in_table_order(self):
        suite = suites.load("table25")
        rows = suites.plan(suite, THINK)
        self.assertEqual([r["task"] for r in rows], [b["task"] for b in suite["benchmarks"]])
        self.assertEqual(rows[0]["metric"], "mean@4")

    def test_unknown_task_or_setup_is_refused(self):
        suite = suites.load("table25")
        with self.assertRaisesRegex(ValueError, "not in suite"):
            suites.flags_for(suite, THINK, "nope")
        with self.assertRaisesRegex(ValueError, "unknown setup"):
            suites.flags_for(suite, "nope", "ifbench")

    def test_malformed_suites_name_the_field(self):
        base = suites.load("table25")
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
            (lambda s: s["setups"][THINK]["flags"].extend(["--repeats", "2"]), "--repeats"),
        ):
            suite = copy.deepcopy(base)
            mutate(suite)
            with self.assertRaisesRegex(ValueError, message):
                suites.validate(suite)

    def test_cli(self):
        for argv in (["list"], ["show", "table25"], ["plan", "table25", "--setup", THINK],
                     ["flags", "table25", "--setup", THINK, "--task", "aime25"]):
            self.assertEqual(suites.main(argv), 0)
        self.assertEqual(suites.main(["flags", "table25", "--setup", "x", "--task", "aime25"]), 1)


if __name__ == "__main__":
    unittest.main()
