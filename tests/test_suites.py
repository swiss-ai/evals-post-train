import copy
import unittest

from scripts.common import suites


class SuiteTests(unittest.TestCase):
    def test_every_shipped_suite_is_valid(self):
        names = [p.stem for p in suites.SUITES_DIR.glob("*.json")]
        self.assertIn("table25", names)
        for name in names:
            suites.load(name)

    def test_repeats_flag_only_where_the_task_has_no_repeats_of_its_own(self):
        suite = suites.load("table25")
        recipe = "apertus-1.5-70b-thinking"
        self.assertEqual(
            suites.flags_for(suite, recipe, "ifbench"),
            ["--enable-thinking", "--chat-template", "--max-length", "32768", "--repeats", "4"])
        self.assertNotIn("--repeats", suites.flags_for(suite, recipe, "aime24"))

    def test_recipe_sets_window_and_format(self):
        flags = suites.flags_for(suites.load("table25"), "apertus-2-chonk-sft", "gsm8k_cot_zeroshot_self_consistency")
        self.assertEqual(flags[:2], ["--chat-format", "apertus2"])
        self.assertEqual(flags[flags.index("--max-length") + 1], "16384")

    def test_unknown_task_or_recipe_is_refused(self):
        suite = suites.load("table25")
        with self.assertRaisesRegex(ValueError, "not in suite"):
            suites.flags_for(suite, "apertus-1.5-70b-thinking", "nope")
        with self.assertRaisesRegex(ValueError, "unknown recipe"):
            suites.flags_for(suite, "nope", "ifbench")

    def test_malformed_suites_are_refused(self):
        base = suites.load("table25")
        for mutate, message in (
            (lambda s: s["benchmarks"]["aime24"].update(framework="x"), "framework"),
            (lambda s: s["benchmarks"]["aime24"].update(repeats=0), "repeats"),
            (lambda s: s["benchmarks"]["ifbench"].update(repeats=None), "repeats_flag"),
            (lambda s: s["recipes"]["apertus-2-chonk-sft"].update(max_length="16k"), "max_length"),
        ):
            suite = copy.deepcopy(base)
            mutate(suite)
            with self.assertRaisesRegex(ValueError, message):
                suites.validate(suite)

    def test_cli_prints_a_shell_safe_line(self):
        self.assertEqual(suites.main(["flags", "table25", "--recipe", "apertus-1.5-70b-thinking", "--task", "aime25"]), 0)
        self.assertEqual(suites.main(["flags", "table25", "--recipe", "x", "--task", "aime25"]), 1)


if __name__ == "__main__":
    unittest.main()
