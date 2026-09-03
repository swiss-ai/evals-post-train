"""AA-LCR task (aaii/aa_lcr.py): _resolve_directory has to survive
two different real corruptions in AA's own published zip (see its
docstring) without ever silently mispairing a name to the wrong file --
these are exactly the cases that motivated it, not hypothetical ones.
IntegrationTests exercises the scorer through inspect_ai's own eval() (see
test_omniscience.py's docstring for why this matters and unit tests alone
don't substitute for it), using mockllm so no network call/API key is
needed."""

import unittest

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.solver import generate

from aaii.aa_lcr import _GRADE_RE, _resolve_directory, aa_lcr_scorer


class ResolveDirectoryTests(unittest.TestCase):
    def test_exact_matches_need_no_pairing(self):
        available = ["d/a.txt", "d/b.txt"]
        resolved = _resolve_directory("d", {"a.txt", "b.txt"}, available)
        self.assertEqual(resolved, {"a.txt": "d/a.txt", "b.txt": "d/b.txt"})

    def test_single_corrupted_filename_is_paired_by_similarity(self):
        # The real case: a UTF-8 apostrophe mis-decoded into "ΓÇÖ" garbage.
        available = ["d/Report on the EUΓÇÖs policy.txt", "d/Unrelated other file.txt"]
        resolved = _resolve_directory(
            "d", {"Report on the EU’s policy.txt", "Unrelated other file.txt"}, available
        )
        self.assertEqual(resolved["Report on the EU’s policy.txt"], "d/Report on the EUΓÇÖs policy.txt")
        self.assertEqual(resolved["Unrelated other file.txt"], "d/Unrelated other file.txt")

    def test_unreferenced_extra_file_in_the_pool_is_left_alone(self):
        # The real case: legal_eu_ai has a genuine extra file no row's
        # data_source_filenames ever names -- the pool being bigger than
        # the name set must not by itself be treated as an error.
        available = ["d/a.txt", "d/orphan-nobody-references.txt"]
        resolved = _resolve_directory("d", {"a.txt"}, available)
        self.assertEqual(resolved, {"a.txt": "d/a.txt"})

    def test_two_simultaneous_corruptions_in_one_directory_pair_correctly(self):
        # The real case that broke a naive "exactly one leftover" rule:
        # legal_eu_ai has two corrupted filenames at once, plus an
        # unrelated orphan file sitting in the same pool as a decoy.
        available = [
            "d/Long awaited EU AI Act becomes law after publication in the EUΓÇÖs Official Journal.txt",
            "d/What the EU AI Act means for youΓÇöand how to prepare.txt",
            "d/Preparing for change_ unrelated third document.txt",
        ]
        names = {
            "Long awaited EU AI Act becomes law after publication in the EU’s Official Journal.txt",
            "What the EU AI Act means for you—and how to prepare.txt",
        }
        resolved = _resolve_directory("d", names, available)
        self.assertEqual(
            resolved["Long awaited EU AI Act becomes law after publication in the EU’s Official Journal.txt"],
            "d/Long awaited EU AI Act becomes law after publication in the EUΓÇÖs Official Journal.txt",
        )
        self.assertEqual(
            resolved["What the EU AI Act means for you—and how to prepare.txt"],
            "d/What the EU AI Act means for youΓÇöand how to prepare.txt",
        )
        # The orphan must not be claimed by either real name.
        self.assertNotIn("d/Preparing for change_ unrelated third document.txt", resolved.values())

    def test_no_confident_match_raises_rather_than_guesses(self):
        # Nothing in the pool is remotely similar -- a wrong document
        # silently entering a ~100k-token prompt is worse than a loud
        # failure at dataset-load time.
        available = ["d/Completely unrelated filename.txt"]
        with self.assertRaises(KeyError):
            _resolve_directory("d", {"Report on the EU’s policy.txt"}, available)


class GradeParsingTests(unittest.TestCase):
    def test_parses_correct_and_incorrect(self):
        self.assertEqual(_GRADE_RE.search("CORRECT").group(1).lower(), "correct")
        self.assertEqual(_GRADE_RE.search("INCORRECT").group(1).lower(), "incorrect")

    def test_binds_to_the_last_mention_not_the_first(self):
        text = "This might be CORRECT at first glance, but it's actually INCORRECT"
        self.assertEqual(_GRADE_RE.search(text).group(1).lower(), "incorrect")

    def test_no_verdict_present_does_not_match(self):
        self.assertIsNone(_GRADE_RE.search("The answer looks reasonable."))


def _run_with_fixed_grade(candidate: str, grade_response: str) -> str:
    """One sample through the real scorer via inspect_ai's own eval(),
    using mockllm for both the target and grader roles. Returns the
    resolved Score.value."""
    dataset = [
        Sample(
            input="question",
            target="the official answer",
            id=0,
            metadata={"question": "the short original question"},
        )
    ]
    target_model = get_model(
        "mockllm/target", custom_outputs=[ModelOutput.from_content("mockllm", candidate)]
    )
    grader_model = get_model(
        "mockllm/grader", custom_outputs=[ModelOutput.from_content("mockllm", grade_response)]
    )
    task = Task(dataset=dataset, solver=[generate()], scorer=aa_lcr_scorer())
    logs = inspect_eval(
        task,
        model=target_model,
        model_roles={"grader": grader_model},
        display="none",
        log_dir="/tmp/test_aa_lcr_integration_logs",
    )
    return logs[0].samples[0].scores["aa_lcr_scorer"].value


class IntegrationTests(unittest.TestCase):
    def test_correct_verdict_scores_correct(self):
        self.assertEqual(_run_with_fixed_grade("an answer", "CORRECT"), "C")

    def test_incorrect_verdict_scores_incorrect(self):
        self.assertEqual(_run_with_fixed_grade("an answer", "INCORRECT"), "I")

    def test_unparseable_grader_output_scores_incorrect_not_a_crash(self):
        self.assertEqual(_run_with_fixed_grade("an answer", "unable to assess this"), "I")


if __name__ == "__main__":
    unittest.main()
