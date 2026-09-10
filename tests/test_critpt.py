"""CritPt task (aaii/aa_critpt.py): the two-step solver (reasoning generate,
then a separate parse generate against the challenge's own code_template),
the scorer's capture-only behavior, and the grading-server Hook's batching/
guard logic -- exercised through inspect_ai's own eval() with mockllm, same
approach as test_aa_lcr.py/test_omniscience.py, so no network call or API
key is needed for any of this."""

import json
import tempfile
import unittest
from pathlib import Path

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageAssistant, ModelOutput, get_model

from aaii.aa_critpt import (
    CHALLENGE_COUNT,
    _group_submissions_by_epoch,
    critpt_capture_scorer,
    critpt_two_step_solver,
)


def _dataset(n: int) -> list[Sample]:
    return [
        Sample(
            input=f"Problem statement {i}",
            target="reference answer code",
            id=f"Challenge_{i}_main",
            metadata={"code_template": f"def answer():\n    # template {i}\n    ..."},
        )
        for i in range(n)
    ]


def _run(n_samples: int, epochs: int = 1):
    """Runs the two-step solver + capture scorer via mockllm: every
    generate() call returns a fixed, distinguishable completion so the test
    can tell the reasoning step's output from the parse step's."""
    model = get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput.from_content("mockllm", "REASONING_COMPLETION"),
            ModelOutput.from_content("mockllm", "PARSED_CODE"),
        ]
        * n_samples
        * epochs,
    )
    task = Task(
        dataset=_dataset(n_samples),
        solver=[critpt_two_step_solver()],
        scorer=critpt_capture_scorer(),
        epochs=epochs,
    )
    # max_samples=1: mockllm serves custom_outputs from one shared, ordered
    # queue -- with samples running concurrently (Inspect's default), two
    # samples' generate() calls can interleave in whatever order requests
    # actually arrive, so a sample could end up consuming the OTHER
    # sample's turn's output. Forcing fully sequential execution keeps the
    # REASONING_COMPLETION/PARSED_CODE assignment deterministic per sample.
    #
    # A fresh tempdir per call, not one shared log_dir: aaii/aa_critpt.py's
    # own Hook is a globally-registered Inspect Hook (confirmed by its
    # "hooks enabled: 1" startup banner), so it fires automatically after
    # EVERY inspect_eval() call in this file, including from tests that
    # have nothing to do with grading -- writing into a directory derived
    # from each run's own log path. A shared log_dir would let unrelated
    # tests' batch files collide/overwrite each other at the same
    # epoch_N_batch.json name.
    return inspect_eval(
        task,
        model=model,
        display="none",
        log_dir=tempfile.mkdtemp(prefix="critpt_test_"),
        max_samples=1,
    )


class TwoStepSolverTests(unittest.TestCase):
    def test_two_separate_generate_calls_happen(self):
        logs = _run(1)
        sample = logs[0].samples[0]
        assistant_messages = [
            m for m in sample.messages if isinstance(m, ChatMessageAssistant)
        ]
        self.assertEqual(len(assistant_messages), 2)
        self.assertEqual(assistant_messages[0].content, "REASONING_COMPLETION")
        self.assertEqual(assistant_messages[1].content, "PARSED_CODE")

    def test_reasoning_completion_is_kept_separately_in_metadata(self):
        logs = _run(1)
        sample = logs[0].samples[0]
        self.assertEqual(sample.metadata["reasoning_completion"], "REASONING_COMPLETION")

    def test_system_prompt_is_inserted_first(self):
        logs = _run(1)
        sample = logs[0].samples[0]
        self.assertEqual(sample.messages[0].role, "system")

    def test_parse_prompt_carries_the_sample_code_template(self):
        logs = _run(1)
        sample = logs[0].samples[0]
        user_messages = [m for m in sample.messages if m.role == "user"]
        # messages[0] is the original problem input; the parse-step prompt
        # is the second user turn, and must carry this sample's own
        # code_template (not some other sample's -- see _dataset()'s
        # per-index template).
        self.assertIn("template 0", user_messages[1].content)


class ScorerTests(unittest.TestCase):
    def test_captures_the_parsed_code_as_the_answer(self):
        logs = _run(1)
        score = logs[0].samples[0].scores["critpt_capture_scorer"]
        self.assertEqual(score.answer, "PARSED_CODE")
        self.assertEqual(score.explanation, "REASONING_COMPLETION")

    def test_scorer_declares_no_metrics(self):
        # metrics=[] means no metrics are computed/reported at all for this
        # stub scorer, not one entry with an empty dict -- Score.value here
        # isn't a real grade (see module docstring), so there is nothing to
        # aggregate.
        logs = _run(1)
        self.assertEqual(logs[0].results.scores, [])


class GroupSubmissionsByEpochTests(unittest.TestCase):
    def test_groups_by_epoch_with_one_entry_per_sample(self):
        logs = _run(2, epochs=2)
        by_epoch = _group_submissions_by_epoch(logs[0].samples, "test-model")
        self.assertEqual(set(by_epoch.keys()), {1, 2})
        self.assertEqual(len(by_epoch[1]), 2)
        self.assertEqual(len(by_epoch[2]), 2)

    def test_each_submission_carries_the_right_problem_id_and_code(self):
        logs = _run(2, epochs=1)
        by_epoch = _group_submissions_by_epoch(logs[0].samples, "test-model")
        ids = {sub["problem_id"] for sub in by_epoch[1]}
        self.assertEqual(ids, {"Challenge_0_main", "Challenge_1_main"})
        for sub in by_epoch[1]:
            self.assertEqual(sub["generated_code"], "PARSED_CODE")
            self.assertEqual(sub["model"], "test-model")


class GradingServerHookTests(unittest.TestCase):
    def test_writes_a_local_batch_file_when_no_credential_is_set(self):
        # No manual hook call needed: Inspect auto-invokes every registered
        # Hook's on_task_end once inspect_eval() finishes (confirmed by the
        # "hooks enabled: 1" banner printed during _run() itself). No
        # CRITPT_API_KEY/CRITPT_GRADING_URL in the test environment -- the
        # expected default state (see module docstring).
        logs = _run(1)
        log_location = logs[0].location
        self.assertTrue(log_location)

        out_dir = Path(log_location).parent / "critpt_submissions"
        batch_files = sorted(out_dir.glob("epoch_*_batch.json"))
        self.assertEqual(len(batch_files), 1)
        payload = json.loads(batch_files[0].read_text())
        self.assertEqual(len(payload["submissions"]), 1)
        # A 1-submission batch is far short of CHALLENGE_COUNT -- confirms
        # the fixture's own premise (this test is about the local-file
        # fallback, not a real complete batch).
        self.assertLess(len(payload["submissions"]), CHALLENGE_COUNT)
        # No result file: nothing was ever POSTed (no credential, and the
        # batch is incomplete either way).
        self.assertFalse(list(out_dir.glob("epoch_*_result.json")))


if __name__ == "__main__":
    unittest.main()
