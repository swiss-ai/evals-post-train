"""AgentClinic (scripts/inspect_ai/tasks/agentclinic.py): the dialogue loop and the
moderator, run end to end through inspect_ai's eval() with mockllm models (no network):
the doctor's turns route to the patient or the measurement reader as in the source, the
patient never sees the doctor's inline thinking, and a case scores 1 only when the
moderator says Yes to a diagnosis given within the turn limit."""

import hashlib
import json
import os
import tempfile
import unittest
from unittest import mock

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset
from inspect_ai.model import ModelOutput, get_model

from scripts.inspect_ai.tasks import agentclinic as ac

CASE = {
    "OSCE_Examination": {
        "Objective_for_Doctor": "Assess the patient with double vision.",
        "Patient_Actor": {"Demographics": "35-year-old female", "History": "Diplopia"},
        "Physical_Examination_Findings": {"Vital_Signs": {"Heart_Rate": "72 bpm"}},
        "Test_Results": {"Acetylcholine_Receptor_Antibodies": "Positive"},
        "Correct_Diagnosis": "Myasthenia gravis",
    }
}


def _outputs(*texts):
    return [ModelOutput.from_content(model="mockllm", content=t) for t in texts]


class _Recorder:
    """A mockllm custom_outputs callable that records what each call was sent."""

    def __init__(self, *texts):
        self.texts = list(texts)
        self.calls = []

    def __call__(self, messages, tools, tool_choice, config):
        self.calls.append([m.text for m in messages])
        return ModelOutput.from_content(model="mockllm", content=self.texts.pop(0))


def _run(doctor, patient, measurement, moderator, max_turns=20):
    task = Task(
        dataset=MemoryDataset([ac.case_to_sample(CASE, 0)]),
        solver=ac.clinic_dialogue(max_turns=max_turns),
        scorer=ac.moderator_scorer(),
        model_roles={
            "patient": get_model("mockllm/model", custom_outputs=patient),
            "measurement": get_model("mockllm/model", custom_outputs=measurement),
            "moderator": get_model("mockllm/model", custom_outputs=moderator),
        },
    )
    with tempfile.TemporaryDirectory() as logs:
        log = inspect_eval(
            task,
            model=get_model("mockllm/model", custom_outputs=doctor),
            log_dir=logs,
            display="none",
        )[0]
        assert log.status == "success", log.error
        return log.samples[0]


class DialogueTests(unittest.TestCase):
    def test_a_right_diagnosis_after_a_question_and_a_test_scores_1(self):
        patient = _Recorder("My vision doubles when I'm tired.")
        measurement = _Recorder("RESULTS: Acetylcholine receptor antibodies positive.")
        moderator = _Recorder("Yes.")
        sample = _run(
            _outputs(
                "What brings you in?",
                "REQUEST TEST: Acetylcholine_Receptor_Antibodies",
                "DIAGNOSIS READY: Myasthenia gravis",
            ),
            patient,
            measurement,
            moderator,
        )
        self.assertEqual(sample.scores["moderator_scorer"].value, "C")
        self.assertEqual(sample.metadata["turns"], 3)
        self.assertEqual(sample.metadata["tests_requested"], 1)
        # one patient reply, one test result, one moderator verdict
        self.assertEqual((len(patient.calls), len(measurement.calls), len(moderator.calls)), (1, 1, 1))
        # the patient sees its own case, the reader the exam with the tests, the moderator both diagnoses
        self.assertIn("35-year-old female", patient.calls[0][0])
        self.assertIn("Acetylcholine_Receptor_Antibodies", measurement.calls[0][0])
        self.assertIn("Myasthenia gravis", moderator.calls[0][1])
        self.assertIn("DIAGNOSIS READY: Myasthenia gravis", moderator.calls[0][1])

    def test_the_moderator_saying_no_scores_0(self):
        sample = _run(
            _outputs("DIAGNOSIS READY: Migraine"), _Recorder(), _Recorder(), _Recorder("No")
        )
        self.assertEqual(sample.scores["moderator_scorer"].value, "I")

    def test_no_diagnosis_within_the_turn_limit_scores_0_without_asking_the_moderator(self):
        moderator = _Recorder("Yes")
        doctor = _Recorder("Tell me more.", "And then?", "Anything else?")
        sample = _run(doctor, _Recorder("a", "b", "c"), _Recorder(), moderator, max_turns=3)
        self.assertEqual(sample.scores["moderator_scorer"].value, "I")
        self.assertEqual(moderator.calls, [])
        self.assertEqual(sample.metadata["turns"], 3)
        # the last turn tells the doctor to give a diagnosis
        self.assertIn(ac.FINAL_TURN, doctor.calls[2][1])
        self.assertNotIn(ac.FINAL_TURN, doctor.calls[1][1])
        # the doctor's prompt counts its questions
        self.assertIn("You have asked 2 questions so far", doctor.calls[2][0])

    def test_the_patient_never_sees_the_doctors_thinking(self):
        patient = _Recorder("It started a month ago.")
        sample = _run(
            _outputs(
                "<|inner_prefix|>Could be MG.<|inner_suffix|>When did it start?",
                "<think>ptosis</think>DIAGNOSIS READY: Myasthenia gravis",
            ),
            patient,
            _Recorder(),
            _Recorder("Yes"),
        )
        self.assertNotIn("Could be MG", patient.calls[0][1])
        self.assertIn("When did it start?", patient.calls[0][1])
        self.assertEqual(sample.metadata["diagnosis"], "DIAGNOSIS READY: Myasthenia gravis")
        self.assertEqual(sample.scores["moderator_scorer"].value, "C")


class HelperTests(unittest.TestCase):
    def test_strip_reasoning(self):
        self.assertEqual(ac.strip_reasoning("<think>x</think> hi"), "hi")
        self.assertEqual(ac.strip_reasoning("<|inner_prefix|>never closed"), "")
        self.assertEqual(ac.strip_reasoning("plain"), "plain")

    def test_is_yes(self):
        for yes in ("Yes", "yes.", " YES ", "<think>hm</think>Yes"):
            self.assertTrue(ac.is_yes(yes), yes)
        for no in ("No", "Yes, mostly", ""):
            self.assertFalse(ac.is_yes(no), no)

    def test_the_reader_gets_the_exam_with_the_tests(self):
        sample = ac.case_to_sample(CASE, 0)
        self.assertEqual(sample.target, "Myasthenia gravis")
        self.assertEqual(sample.metadata["exam_info"]["tests"], {"Acetylcholine_Receptor_Antibodies": "Positive"})


class DatasetTests(unittest.TestCase):
    def _cached(self, content: bytes, name="agentclinic_medqa.jsonl"):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "agentclinic", ac.SOURCE_COMMIT, name)
        os.makedirs(os.path.dirname(path))
        with open(path, "wb") as f:
            f.write(content)
        return mock.patch.dict(os.environ, {"XDG_CACHE_HOME": tmp.name})

    def test_a_cached_file_with_the_pinned_checksum_loads(self):
        content = (json.dumps(CASE) + "\n").encode()
        sha = hashlib.sha256(content).hexdigest()
        with self._cached(content), mock.patch.dict(ac.DATASETS, {"medqa": ("agentclinic_medqa.jsonl", sha)}):
            self.assertEqual(ac.load_cases("medqa"), [CASE])

    def test_a_file_with_another_checksum_is_refused(self):
        with self._cached(b"{}\n"), self.assertRaisesRegex(RuntimeError, "sha256"):
            ac.load_cases("medqa")

    def test_unknown_dataset(self):
        with self.assertRaisesRegex(ValueError, "medqa"):
            ac.load_cases("nejm")


if __name__ == "__main__":
    unittest.main()
