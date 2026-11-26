"""AgentClinic (Schmidgall et al., 2024, arXiv:2405.07960): diagnosis by dialogue in a
simulated clinic, run via
`inspect eval scripts/inspect_ai/tasks/agentclinic.py -T helper_model=<model>`.

The model under test is the doctor. It talks to a patient for at most `max_turns` turns,
can ask a measurement reader for test results ("REQUEST TEST: ..."), and ends with
"DIAGNOSIS READY: ...". A moderator then says whether that diagnosis is the case's correct
one; the score is the share of cases it says Yes to. A case with no diagnosis within
`max_turns` scores 0, as in the paper.

The patient, measurement reader and moderator are fixed helper models, not the model
under test: `helper_model` plays all three, `moderator_model` replaces it as the moderator.
A helper on the gateway is `openai-api/gateway/<served model name>` (as in
simpleqa_verified_gateway.py: https://api.swissai.svc.cscs.ch/v1 with the job's
CSCS_SERVING_API key; GATEWAY_BASE_URL / GATEWAY_API_KEY override them). The paper used
GPT-4 for all three, so scores depend on the helpers: compare models with the same ones.

Cases are the paper's text-only sets, pinned to the source repository's commit with their
checksums: `medqa` (AgentClinic-MedQA, 107 cases) and `medqa_ext` (215). NEJM needs images
and MIMIC-IV needs PhysioNet credentials; neither is included.

The prompts, the turn loop and the moderator question are the source's agentclinic.py
(github.com/SamuelSchmidgall/AgentClinic, MIT) without biases or images. Two changes:

- Reasoning left inline in a reply (`<|inner_prefix|>...<|inner_suffix|>` or
  `<think>...</think>`) is removed before anyone else sees it, so the patient never reads
  the doctor's thinking. A reply whose thinking never closed counts as empty.
- The moderator's answer counts as Yes after trimming spaces and a final period ("Yes."),
  where the source needs exactly "yes".
"""

import hashlib
import json
import os
import re
import urllib.request

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    Model,
    get_model,
)
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver

SOURCE_REPO = "SamuelSchmidgall/AgentClinic"
SOURCE_COMMIT = "b6fbe22300e99a267a7ac94eaa465ab552eef741"
DATASETS = {
    "medqa": (
        "agentclinic_medqa.jsonl",
        "d91038a2984f21bb1d43edd88c7958d090ef42ba80f5be487b22b903bf3a35ea",
    ),
    "medqa_ext": (
        "agentclinic_medqa_extended.jsonl",
        "d35d276481f3810f0fcb920c72f10b629ec4266d85ce5f8db1ae56467c9a5a34",
    ),
}
GATEWAY_BASE_URL = "https://api.swissai.svc.cscs.ch/v1"

DIAGNOSIS_READY = "DIAGNOSIS READY"
REQUEST_TEST = "REQUEST TEST"

# The source's prompts, word for word (typos included), without the bias and image parts.
DOCTOR_SYSTEM = (
    "You are a doctor named Dr. Agent who only responds in the form of dialogue. You are "
    "inspecting a patient who you will ask questions in order to understand their disease. "
    "You are only allowed to ask {max_turns} questions total before you must make a "
    "decision. You have asked {asked} questions so far. You can request test results using "
    'the format "REQUEST TEST: [test]". For example, "REQUEST TEST: Chest_X-Ray". Your '
    "dialogue will only be 1-3 sentences in length. Once you have decided to make a "
    'diagnosis please type "DIAGNOSIS READY: [diagnosis here]"'
    "\n\nBelow is all of the information you have. {info}. \n\n Remember, you must "
    "discover their disease by asking them questions. You are also able to provide exams."
)
PATIENT_SYSTEM = (
    "You are a patient in a clinic who only responds in the form of dialogue. You are being "
    "inspected by a doctor who will ask you questions and will perform exams on you in "
    "order to understand your disease. Your answer will only be 1-3 sentences in length."
    "\n\nBelow is all of your information. {info}. \n\n Remember, you must not reveal your "
    "disease explicitly but may only convey the symptoms you have in the form of dialogue "
    "if you are asked."
)
MEASUREMENT_SYSTEM = (
    "You are an measurement reader who responds with medical test results. Please respond "
    'in the format "RESULTS: [results here]"'
    "\n\nBelow is all of the information you have. {info}. \n\n If the requested results "
    "are not in your data then you can respond with NORMAL READINGS."
)
MODERATOR_SYSTEM = (
    "You are responsible for determining if the corrent diagnosis and the doctor diagnosis "
    "are the same disease. Please respond only with Yes or No. Nothing else."
)
FINAL_TURN = "This is the final question. Please provide a diagnosis.\n"

_THINK_PAIRS = (("<|inner_prefix|>", "<|inner_suffix|>"), ("<think>", "</think>"))


def strip_reasoning(text: str) -> str:
    """The reply without inline thinking; "" if the thinking never closed."""
    for start, end in _THINK_PAIRS:
        text = re.sub(re.escape(start) + r".*?" + re.escape(end), "", text, flags=re.DOTALL)
        if end in text:
            text = text.rsplit(end, 1)[1]
        if start in text:
            text = text.split(start, 1)[0]
    return text.strip()


def is_yes(answer: str) -> bool:
    return strip_reasoning(answer).strip().rstrip(".").strip().lower() == "yes"


def _cache_dir() -> str:
    root = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(root, "agentclinic", SOURCE_COMMIT)


def load_cases(dataset: str) -> list[dict]:
    """The pinned case file, downloaded once and checked against its checksum."""
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {sorted(DATASETS)} (got {dataset!r})")
    name, sha256 = DATASETS[dataset]
    path = os.path.join(_cache_dir(), name)
    if not os.path.exists(path):
        url = f"https://raw.githubusercontent.com/{SOURCE_REPO}/{SOURCE_COMMIT}/{name}"
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".part"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    with open(path, "rb") as f:
        data = f.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        raise RuntimeError(f"{name}: sha256 {digest}, expected {sha256}")
    return [json.loads(line) for line in data.decode().splitlines() if line.strip()]


def case_to_sample(case: dict, index: int) -> Sample:
    osce = case["OSCE_Examination"]
    # The measurement reader sees the physical exam with the test results under "tests".
    exams = {**osce["Physical_Examination_Findings"], "tests": osce["Test_Results"]}
    return Sample(
        id=index,
        input=osce["Objective_for_Doctor"],
        target=osce["Correct_Diagnosis"],
        metadata={
            "doctor_info": osce["Objective_for_Doctor"],
            "patient_info": osce["Patient_Actor"],
            "exam_info": exams,
        },
    )


async def _reply(model: Model, system: str, user: str) -> str:
    output = await model.generate([ChatMessageSystem(content=system), ChatMessageUser(content=user)])
    return strip_reasoning(output.completion)


@solver
def clinic_dialogue(max_turns: int = 20) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        doctor = get_model()
        patient = get_model(role="patient")
        measurement = get_model(role="measurement")
        info = state.metadata
        patient_system = PATIENT_SYSTEM.format(info=info["patient_info"])
        measurement_system = MEASUREMENT_SYSTEM.format(info=info["exam_info"])

        doctor_hist = patient_hist = measurement_hist = ""
        pi_dialogue = ""
        transcript = []
        diagnosis = None
        tests = 0
        turns = 0
        for turn in range(max_turns):
            if turn == max_turns - 1:
                pi_dialogue += FINAL_TURN
            system = DOCTOR_SYSTEM.format(max_turns=max_turns, asked=turn, info=info["doctor_info"])
            said = await _reply(
                doctor,
                system,
                "\nHere is a history of your dialogue: " + doctor_hist
                + "\n Here was the patient response: " + pi_dialogue
                + "Now please continue your dialogue\nDoctor: ",
            )
            doctor_hist += pi_dialogue + "\n\n" + said + "\n\n"
            transcript.append(ChatMessageAssistant(content=said or "(empty)"))
            turns = turn + 1
            if DIAGNOSIS_READY in said:
                diagnosis = said
                break
            if REQUEST_TEST in said:
                tests += 1
                pi_dialogue = await _reply(
                    measurement,
                    measurement_system,
                    "\nHere is a history of the dialogue: " + measurement_hist
                    + "\n Here was the doctor measurement request: " + said,
                )
                measurement_hist += said + "\n\n" + pi_dialogue + "\n\n"
                patient_hist += pi_dialogue + "\n\n"
                transcript.append(ChatMessageUser(content=f"[Measurement] {pi_dialogue}"))
            else:
                pi_dialogue = await _reply(
                    patient,
                    patient_system,
                    "\nHere is a history of your dialogue: " + patient_hist
                    + "\n Here was the doctor response: " + said
                    + "Now please continue your dialogue\nPatient: ",
                )
                patient_hist += said + "\n\n" + pi_dialogue + "\n\n"
                measurement_hist += pi_dialogue + "\n\n"
                transcript.append(ChatMessageUser(content=f"[Patient] {pi_dialogue}"))

        state.messages = transcript
        state.metadata.update({"diagnosis": diagnosis, "turns": turns, "tests_requested": tests})
        state.output.completion = diagnosis or ""
        return state

    return solve


@scorer(metrics=[accuracy(), stderr()])
def moderator_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        diagnosis = state.metadata.get("diagnosis")
        if not diagnosis:
            return Score(value=INCORRECT, answer="", explanation="no diagnosis within the turn limit")
        verdict = await get_model(role="moderator").generate(
            [
                ChatMessageSystem(content=MODERATOR_SYSTEM),
                ChatMessageUser(
                    content="\nHere is the correct diagnosis: " + target.text
                    + "\n Here was the doctor dialogue: " + diagnosis
                    + "\nAre these the same?"
                ),
            ]
        )
        return Score(
            value=CORRECT if is_yes(verdict.completion) else INCORRECT,
            answer=diagnosis,
            explanation=verdict.completion,
        )

    return score


def _configure_gateway_provider(model: str) -> None:
    if not model.startswith("openai-api/gateway/"):
        return
    os.environ.setdefault("GATEWAY_BASE_URL", GATEWAY_BASE_URL)
    key = os.environ.get("GATEWAY_API_KEY") or os.environ.get("CSCS_SERVING_API")
    if not key:
        raise RuntimeError(
            f"{model} uses the `gateway` provider but neither GATEWAY_API_KEY nor "
            "CSCS_SERVING_API is set"
        )
    os.environ["GATEWAY_API_KEY"] = key


@task
def agentclinic(
    helper_model: str,
    moderator_model: str | None = None,
    dataset: str = "medqa",
    max_turns: int = 20,
) -> Task:
    """
    Args:
        helper_model: Plays the patient and the measurement reader, and the moderator unless
            `moderator_model` is set (required, no default).
        moderator_model: Judges whether the diagnosis is right (default: `helper_model`).
        dataset: "medqa" (107 cases, the paper's) or "medqa_ext" (215).
        max_turns: The doctor's turns before the case ends without a diagnosis (paper: 20).
    """
    moderator_model = moderator_model or helper_model
    for model in (helper_model, moderator_model):
        _configure_gateway_provider(model)
    helper = get_model(helper_model)
    cases = load_cases(dataset)
    return Task(
        dataset=MemoryDataset([case_to_sample(c, i) for i, c in enumerate(cases)], name=f"agentclinic_{dataset}"),
        solver=clinic_dialogue(max_turns=max_turns),
        scorer=moderator_scorer(),
        model_roles={
            "patient": helper,
            "measurement": helper,
            "moderator": get_model(moderator_model),
        },
        version="1.0.0",
        metadata={"source": f"{SOURCE_REPO}@{SOURCE_COMMIT}", "dataset": dataset},
    )
