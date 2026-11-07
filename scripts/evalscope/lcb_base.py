"""LiveCodeBench's base-model setup for EvalScope's `live_code_bench` (run.py --completions).

EvalScope's adapter prompts like an instruct model: a system message plus a chat-style question, with
the code expected in a fenced block. LiveCodeBench's official runner treats base models differently
(lcb_runner LMStyle.GenericBase): the prompt is plain text with one worked example, no system message
and no chat template, the model's reply is the code itself, and generation stops at "###".
This module supplies the two pieces the adapter needs for that: the prompt (`base_prompt`) and the
patches that make the adapter use it and extract the reply the way the official runner does
(`install`). The text goes to the server through completions_model.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

EXAMPLES: Dict[str, Any] = json.loads((Path(__file__).with_name("lcb_base_examples.json")).read_text())
# lcb_runner's --stop default, applied to every run of the official runner.
STOP = ["###"]
# lcb_runner's --max_tokens default. /v1/completions otherwise defaults to 16 tokens.
MAX_TOKENS = 2000


def _example(question: str, sample_code: Optional[str], answer: str, starter: bool) -> str:
    text = "### Question\n" + question + "\n\n"
    if starter:
        text += "### Starter Code\n" + (sample_code or "") + "\n\n"
    text += "### Answer\n\n" + answer
    if answer:
        text += "\n\n"
    return text


def base_prompt(question_content: str, starter_code: Optional[str]) -> str:
    """lcb_runner.prompts.code_generation.get_base_model_question_template_answer, for one problem."""
    starter = bool(starter_code)
    shot = EXAMPLES["func" if starter else "stdin"]
    return _example(shot["question"], shot.get("sample_code"), shot["answer"], starter) + _example(
        question_content, starter_code, "", starter
    )


def install() -> None:
    """Make EvalScope's live_code_bench adapter use the base-model prompt and extraction.

    Only the prompt (the Sample's input) and the answer extraction change; the dataset, the test
    cases and the sandboxed scoring are the adapter's own.
    """
    from evalscope.api.messages.chat_message import ChatMessageUser
    from evalscope.benchmarks.live_code_bench import extract_utils
    from evalscope.benchmarks.live_code_bench.live_code_bench_adapter import LiveCodeBenchAdapter

    if getattr(LiveCodeBenchAdapter, "_lcb_base_installed", False):
        return
    record_to_sample = LiveCodeBenchAdapter.record_to_sample
    extract_answer = LiveCodeBenchAdapter.extract_answer

    def base_record_to_sample(self, record):
        sample = record_to_sample(self, dict(record))
        sample.input = [ChatMessageUser(content=base_prompt(record["question_content"], record.get("starter_code")))]
        return sample

    def base_extract_answer(self, prediction, task_state):
        return extract_utils.extract_code_generation(prediction, model_type="base")

    LiveCodeBenchAdapter.record_to_sample = base_record_to_sample
    LiveCodeBenchAdapter.extract_answer = base_extract_answer
    LiveCodeBenchAdapter._lcb_base_installed = True
    LiveCodeBenchAdapter._lcb_original = (record_to_sample, extract_answer)


def uninstall() -> None:
    """Undo install() (tests)."""
    from evalscope.benchmarks.live_code_bench.live_code_bench_adapter import LiveCodeBenchAdapter

    if getattr(LiveCodeBenchAdapter, "_lcb_base_installed", False):
        LiveCodeBenchAdapter.record_to_sample, LiveCodeBenchAdapter.extract_answer = LiveCodeBenchAdapter._lcb_original
        LiveCodeBenchAdapter._lcb_base_installed = False
