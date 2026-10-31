import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.lm_eval_harness.task_output_types import Classifier

REPO_ROOT = Path(__file__).resolve().parents[1]

# A stand-in lm_eval package for the command line: two tasks, one scored by log-likelihood.
_FAKE_LM_EVAL = {
    "lm_eval/__init__.py": "",
    "lm_eval/tasks/_yaml_loader.py": "load_yaml = None\n",
    "lm_eval/tasks/__init__.py": (
        "from types import SimpleNamespace as N\n"
        "class TaskManager:\n"
        "    task_index = {name: N(kind=N(name='TASK'), yaml_path=None, cfg={'output_type': kind}, tags=())\n"
        "                  for name, kind in [('gsm8k', 'generate_until'), ('hellaswag', 'multiple_choice')]}\n"
    ),
}


def _entry(kind, cfg=None, tags=()):
    return SimpleNamespace(kind=SimpleNamespace(name=kind), yaml_path=None, cfg=cfg, tags=set(tags))


class _TaskManager:
    task_index = {
        "arc_easy": _entry("TASK", {"output_type": "multiple_choice"}, tags=["ai2_arc"]),
        "arc_challenge": _entry("TASK", {"output_type": "multiple_choice"}, tags=["ai2_arc"]),
        "wikitext": _entry("TASK", {"output_type": "loglikelihood_rolling"}),
        "gsm8k": _entry("TASK", {"output_type": "generate_until"}),
        "ifeval": _entry("TASK", {"dataset_path": "x"}),  # no output_type: lm-eval's default
        "ai2_arc": _entry("TAG"),
        "mc_group": _entry("GROUP", {"task": ["arc_easy", {"task": "arc_challenge"}]}),
        "mixed_group": _entry("GROUP", {"task": ["arc_easy", "gsm8k"]}),
        "py_task": _entry("PY_TASK"),
    }


class ClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = Classifier(_TaskManager(), load_yaml=None)

    def test_splits_in_input_order(self) -> None:
        chat, raw = self.classifier.split(["gsm8k", "arc_easy", "ifeval", "wikitext"])
        self.assertEqual(chat, ["gsm8k", "ifeval"])
        self.assertEqual(raw, ["arc_easy", "wikitext"])

    def test_groups_and_tags_follow_their_members(self) -> None:
        chat, raw = self.classifier.split(["mc_group", "ai2_arc", "mixed_group"])
        self.assertEqual(raw, ["mc_group", "ai2_arc"])
        # A mix keeps the chat template, the launcher's default.
        self.assertEqual(chat, ["mixed_group"])

    def test_unknown_names_keep_the_chat_template(self) -> None:
        chat, raw = self.classifier.split(["not_a_task", "py_task", "arc_easy"])
        self.assertEqual(chat, ["not_a_task", "py_task"])
        self.assertEqual(raw, ["arc_easy"])

    def test_any_loglikelihood_member_is_reported(self) -> None:
        # A generative-only model (CHAT_FORMAT=apertus2) cannot run a mixed group either.
        tasks = ["gsm8k", "ifeval", "mc_group", "mixed_group", "ai2_arc", "wikitext", "not_a_task"]
        self.assertEqual(
            self.classifier.with_loglikelihood(tasks),
            ["mc_group", "mixed_group", "ai2_arc", "wikitext"],
        )


class RequireGenerativeTests(unittest.TestCase):
    def _run(self, *arguments: str) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as tmp:
            for path, text in _FAKE_LM_EVAL.items():
                (Path(tmp) / path).parent.mkdir(parents=True, exist_ok=True)
                (Path(tmp) / path).write_text(text)
            return subprocess.run(
                [sys.executable, "-m", "scripts.lm_eval_harness.task_output_types", *arguments],
                cwd=REPO_ROOT, env={"PYTHONPATH": tmp}, capture_output=True, text=True,
            )

    def test_stops_on_a_loglikelihood_task(self) -> None:
        refused = self._run("--tasks", "gsm8k,hellaswag", "--require-generative")
        self.assertEqual(refused.returncode, 1)
        self.assertIn("cannot do: hellaswag", refused.stderr)
        accepted = self._run("--tasks", "gsm8k", "--require-generative")
        self.assertEqual(accepted.returncode, 0, accepted.stderr)

    def test_checking_and_writing_are_exclusive(self) -> None:
        self.assertEqual(self._run("--tasks", "gsm8k", "--require-generative", "--output", "x").returncode, 2)


if __name__ == "__main__":
    unittest.main()
