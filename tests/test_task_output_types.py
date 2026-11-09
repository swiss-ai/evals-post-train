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
        "inline_group": _entry("GROUP", {"task": ["gsm8k", {"task": "custom", "dataset_path": "x"}]}),
        "py_task": _entry("PY_TASK"),
    }


class ClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = Classifier(_TaskManager(), load_yaml=None)

    def test_kinds_follow_the_output_type(self) -> None:
        self.assertEqual(self.classifier.kinds("gsm8k"), {False})
        self.assertEqual(self.classifier.kinds("ifeval"), {False})
        self.assertEqual(self.classifier.kinds("arc_easy"), {True})
        self.assertEqual(self.classifier.kinds("wikitext"), {True})

    def test_groups_and_tags_follow_their_members(self) -> None:
        self.assertEqual(self.classifier.kinds("mc_group"), {True})
        self.assertEqual(self.classifier.kinds("ai2_arc"), {True})
        self.assertEqual(self.classifier.kinds("mixed_group"), {True, False})

    def test_inline_group_tasks_take_lm_evals_names(self) -> None:
        self.assertEqual(self.classifier.leaves("inline_group"), {"gsm8k", "inline_group::custom"})

    def test_unknown_names_have_no_kind(self) -> None:
        self.assertEqual(self.classifier.kinds("not_a_task"), set())
        self.assertEqual(self.classifier.kinds("py_task"), set())

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


if __name__ == "__main__":
    unittest.main()
