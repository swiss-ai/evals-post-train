import unittest
from types import SimpleNamespace

from scripts.task_output_types import Classifier


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


if __name__ == "__main__":
    unittest.main()
