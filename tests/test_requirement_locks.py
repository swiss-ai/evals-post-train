"""The requirement inputs that must agree, and which a lock can't check by itself."""

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class HarborVersionTest(unittest.TestCase):
    def test_pin_adapter_tag_and_runner_agree(self) -> None:
        pin = re.search(
            r"^harbor==(\S+)$",
            (REPO_ROOT / "requirements/harbor-runtime.txt").read_text(),
            re.MULTILINE,
        )
        tag = re.search(
            r"harbor@v([^#]+)#", (REPO_ROOT / "requirements/harbor-adapters.txt").read_text()
        )
        runner = re.search(
            r'^HARBOR_VERSION = "([^"]+)"$',
            (REPO_ROOT / "scripts/harbor/run.py").read_text(),
            re.MULTILINE,
        )
        self.assertEqual({pin[1], tag[1]}, {runner[1]})


class Gaia2CommitTest(unittest.TestCase):
    def test_runtime_runner_and_images_agree(self) -> None:
        pins = set(
            re.findall(
                r"meta-agents-research-environments@([0-9a-f]{40})#",
                (REPO_ROOT / "requirements/gaia2-runtime.txt").read_text(),
            )
        )
        runner = re.search(
            r'^ARE_COMMIT = "([0-9a-f]{40})"$', (REPO_ROOT / "scripts/gaia2/run.py").read_text(), re.MULTILINE
        )
        images = re.search(
            r"^  ARE_COMMIT: ([0-9a-f]{40})$",
            (REPO_ROOT / ".github/workflows/gaia2-images.yml").read_text(),
            re.MULTILINE,
        )
        self.assertEqual(pins | {runner[1], images[1]}, {runner[1]})


if __name__ == "__main__":
    unittest.main()
