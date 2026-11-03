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


if __name__ == "__main__":
    unittest.main()
