import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class InspectSandboxPatchPathTest(unittest.TestCase):
    def test_pythonpath_points_at_the_sitecustomize_patch(self) -> None:
        script = (REPO / "scripts/inspect_ai/run_inspect_eval.sh").read_text()
        match = re.search(r'export PYTHONPATH="\$\(pwd\)/([^$"]+)', script)
        self.assertIsNotNone(match, "run_inspect_eval.sh no longer exports PYTHONPATH")
        self.assertTrue((REPO / match.group(1) / "sitecustomize.py").is_file(), match.group(1))


if __name__ == "__main__":
    unittest.main()
