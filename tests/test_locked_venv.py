"""scripts/common/locked_venv.sh: builds a lock's venv once, and a new one when the lock changes."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER = REPO_ROOT / "scripts/common/locked_venv.sh"


class LockedVenvTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.lock = self.tmp / "demo-runtime.lock"
        self.lock.write_text("# no packages\n")
        self.cache = self.tmp / "cache"
        # Only python3 and the basic tools: the pip path, without uv and offline.
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        for tool in ("python3", "basename", "cat", "cut", "date", "mkdir", "mv", "shasum", "sha256sum", "touch", "uname"):
            found = shutil.which(tool)
            if found:
                (self.bin / tool).symlink_to(found)

    def _locked_venv(self) -> str:
        script = f'source "{HELPER}"; locked_venv "{self.lock}" 3.11 || exit 1; echo "VENV=$LOCKED_VENV"'
        result = subprocess.run(
            ["/bin/bash", "-c", script],
            env={"PATH": str(self.bin), "EVAL_ENV_CACHE_ROOT": str(self.cache), "HOME": str(self.tmp)},
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip().rsplit("VENV=", 1)[1]

    def test_built_under_the_cache_root_then_reused(self) -> None:
        venv = Path(self._locked_venv())
        self.assertEqual(venv.parent, self.cache / "demo-runtime")
        self.assertTrue(os.access(venv / "bin/python", os.X_OK))
        sentinel = venv / "sentinel"
        sentinel.touch()
        self.assertEqual(self._locked_venv(), str(venv))
        self.assertTrue(sentinel.exists())

    def test_a_changed_lock_gets_its_own_venv(self) -> None:
        first = self._locked_venv()
        self.lock.write_text("# no packages, still\n")
        self.assertNotEqual(self._locked_venv(), first)

    def test_an_interrupted_build_is_set_aside_and_rebuilt(self) -> None:
        venv = Path(self._locked_venv())
        (venv / ".complete").unlink()
        self.assertEqual(self._locked_venv(), str(venv))
        self.assertTrue((venv / ".complete").exists())
        self.assertEqual(len(list(venv.parent.glob(f"{venv.name}.incomplete.*"))), 1)


if __name__ == "__main__":
    unittest.main()
