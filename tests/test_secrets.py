import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

HELPER = Path(__file__).resolve().parents[1] / "scripts" / "common" / "secrets.sh"
NAMES = ("A", "B", "C", "D", "E")


def load(secrets_env, env=None):
    """Source a copy of secrets.sh next to the given scripts/secrets.env; returns (variables, stderr)."""
    with tempfile.TemporaryDirectory() as root:
        Path(root, "scripts", "common").mkdir(parents=True)
        shutil.copy(HELPER, Path(root, "scripts", "common", "secrets.sh"))
        if secrets_env is not None:
            Path(root, "scripts", "secrets.env").write_text(secrets_env)
        script = 'set -eu; source scripts/common/secrets.sh; ' + \
            "; ".join(f'echo "{n}=${{{n}:-}}"' for n in NAMES) + '; env | grep -c "^A=" || true'
        base = {k: v for k, v in os.environ.items() if k not in NAMES}
        completed = subprocess.run(["bash", "-c", script], cwd=root, env=base | (env or {}),
                                   text=True, capture_output=True, check=True)
    lines = completed.stdout.splitlines()
    return dict(line.split("=", 1) for line in lines[:-1]) | {"exported_A": lines[-1]}, completed.stderr


class SecretsTest(unittest.TestCase):
    def test_reads_values_comments_quotes_and_crlf(self):
        values, err = load('# comment\n\nA=plain\nB="double quoted"\r\nC=\'single\'\nD=x=y#z')
        self.assertEqual((values["A"], values["B"], values["C"], values["D"]),
                         ("plain", "double quoted", "single", "x=y#z"))
        self.assertEqual(err, "")

    def test_environment_wins_and_nothing_is_exported(self):
        values, _ = load("A=file\nB=file\n", env={"B": "env"})
        self.assertEqual((values["A"], values["B"], values["exported_A"]), ("file", "env", "0"))

    def test_malformed_line_is_skipped_without_printing_it(self):
        values, err = load("export A=secret123\nB=ok\n")
        self.assertEqual((values["A"], values["B"]), ("", "ok"))
        self.assertIn("line 1", err)
        self.assertNotIn("secret123", err)

    def test_missing_file_is_fine(self):
        values, err = load(None)
        self.assertEqual((values["A"], err), ("", ""))


if __name__ == "__main__":
    unittest.main()
