import os
import subprocess
import tempfile
import unittest
from pathlib import Path

HELPER = Path(__file__).resolve().parents[1] / "scripts" / "common" / "wandb.sh"


def wandb_setup(*args, env=None, key_file=None):
    """Run wandb_setup in a scratch repo root; returns (exit code, stdout, the resulting settings)."""
    with tempfile.TemporaryDirectory() as root:
        if key_file is not None:
            Path(root, "scripts").mkdir()
            Path(root, "scripts", "wandb_api_key.txt").write_text(key_file)
        script = (f'source "{HELPER}"; wandb_setup "$@" || exit $?; '
                  'echo "UPLOAD=$WANDB_UPLOAD ENTITY=${WANDB_ENTITY:-} PROJECT=${WANDB_PROJECT:-} KEY=${WANDB_API_KEY:-}"')
        base = {k: v for k, v in os.environ.items() if not k.startswith("WANDB_")}
        completed = subprocess.run(["bash", "-c", script, "wandb_setup", *args], cwd=root, env=base | (env or {}),
                                   text=True, capture_output=True)
    settings = dict(item.split("=", 1) for item in completed.stdout.strip().splitlines()[-1].split()) \
        if completed.returncode == 0 else {}
    return completed.returncode, completed.stdout, settings


class WandbSetupTest(unittest.TestCase):
    def test_uploads_to_the_defaults_when_a_key_is_available(self):
        code, _, s = wandb_setup(env={"WANDB_API_KEY": "k"})
        self.assertEqual((code, s["UPLOAD"], s["ENTITY"], s["PROJECT"]),
                         (0, "1", "apertus", "apertus-1.5-post-training-v0.1"))

    def test_key_file(self):
        code, _, s = wandb_setup(key_file="filekey\n")
        self.assertEqual((s["UPLOAD"], s["KEY"]), ("1", "filekey"))

    def test_given_entity_or_project_overrides_one_default(self):
        _, _, s = wandb_setup(env={"WANDB_API_KEY": "k", "WANDB_PROJECT": "p"})
        self.assertEqual((s["ENTITY"], s["PROJECT"]), ("apertus", "p"))

    def test_no_key_skips_unless_an_upload_was_asked_for(self):
        code, out, s = wandb_setup()
        self.assertEqual((code, s["UPLOAD"]), (0, "0"))
        self.assertIn("no WANDB_API_KEY", out)
        code, _, _ = wandb_setup(env={"WANDB_ENTITY": "e"})
        self.assertEqual(code, 1)

    def test_skips(self):
        for args, env in (((), {"WANDB_MODE": "disabled"}), (("--no-wandb",), {})):
            code, out, s = wandb_setup(*args, env={"WANDB_API_KEY": "k"} | env)
            self.assertEqual((code, s["UPLOAD"]), (0, "0"))
            self.assertIn("W&B upload: off", out)


if __name__ == "__main__":
    unittest.main()
