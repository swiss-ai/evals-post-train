import os
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (REPO_ROOT / "scripts/inspect_ai/run_inspect_eval.sh").read_text()
BLOCK = SCRIPT[
    SCRIPT.index("# Sampling of the model under test (scripts/common/sampling.py"):
    SCRIPT.index('COMMON_ARGS+=(${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})')
]


def _args(extra_args=(), **flags: str) -> subprocess.CompletedProcess:
    """The `inspect eval` arguments run_inspect_eval.sh's sampling block adds (COMMON_ARGS)."""
    names = {
        "temperature": "SAMPLING_TEMPERATURE_FLAG", "top_p": "SAMPLING_TOP_P_FLAG", "top_k": "SAMPLING_TOP_K_FLAG",
        "min_p": "SAMPLING_MIN_P_FLAG", "repetition_penalty": "SAMPLING_REPETITION_PENALTY_FLAG",
        "profile": "SAMPLING_PROFILE_FLAG", "config": "SAMPLING_CONFIG_FLAG",
        "thinking_switch": "THINKING_ENABLE_FLAG", "reasoning_effort": "THINKING_EFFORT_FLAG",
    }
    env = {"PATH": os.environ["PATH"]} | {variable: flags.get(key, "") for key, variable in names.items()}
    script = (
        "COMMON_ARGS=()\nEXTRA_ARGS=(" + " ".join(f"'{arg}'" for arg in extra_args) + ")\n"
        + BLOCK
        + 'printf "%s\\n" "${COMMON_ARGS[@]}"\n'
    )
    return subprocess.run(["bash", "-c", script], cwd=REPO_ROOT, env=env, capture_output=True, text=True)


class InspectSamplingTest(unittest.TestCase):
    def test_defaults(self):
        done = _args()
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.split(), ["--temperature", "0.6", "--top-p", "0.95"])

    def test_flags(self):
        done = _args(temperature="1", top_p="none", top_k="20", min_p="0.05", repetition_penalty="1.05")
        self.assertEqual(
            done.stdout.splitlines(),
            ["--temperature", "1.0", "--top-k", "20", "--extra-body",
             '{"min_p": 0.05, "repetition_penalty": 1.05}'],
        )

    def test_profile_and_greedy_warning(self):
        done = _args(profile="greedy")
        self.assertEqual(done.stdout.split(), ["--temperature", "0.0"])
        self.assertIn("greedy decoding", done.stderr)
        self.assertNotIn("greedy decoding", _args().stderr)

    def test_thinking_flags_pick_the_sampling_profile(self):
        off = _args(thinking_switch="--no-enable-thinking").stdout.split()
        self.assertEqual(off[:2], ["--temperature", "0.0"])
        self.assertNotIn("--top-p", off)
        on = _args(thinking_switch="--enable-thinking").stdout.split()
        self.assertEqual(on[:4], ["--temperature", "0.6", "--top-p", "0.95"])
        self.assertEqual(_args(reasoning_effort="none").stdout.split()[:2], ["--temperature", "0.0"])
        # An explicit sampling flag still wins over the mode.
        self.assertEqual(_args(thinking_switch="--no-enable-thinking", temperature="0.4").stdout.split()[:2],
                         ["--temperature", "0.4"])

    def test_a_literal_extra_body_is_called_out(self):
        done = _args(extra_args=("--extra-body", "{}"), min_p="0.1")
        self.assertIn("replaces the min-p", done.stderr)

    def test_invalid_value_stops_the_run(self):
        done = _args(top_p="3")
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("top_p must be in (0, 1]", done.stderr)
