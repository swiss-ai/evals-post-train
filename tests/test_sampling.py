import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.common import sampling

ROOT = Path(__file__).resolve().parents[1]


def _cli(*arguments, **env):
    return subprocess.run(
        [sys.executable, "-m", "scripts.common.sampling", *arguments],
        cwd=ROOT,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
    )


class SamplingConfigTest(unittest.TestCase):
    def test_shipped_defaults(self):
        config = sampling.read_config()
        for framework in ("lm_eval", "inspect", "evalscope", "harbor"):
            self.assertEqual(
                sampling.resolve(framework, config=config), {"temperature": 0.6, "top_p": 0.95}
            )
        # tau2 keeps its own behaviour until the protocol's value is confirmed.
        self.assertEqual(sampling.resolve("tau2", config=config), {})

    def test_precedence_override_profile_default(self):
        config = sampling.read_config()
        self.assertEqual(
            sampling.resolve("inspect", config=config, overrides={"temperature": 1.0}),
            {"temperature": 1.0, "top_p": 0.95},
        )
        self.assertEqual(sampling.resolve("inspect", config=config, profile="greedy"), {"temperature": 0.0})
        self.assertEqual(
            sampling.resolve("inspect", config=config, profile="greedy", overrides={"top_k": 20}),
            {"temperature": 0.0, "top_k": 20},
        )

    def test_none_omits_a_sampler(self):
        config = sampling.read_config()
        self.assertEqual(
            sampling.resolve("inspect", config=config, overrides={"temperature": 1.0, "top_p": None}),
            {"temperature": 1.0},
        )

    def test_no_default_keeps_only_explicit_settings(self):
        config = sampling.read_config()
        self.assertEqual(sampling.resolve("lm_eval", config=config, use_default=False), {})
        self.assertEqual(
            sampling.resolve("lm_eval", config=config, use_default=False, overrides={"temperature": 0.7}),
            {"temperature": 0.7},
        )

    def test_parse_value(self):
        self.assertEqual(sampling.parse_value("temperature", "1"), 1.0)
        self.assertEqual(sampling.parse_value("top_k", "20"), 20)
        self.assertIsInstance(sampling.parse_value("top_k", "20"), int)
        self.assertIsNone(sampling.parse_value("top_p", "none"))
        self.assertIsNone(sampling.parse_value("top_p", "None"))
        for name, bad in [("temperature", "-1"), ("top_p", "0"), ("top_p", "1.5"), ("top_k", "2.5"),
                          ("min_p", "2"), ("repetition_penalty", "0"), ("temperature", "hot")]:
            with self.assertRaises(ValueError, msg=f"{name}={bad}"):
                sampling.parse_value(name, bad)

    def test_config_is_strict(self):
        def write(content):
            handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
            handle.write(json.dumps(content))
            handle.close()
            self.addCleanup(os.unlink, handle.name)
            return handle.name

        for bad in (
            {"profiles": {}, "typo": 1},
            {"profiles": {"p": {"temprature": 1}}},
            {"profiles": {"p": {"top_p": 3}}},
            {"profiles": {"p": {}}, "frameworks": {"spark": "p"}},
            {"profiles": {"p": {}}, "frameworks": {"inspect": "missing"}},
        ):
            with self.assertRaises(ValueError, msg=str(bad)):
                sampling.read_config(write(bad))
        ok = sampling.read_config(write({"profiles": {"p": {"temperature": 0.3, "top_p": None}},
                                          "frameworks": {"inspect": "p"}}))
        self.assertEqual(sampling.resolve("inspect", config=ok), {"temperature": 0.3})

    def test_unknown_profile_and_framework(self):
        config = sampling.read_config()
        with self.assertRaises(ValueError):
            sampling.resolve("inspect", config=config, profile="nope")
        with self.assertRaises(ValueError):
            sampling.resolve("spark", config=config)

    def test_greedy_warning(self):
        self.assertIn("greedy", sampling.greedy_warning({"temperature": 0}))
        self.assertIsNone(sampling.greedy_warning({"temperature": 0.6}))
        self.assertIsNone(sampling.greedy_warning({}))


class SamplingCliTest(unittest.TestCase):
    def test_default_json(self):
        done = _cli("--framework", "harbor")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(json.loads(done.stdout), {"temperature": 0.6, "top_p": 0.95})

    def test_flags_and_none(self):
        done = _cli("--framework", "inspect", "--temperature", "1", "--top-p", "none", "--min-p", "0.05")
        self.assertEqual(json.loads(done.stdout), {"min_p": 0.05, "temperature": 1.0})

    def test_env_prefix_and_flag_wins(self):
        done = _cli("--framework", "lm_eval", "--env-prefix", "THINK_", THINK_TEMPERATURE="0.8", THINK_TOP_K="40")
        self.assertEqual(json.loads(done.stdout), {"temperature": 0.8, "top_k": 40, "top_p": 0.95})
        done = _cli("--framework", "lm_eval", "--env-prefix", "THINK_", "--temperature", "0.2", THINK_TEMPERATURE="0.8")
        self.assertEqual(json.loads(done.stdout)["temperature"], 0.2)
        done = _cli("--framework", "lm_eval", "--env-prefix", "THINK_", THINK_TOP_P="none")
        self.assertEqual(json.loads(done.stdout), {"temperature": 0.6})
        done = _cli("--framework", "lm_eval", "--env-prefix", "THINK_", THINK_TEMPERATURE="")
        self.assertEqual(json.loads(done.stdout)["temperature"], 0.6)

    def test_shell_and_summary(self):
        done = _cli("--framework", "evalscope", "--top-k", "20", "--format", "shell")
        self.assertEqual(
            done.stdout.splitlines(),
            ["SAMPLING_TEMPERATURE=0.6", "SAMPLING_TOP_P=0.95", "SAMPLING_TOP_K=20",
             "SAMPLING_MIN_P=", "SAMPLING_REPETITION_PENALTY="],
        )
        done = _cli("--framework", "evalscope", "--format", "summary")
        self.assertEqual(
            done.stdout.strip(),
            "temperature=0.6, top_p=0.95, top_k=<unset>, min_p=<unset>, repetition_penalty=<unset>",
        )

    def test_litellm_format(self):
        done = _cli("--framework", "tau2", "--temperature", "1", "--top-k", "20", "--min-p", "0.05", "--format", "litellm")
        self.assertEqual(
            json.loads(done.stdout), {"temperature": 1.0, "extra_body": {"min_p": 0.05, "top_k": 20}}
        )
        self.assertEqual(json.loads(_cli("--framework", "tau2", "--format", "litellm").stdout), {})
        done = _cli("--framework", "inspect", "--format", "litellm")
        self.assertEqual(json.loads(done.stdout), {"temperature": 0.6, "top_p": 0.95})

    def test_errors_and_warning(self):
        done = _cli("--framework", "inspect", "--temperature", "-3")
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("temperature must be >= 0", done.stderr)
        done = _cli("--framework", "inspect", "--profile", "nope")
        self.assertNotEqual(done.returncode, 0)
        done = _cli("--framework", "inspect", "--temperature", "0", "--warn-greedy")
        self.assertEqual(done.returncode, 0)
        self.assertIn("greedy", done.stderr)
        self.assertNotIn("greedy", _cli("--framework", "inspect", "--temperature", "0").stderr)

    def test_output_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "run" / "sampling.json"
            done = _cli("--framework", "inspect", "--output", str(target))
            self.assertEqual(json.loads(target.read_text()), json.loads(done.stdout))


if __name__ == "__main__":
    unittest.main()
