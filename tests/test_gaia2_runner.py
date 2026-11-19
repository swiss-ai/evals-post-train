import importlib.util
import json
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

from scripts.gaia2 import run

HAS_GAIA2 = importlib.util.find_spec("gaia2_runner") is not None

MODEL_ARGS = ["--model", "SwissAI-Research/swiss-ai/chonk", "--api-base-url", "https://api.swissai.svc.cscs.ch/v1"]


def write_result(root: Path, subset: str, scenario: str, success, attempt=None, **extra):
    base = root / f"run_{attempt}" if attempt else root
    path = base / subset / scenario / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"scenario_id": scenario, "success": success, **extra}))


class ArgsTest(unittest.TestCase):
    def test_model_or_gold(self):
        with self.assertRaises(SystemExit):
            run.parse_args(["--work-dir", "w"])
        with self.assertRaises(SystemExit):  # a model needs its endpoint
            run.parse_args(["--model", "m", "--work-dir", "w"])
        with self.assertRaises(SystemExit):  # the oracle calls no model
            run.parse_args(["--gold", "--api-base-url", "https://x/v1", "--work-dir", "w"])
        args = run.parse_args(["--gold", "--work-dir", "w"])
        self.assertEqual(args.language, list(run.LANGUAGES))
        self.assertEqual(args.subset, list(run.SUBSETS))

    def test_language_and_subset_lists(self):
        args = run.parse_args([*MODEL_ARGS, "--work-dir", "w", "--language", "deu_Latn,fra_Latn,deu_Latn", "--subset", "search"])
        self.assertEqual(args.language, ["deu_Latn", "fra_Latn"])
        self.assertEqual(args.subset, ["search"])
        with self.assertRaises(SystemExit):
            run.parse_args([*MODEL_ARGS, "--work-dir", "w", "--language", "eng_Latn"])
        with self.assertRaises(SystemExit):
            run.parse_args([*MODEL_ARGS, "--work-dir", "w", "--subset", "time"])  # English-only split
        with self.assertRaises(SystemExit):
            run.parse_args([*MODEL_ARGS, "--work-dir", "w", "--limit", "0"])


class ConfigTest(unittest.TestCase):
    def config(self, argv, manifest=None):
        args = run.parse_args([*argv, "--work-dir", "w"])
        cfg = run.runner_config(args, Path("/cache/deu_Latn"), "execution", Path("/out/results"), manifest)
        return tomllib.loads(run.to_toml(cfg))

    def test_model_run(self):
        cfg = self.config([*MODEL_ARGS, "--pass-at", "3", "--thinking", "high"], Path("/out/subset.json"))
        self.assertEqual(
            cfg["target"],
            {"dataset_root": "/cache/deu_Latn", "splits": ["execution"], "subset_manifest": "/out/subset.json"},
        )
        self.assertEqual(cfg["agent"]["image"], run.AGENT_IMAGE)
        self.assertIn(run.ARE_COMMIT, run.AGENT_IMAGE)
        self.assertEqual(cfg["agent"]["provider"], "openai-compat")
        self.assertEqual(cfg["agent"]["model"], "SwissAI-Research/swiss-ai/chonk")
        self.assertEqual(cfg["agent"]["thinking"], "high")
        # Keys go by name, never into the file.
        self.assertEqual(cfg["agent"]["api_key_env"], run.MODEL_KEY_ENV)
        self.assertNotIn("api_key", cfg["agent"])
        self.assertEqual(cfg["judge"]["model"], "CSCS-Inference/zai-org/GLM-5.3")
        self.assertEqual(cfg["judge"]["prompt_version"], "omnilingual-gaia2")
        self.assertEqual(cfg["judge"]["api_key_env"], run.JUDGE_KEY_ENV)
        self.assertNotIn("extra_body", cfg["judge"])
        self.assertEqual(cfg["run"]["pass_at"], 3)
        self.assertEqual(cfg["run"]["output_dir"], "/out/results")

    def test_gold_run(self):
        cfg = self.config(["--gold", "--judge-extra-body", '{"reasoning_effort": "low", "x": {"y": true}}'])
        self.assertEqual(cfg["agent"], {"image": run.ORACLE_IMAGE, "runtime": "opensandbox"})
        self.assertNotIn("subset_manifest", cfg["target"])
        self.assertEqual(cfg["judge"]["extra_body"], {"reasoning_effort": "low", "x": {"y": True}})

    def test_toml_strings_are_escaped(self):
        self.assertEqual(tomllib.loads(run.to_toml({"t": {"s": 'a "b"\\c\n'}}))["t"]["s"], 'a "b"\\c\n')

    @unittest.skipUnless(HAS_GAIA2, "gaia2-runner is not installed (requirements/gaia2-runtime.lock)")
    def test_gaia2_runner_accepts_it(self):
        from gaia2_runner.config import load_runner_toml_config

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            "os.environ", {run.MODEL_KEY_ENV: "k", run.JUDGE_KEY_ENV: "j"}
        ):
            root = Path(tmp) / "fra_Latn"
            (root / "search").mkdir(parents=True)
            for i in range(5):
                (root / "search" / f"s{i}.json").write_text("{}")
            manifest = run.limit_manifest(root, "search", 2, Path(tmp) / "subset.json")
            for argv in (MODEL_ARGS, ["--gold"]):
                args = run.parse_args([*argv, "--work-dir", tmp, "--limit", "2"])
                path = Path(tmp) / "config.toml"
                path.write_text(run.to_toml(run.runner_config(args, root, "search", Path(tmp) / "results", manifest)))
                config = load_runner_toml_config(str(path))
                self.assertEqual(config.target.dataset_root, str(root.resolve()))
                self.assertEqual(config.target.splits, ("search",))
                self.assertEqual(config.target.subset_manifest, str(manifest.resolve()))
                self.assertEqual(config.judge.api_key, "j")
                self.assertEqual(config.judge.prompt_version, "omnilingual-gaia2")
            self.assertIsNone(config.agent.model)  # the oracle image needs none

    def test_limit_is_a_seeded_random_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "search").mkdir()
            for i in range(20):
                (root / "search" / f"s{i:02d}.json").write_text("{}")
            first = json.loads(run.limit_manifest(root, "search", 5, root / "a.json").read_text())
            again = json.loads(run.limit_manifest(root, "search", 5, root / "b.json").read_text())
            everything = json.loads(run.limit_manifest(root, "search", 50, root / "c.json").read_text())
        self.assertEqual(first, again)
        self.assertEqual(len(first["splits"]["search"]), 5)
        self.assertNotEqual(first["splits"]["search"], [f"s{i:02d}" for i in range(5)])  # not the first five
        self.assertEqual(len(everything["splits"]["search"]), 20)

    def test_materialize(self):
        rows = [{"scenario_id": "a", "scenario": '{"x": 1}'}, {"scenario_id": "b", "scenario": "{}"}]
        fake = mock.Mock(load_dataset=mock.Mock(return_value=rows))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict("sys.modules", {"datasets": fake}):
            root = run.materialize("ita_Latn", "ambiguity", Path(tmp))
            again = run.materialize("ita_Latn", "ambiguity", Path(tmp))
            self.assertEqual(root, Path(tmp) / "ita_Latn")
            self.assertEqual(again, root)
            self.assertEqual((root / "ambiguity" / "a.json").read_text(), '{"x": 1}')
            self.assertEqual(sorted(p.name for p in (Path(tmp) / "ita_Latn").iterdir()), ["ambiguity"])
        fake.load_dataset.assert_called_once_with(
            "facebook/omnilingual-gaia2", "ita_Latn_ambiguity", split="test", revision=run.DATASET_REVISION
        )

    def test_allowed_hosts(self):
        args = run.parse_args([*MODEL_ARGS, "--work-dir", "w"])
        self.assertEqual(run.allowed_hosts(args), ["api.swissai.svc.cscs.ch"])
        args = run.parse_args(["--gold", "--work-dir", "w", "--judge-base-url", "https://judge.example/v1"])
        self.assertEqual(run.allowed_hosts(args), ["judge.example"])

    def test_runner_env(self):
        args = run.parse_args([*MODEL_ARGS, "--work-dir", "w"])
        with mock.patch.dict("os.environ", {run.MODEL_KEY_ENV: "k"}, clear=True):
            env = run.runner_env(args)
        self.assertEqual(env[run.JUDGE_KEY_ENV], "k")  # the judge defaults to the model's key
        self.assertEqual(env["GAIA2_ALLOWED_HOSTS"], "api.swissai.svc.cscs.ch")
        self.assertTrue(env["OPENSANDBOX_RUN_LABEL"].startswith("gaia2-"))
        args = run.parse_args([*MODEL_ARGS, "--work-dir", "w", "--sandbox-network", "public"])
        with mock.patch.dict("os.environ", {"GAIA2_ALLOWED_HOSTS": "x", "OPENSANDBOX_RUN_LABEL": "mine"}, clear=True):
            env = run.runner_env(args)
        self.assertNotIn("GAIA2_ALLOWED_HOSTS", env)
        self.assertEqual(env["OPENSANDBOX_RUN_LABEL"], "mine")


class SummaryTest(unittest.TestCase):
    def test_summary_pass_at_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            deu = work / "runs" / "deu_Latn" / "execution" / "results"
            write_result(deu, "execution", "s1", True)
            write_result(deu, "execution", "s2", False, failure_reasons=["wrong email"])
            write_result(deu, "execution", "s3", None, error="Adapter /status endpoint unreachable")
            fra = work / "runs" / "fra_Latn" / "execution" / "results"
            write_result(fra, "execution", "s1", True)
            summary = run.summarize(work, ["deu_Latn", "fra_Latn", "ita_Latn"], ["execution", "search"])

        cell = summary["cells"]["deu_Latn/execution"]
        self.assertAlmostEqual(cell["success_rate"], 1 / 3)  # an error counts as not passed
        self.assertEqual((cell["n_passed"], cell["n_failed"], cell["n_errors"]), (1, 1, 1))
        self.assertEqual(cell["errors"], [{"scenario": "s3", "attempt": 1, "error": "Adapter /status endpoint unreachable"}])
        self.assertNotIn("pass_at_k", cell)
        self.assertEqual(set(summary["cells"]), {"deu_Latn/execution", "fra_Latn/execution"})
        self.assertEqual(set(summary["languages"]), {"deu_Latn", "fra_Latn"})  # nothing ran for ita_Latn
        self.assertEqual(set(summary["subsets"]), {"execution"})
        # Pooled: the same scenario id in two languages is two scenarios.
        self.assertEqual(summary["subsets"]["execution"]["n_scenarios"], 4)
        self.assertEqual(summary["overall"]["success_rate"], 0.5)
        self.assertEqual(summary["overall"]["errors"][0]["cell"], "deu_Latn/execution")

    def test_summary_pass_at_k(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            results = work / "runs" / "ita_Latn" / "search" / "results"
            for attempt, outcome in enumerate([False, True, None], 1):
                write_result(results, "search", "a", outcome, attempt=attempt)
            for attempt in (1, 2, 3):
                write_result(results, "search", "b", False, attempt=attempt)
            (results / "results.jsonl").write_text("not a result.json\n")
            summary = run.summarize(work, ["ita_Latn"], ["search"])

        cell = summary["cells"]["ita_Latn/search"]
        self.assertEqual(cell["n_scenarios"], 2)
        self.assertEqual(cell["n_attempts"], 6)
        self.assertAlmostEqual(cell["success_rate"], 1 / 6)  # avg@3
        self.assertEqual(cell["pass_at_k"], {"3": 0.5})
        self.assertEqual(cell["errors"][0]["attempt"], 3)

    def test_main_writes_summary_and_fails_on_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)

            def fake_popen(cmd, env, cwd):
                config = tomllib.loads(Path(cmd[-1]).read_text())
                out = Path(config["run"]["output_dir"])
                subset = config["target"]["splits"][0]
                write_result(out, subset, "s1", True)
                write_result(out, subset, "s2", None, error="Daemon error")
                self.assertEqual(env["GAIA2_ALLOWED_HOSTS"], "api.swissai.svc.cscs.ch")
                return mock.Mock(wait=mock.Mock(return_value=0), poll=mock.Mock(return_value=0))

            with mock.patch.dict("os.environ", {"OPENSANDBOX_API_KEY": "x", run.MODEL_KEY_ENV: "k"}), mock.patch.object(
                run.subprocess, "Popen", side_effect=fake_popen
            ) as popen, mock.patch.object(run, "delete_leftover_sandboxes") as cleanup, mock.patch.object(
                run, "materialize", return_value=work / "cache" / "deu_Latn"
            ):
                code = run.main([*MODEL_ARGS, "--work-dir", tmp, "--language", "deu_Latn", "--subset", "execution,search"])
            summary = json.loads((work / "summary.json").read_text())

        self.assertEqual(code, 1)
        self.assertEqual(popen.call_count, 2)
        self.assertEqual(popen.call_args.args[0][1:4], ["-m", "scripts.gaia2.opensandbox_launcher", "run-config"])
        cleanup.assert_called_once()
        self.assertEqual(summary["model"], "SwissAI-Research/swiss-ai/chonk")
        self.assertEqual(summary["judge_model"], "CSCS-Inference/zai-org/GLM-5.3")
        self.assertEqual(summary["overall"]["n_errors"], 2)
        self.assertEqual(summary["languages"]["deu_Latn"]["success_rate"], 0.5)
        self.assertIn("started_at", summary)
        self.assertEqual(summary["dataset_revision"], run.DATASET_REVISION)

    def test_main_needs_opensandbox_key(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(run.main(["--gold", "--work-dir", "w"]), 2)


if __name__ == "__main__":
    unittest.main()
