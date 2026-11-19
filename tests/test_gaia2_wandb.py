import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.gaia2 import run
from scripts.results import update_wandb_gaia2
from scripts.results.gaia2_wandb_utils import create_model_evaluation_from_gaia2_run


def write_result(work: Path, language: str, subset: str, scenario: str, success, **extra):
    path = work / "runs" / language / subset / "results" / subset / scenario / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"scenario_id": scenario, "success": success, **extra}))


class Gaia2WandbTests(unittest.TestCase):
    def make_run(self, work: Path):
        write_result(work, "deu_Latn", "execution", "a", True, agent_response="Erledigt")
        write_result(work, "deu_Latn", "execution", "b", False, failure_reasons=["wrong recipient"])
        write_result(work, "deu_Latn", "execution", "c", None, error="Adapter unreachable")
        write_result(work, "fra_Latn", "execution", "a", True)
        summary = run.summarize(work, ["deu_Latn", "fra_Latn"], ["execution"])
        summary.update(started_at="2026-11-19T10:00:00+00:00", finished_at="2026-11-19T10:30:00+00:00")
        (work / "summary.json").write_text(json.dumps(summary))

    def test_model_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            self.make_run(work)
            evaluation, duration = create_model_evaluation_from_gaia2_run("chonk", work)

        self.assertEqual(duration, 30 * 60)
        flat = evaluation.get_flattened_metrics()
        self.assertEqual(flat["omnilingual_gaia2/success_rate"], 0.5)
        self.assertEqual(flat["omnilingual_gaia2/n_errors"], 1)
        self.assertAlmostEqual(flat["omnilingual_gaia2_deu_Latn/success_rate"], 1 / 3)
        self.assertEqual(flat["omnilingual_gaia2_fra_Latn/success_rate"], 1.0)
        self.assertEqual(flat["omnilingual_gaia2_execution/n_scenarios"], 4)
        self.assertEqual(flat["omnilingual_gaia2_deu_Latn_execution/n_attempts"], 3)
        cell = next(t for t in evaluation.tasks if t.task_name == "omnilingual_gaia2_deu_Latn_execution")
        samples = [s.sample_data for s in cell.samples]
        self.assertEqual(len(samples), 3)
        self.assertEqual({s["scenario"]: s["is_correct"] for s in samples}, {"a": True, "b": False, "c": None})
        self.assertEqual(next(s for s in samples if s["scenario"] == "b")["failure_reasons"], ["wrong recipient"])

    def test_main_table_is_overall_and_languages(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            self.make_run(work)
            with mock.patch.object(update_wandb_gaia2, "upload_multi_model_results") as upload:
                update_wandb_gaia2.main("e", "p", "chonk", work)
        main_metrics = upload.call_args.args[3]
        self.assertEqual(
            main_metrics,
            [
                "omnilingual_gaia2/success_rate",
                "omnilingual_gaia2_deu_Latn/success_rate",
                "omnilingual_gaia2_fra_Latn/success_rate",
            ],
        )


if __name__ == "__main__":
    unittest.main()
