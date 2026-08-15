import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class BenchmarkConfigTests(unittest.TestCase):
    def test_posttrain_includes_alignment_benchmarks(self) -> None:
        tasks = (
            REPO_ROOT / "configs" / "apertus" / "tasks_posttrain_final.txt"
        ).read_text(encoding="utf-8")

        self.assertIn("\nbfcl_v3\n", tasks)
        self.assertIn("\nswiss_ai_charter_alignment\n", tasks)

    def test_minerva_tables_promote_math_verify(self) -> None:
        table_files = list((REPO_ROOT / "configs").glob("**/*main_table.txt"))
        offending = []
        for table_file in table_files:
            for line_number, line in enumerate(
                table_file.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if line.startswith(("minerva_math/", "minerva_math500/")) and not line.endswith(
                    "/math_verify"
                ):
                    offending.append(f"{table_file}:{line_number}:{line}")
                if line.startswith("minerva_math500_self_consistency/") and not line.startswith(
                    "minerva_math500_self_consistency/math_verify,"
                ):
                    offending.append(f"{table_file}:{line_number}:{line}")

        self.assertEqual(offending, [])

    def test_gpt_suite_has_matching_metrics(self) -> None:
        tasks = [
            line
            for line in (
                REPO_ROOT / "configs" / "apertus" / "tasks_gpt.txt"
            ).read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        ]
        metrics = [
            line.split("/", 1)[0]
            for line in (
                REPO_ROOT / "configs" / "apertus" / "tasks_gpt_main_table.txt"
            ).read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        ]

        self.assertEqual(tasks, metrics)

    def test_only_swiss_ai_and_ymetz_harness_forks_are_referenced(self) -> None:
        scripts = "\n".join(
            (REPO_ROOT / relative).read_text(encoding="utf-8")
            for relative in (
                "scripts/evaluation_orchestrator.sh",
                "scripts/evaluate.sbatch",
            )
        )

        repos = set(re.findall(r'"([\w.-]+/lm-evaluation-harness)"', scripts))
        self.assertEqual(
            repos,
            {
                "swiss-ai/lm-evaluation-harness",
                "ymetz/lm-evaluation-harness",
            },
        )


if __name__ == "__main__":
    unittest.main()
