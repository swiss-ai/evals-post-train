from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_launcher_passes_api_and_judge_rate_limits():
    launcher = (REPO_ROOT / "scripts" / "launch_evaluations.sh").read_text()
    evaluate = (REPO_ROOT / "scripts" / "evaluate.sbatch").read_text()

    assert "--api-requests-per-minute" in launcher
    assert "--judge-requests-per-minute" in launcher
    assert "requests_per_minute=$API_REQUESTS_PER_MINUTE" in evaluate
    assert "export JUDGE_REQUESTS_PER_MINUTE" in evaluate
    assert "export JUDGE_MODEL_PREFIX" in evaluate


def test_orchestrator_assigns_shared_rate_limit_state():
    orchestrator = (REPO_ROOT / "scripts" / "evaluation_orchestrator.sh").read_text()

    assert 'LM_EVAL_RATE_LIMIT_STATE_DIR="$state_dir/rate_limits"' in orchestrator
