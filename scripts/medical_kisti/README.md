# Medical expert evals on KISTI HANGANG

The evals behind the medical RL expert's tables, for Apertus 2 (Chonk) HF exports on KISTI HANGANG (account
`in0138`, partition `intTest`). They are the scripts the medical team runs. They are not wired into this repo's
launchers.

| script | benchmarks | nodes | thinking (default) | sampling |
|---|---|---|---|---|
| `medbench.sbatch` | PubMedQA, MedMCQA, MedQA, MedXpertQA (`*_g`: step-by-step prompt, "Final Answer: X"), MMLU-Pro health (`mmlu_pro_zs_health`), IFEval | 1 | `high` | T 0.6, top-p 0.95, seed 42 |
| `healthbench.sbatch` | HealthBench Professional + HealthBench full (5,000), Gemma-4-31B-it grader | 2 | `low` | T 0.7 |
| `hallubench.sbatch` | OR-Bench hard/toxic, Health-ORSC, MedSafetyBench, AA-Omniscience Health, Pokemon probe; Gemma-4-31B-it judge | 2 | `medium` | hallubench's own (greedy) |

How it works:

- **Serving:** vLLM serves the export in the NeMo-RL Apertus 2 image (TP1, DP4, expert parallel). The export has no
  chat template, so `a2_gateway.py` sits in front of vLLM and encodes each chat in the Apertus 2 interaction format
  with apertus-common. The system frame is `{"thinking": <level>, "tools": []}`.
- **What the gateway returns:** the reply frame as `content`, the think frame as `reasoning_content` and the claim
  frame as `claim`. Every request is logged.
- **Thinking levels:** Chonk SFT v0.2 knows only `low` and `high`, and thinks only at `high`. The defaults are the
  levels the expert was trained at: multiple choice at `high`, open-ended at `low`.
- **Claim score:** `claim_score.py` scores the medbench samples a second time, on the claim frame.
- **AutoMOOVE:** not included. It uses MOOVE questions that are not public.

## Setup (once, login node)

```bash
git clone -b medical-expert-evals git@github.com:swiss-ai/evals-post-train.git ~/evals-post-train
bash ~/evals-post-train/scripts/medical_kisti/setup.sh          # --check lists what is missing
```

Everything goes to `/scratch/$USER/medical_evals/env` (set `MEDICAL_EVAL_ENV` to change it). Setup does the
following:

- **Code checkouts, at pinned commits:**
  - our lm-eval fork (public), which holds the `*_g` medbench tasks
  - `EPFLiGHT/healthbench` and `EPFLiGHT/hallubench`, which are private. Ask the medical team for read access, or
    set `HEALTHBENCH_SRC`/`HALLUBENCH_SRC` to a checkout you can read.
- **Gateway environment:** builds the gateway venv from `swiss-ai/apertus-common` at 494c2be.
- **Tokenizer:** copies it from the v0.2 SFT export (`TOKENIZER_FROM`).
- **Judge:** downloads Gemma-4-31B-it, or links an existing copy with `JUDGE_FROM=<dir>`.
- **Datasets:** fetches every dataset, because compute nodes have no internet.

## Run

Submit from a `/scratch` directory; logs land there as `med-<eval>-<job>.out`.

```bash
cd /scratch/$USER
M=<HF export dir>
sbatch --export=ALL,TAG=<tag>,MODEL=$M  ~/evals-post-train/scripts/medical_kisti/medbench.sbatch
sbatch --export=ALL,TAG=<tag>,MODEL=$M  ~/evals-post-train/scripts/medical_kisti/healthbench.sbatch
sbatch --export=ALL,NAME=<tag>,MODEL=$M ~/evals-post-train/scripts/medical_kisti/hallubench.sbatch
```

- **Overrides:** `THINKING_LEVEL=low|medium|high` changes the level. `SMOKE=1` runs a quick check (20 per task).
- **Task list:** for medbench, pass `TASKS=` in the environment, not inside `--export`, because sbatch splits
  `--export` on commas.
- **Run times:** about 1 to 2.5 h each.

## Results

All results are under `/scratch/$USER/medical_evals/`:

| eval | where | headline metric |
|---|---|---|
| medbench | `eval_results/medbench/<tag>_<job>/…/results_*.json`, `claim_score.json` | `exact_match,strict-final-answer` (`*_g`), `exact_match,custom-extract` (MMLU-Pro), IFEval `prompt_level_strict_acc` |
| HealthBench | `env/src/healthbench/results/{professional,full}_gemma4_31b/results_<tag>_<subset>.json` | overall score |
| hallubench | `eval_results/hallubench/<tag>/summary.csv` | per task |

Each job's `runs/<job>/` folder, or `eval_results/hallubench/servers/` for hallubench, holds the gateway log. It has
the think, claim and reply of every request.
