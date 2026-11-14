# Batch helpfulness checkpoint evaluations

The `scripts.helpfulness` CLI bundles five existing Swiss AI harness adapters
and three new judge-based adapters. Optional IFBench two-turn evaluation is a
separate ninth adapter, not a replacement for the single-turn benchmark.
It is independent of evals-svc: no dashboard, database, model launching, or
W&B account is required.

The expert's active suite selects four families: AlpacaEval 2 LC, non-code
MT-Bench, biography sycophancy and SummEval. IFEval, IFBench (including two-turn),
TruthfulQA generation and SimpleQA were removed from that suite by request.
Their optional adapters remain available to other callers; the reference
documentation below does not schedule them for this teacher.

Here "offline" means a batch evaluation rather than interactive training-time
evaluation. Inference and independent judges can be hosted. Dataset acquisition
is a separate `prepare` step; `run` disables Hugging Face/NLTK downloads and W&B.
This is **not** an air-gapped distribution: native tasks also need the HF cache
populated during preparation, and hosted models/judges require network access.

## Runtime and data preparation

Use the existing evaluation environment/image described in
[lm-eval-harness.md](lm-eval-harness.md). It must contain the exact Swiss AI
harness git commit in `requirements/lm-eval-harness.txt`, not the similarly named
PyPI package. The CLI checks the installed distribution's `direct_url.json`.
Editable installs with no verifiable git commit are rejected.

The native runtime requirements and image already include the harness,
datasets, AlpacaEval and its GLM dependencies. A changed harness pin requires
preparing a new bundle. Keep the same HF cache mounted for prepare and run.
Do not install a second evaluator framework for these tasks.

Preparation stages checksummed NLTK resources for instruction verifiers and
AlpacaEval's released `df_gamed.csv` length-control features. Keep the complete
bundle, including `scorer-assets/`, intact. NLTK requires owner-private data
directories: prepare instruction benchmarks in a private Linux directory, not
a group-writable shared folder or a Windows-mounted directory with permissive
Unix modes. Preserve the downloaded resources' license notices. Evaluation
rejects missing/changed assets instead of acquiring them implicitly.

Teacher-specific suite configuration belongs in the teacher/specification
repository. The runnable example there is `evals/helpfulness.toml`; it describes
the two Clariden checkpoints but deliberately requires actual served model IDs
and a separate served judge model ID through environment variables.

```bash
# In an evals-post-train checkout, inside the existing evaluation environment:
python -m scripts.helpfulness.cli prepare \
  --suite /path/to/expert/evals/helpfulness.toml --bundle /work/data/helpfulness-v1

export TEACHER_QUARTER_MODEL='<actual fully scoped served model id>'
export TEACHER_FINAL_MODEL='<actual fully scoped served model id>'
export HELPFULNESS_JUDGE_MODEL='<actual fully scoped independent judge id>'
# Set TEACHER_API_KEY and JUDGE_API_KEY in your environment, never the manifest.
python -m scripts.helpfulness.cli run \
  --suite /path/to/expert/evals/helpfulness.toml --bundle /work/data/helpfulness-v1 \
  --output /work/results/helpfulness-v1
```

Use repeated `--benchmark NAME` or `--checkpoint NAME` to explicitly select a
subset. `--limit N` selects the first N examples **per native task**, after
category filtering; for sycophancy this is N examples in each of three topics.
Such limited runs are diagnostics, not full benchmark headline scores.
Prepare a subset explicitly when another benchmark's required data is unavailable.
`run` refuses benchmarks absent from the bundle rather than silently skipping them.

### Required local sources

Native preparation loads the pinned task's released dataset, preserves all rows
in a checksummed JSON bundle, and warms the existing HF cache. IFEval's HF split
is named `train` despite being evaluation data. IFBench uses
`allenai/IFBench_test`; its separately released RL training corpus is never
selected by this suite.

Three sources must be acquired separately before preparing the full suite.
Paths in the manifest are relative to the manifest, not the working directory.
Record the release revision and applicable data license in the manifest. The
prepared artifact additionally records a content hash.

| Adapter | Accepted source |
|---|---|
| `truthfulqa_generation` | Upstream `TruthfulQA.csv`, columns `Question`, `Correct Answers`, `Incorrect Answers`; normalized JSONL with `question`, `true_answers`, `false_answers` is also accepted. |
| `simpleqa` | Reference `simple_qa_test_set.csv`, columns `problem`, `answer`; JSONL with the same columns is also accepted. |
| `summeval` | Reconstructed, joined JSONL: nonempty `id`, `source_id`, `source`; `mode="validate"` additionally requires `summary` and `human_scores` with all four dimensions. |
| `ifbench_multiturn` (optional) | Released held-out two-turn data converted to JSONL with `key`, `turns` (exactly two user strings), the original verifier `prompt`, aligned `instruction_id_list` and `kwargs`. Do not synthesize a second turn from the single-turn set and call it the released protocol. |

SummEval's released summaries/annotations **do not contain source articles**.
Follow its upstream [reconstruction instructions](https://github.com/Yale-LILY/SummEval#data-preparation)
and verify article IDs before normalizing. A missing source is a hard preparation
error. Conflicting texts for the same source ID are rejected. In generation mode,
one article is one checkpoint prompt even if the release contains many summaries
of that article.

For validation, average the desired human annotator group's four scores yourself
and identify that group in `source_revision`. Example **schema only**, not data:

```json
{"id":"article-summary-id","source_id":"article-id","source":"reconstructed article text","summary":"released summary text","human_scores":{"consistency":4.5,"relevance":4.0,"coherence":4.5,"fluency":5.0}}
```

Use `mode="validate"` in a separate suite/bundle to evaluate the grounding grader
against released summaries, without asking the checkpoint to generate new ones.
Reports include Pearson and tied-rank Spearman correlations per dimension;
undefined correlations are `null`, not fabricated zeros. These are per-summary
diagnostics, not significance tests or article-cluster-adjusted statistics.

Code licenses are not blanket dataset licenses. Check the upstream release terms,
keep required notices with acquired data, and do not commit redistributed corpora
or article text. In particular, IFBench code is Apache-2.0 but its data is
ODC-BY-1.0 with third-party output terms; Anthropic's probes are CC-BY-4.0;
SummEval's MIT toolkit license does not grant rights to arbitrary news articles.
The CLI stages supplied assets; it does not resolve licensing or reconstruct
unavailable source text on your behalf.

## Protocols and metrics

| Adapter | Report and limitations |
|---|---|
| IFEval | Native strict/loose prompt compliance and instruction compliance. Instruction accuracy is weighted by the number of instructions, not averaged prompt percentages. This checks constraints, not factual usefulness. |
| IFBench | Same four native metrics; prompt-level loose accuracy is the paper's usual headline. Optional two-turn results have a separate adapter/protocol label and score only the second answer while retaining the first answer in its context. |
| AlpacaEval 2 LC | Reuses the pinned native generation/processing and annotator configuration, and the actual AlpacaEval length-control implementation and released features. Reports LC and raw win rates, word count, validated judgment coverage, and preserved annotations/GLM weights. The replacement judge makes this a Swiss AI adaptation, not the canonical GPT-4-Turbo judging protocol. No length penalty is substituted for the GLM. |
| MT-Bench | Native turn-one, turn-two, combined score and judge success rate. Category filters and replacement judges are labeled adapted. A judge success rate below 1 makes the run incomplete. Two turns are not a retention test. |
| TruthfulQA generation | Separate truthfulness, informativeness and their conjunction, using true/false references. Explicitly adapted replacement-judge rubric; not legacy GPT-judge/GPT-info and not a BLEU/ROUGE proxy. |
| SimpleQA | Correct, incorrect and not-attempted counts/rates among successfully judged cases. Invalid judge output is a **failed case**, never not-attempted. The rubric and judge differ from the full upstream prompt, so the protocol is explicitly adapted. |
| Sycophancy | Three native likelihood probes, renamed `biography_alignment_rate`; higher is not inherently better. A runner-local tokenizer and completion endpoint capable of likelihood scoring are required. Chat-only generation is not substituted. The harness prompt adaptation is labeled; these probes do not establish that philosophical agreement is wrong, nor supply a paired causal biography-shift metric. |
| SummEval | Independent source-grounded consistency, relevance, coherence and fluency judge scores; optional human-grader validation. Generated summaries are diagnostics, not the original released human evaluation. Overlap metrics are not treated as faithfulness. |

The SimpleQA and TruthfulQA rubrics in `adapters.py` are deliberately transparent,
versioned adaptations. They do not claim canonical benchmark scores. All judge
roles are separate from the target and require explicit served IDs; exact target/
judge ID reuse is refused. The user must also avoid aliasing the same underlying
checkpoint under different IDs.

MT-Bench and AlpacaEval's SDK clients require a nonempty judge API key through
the configured environment variable, even for a local server that ignores it.
AlpacaEval additionally requires chat log probabilities for both `m` and `M`;
a chat-only judge without this capability cannot run the weighted protocol.

Target and judge request rates are independently configurable. Native tasks use
the harness's limiter; custom tasks pace requests locally. Running multiple CLI
processes needs an endpoint-wide quota policy, not merely each process's limit.
No retries, logprob grading or reasoning-answer extraction changes are hidden as
official protocol parity.

## Checkpoint serving

This CLI never converts weights, requests a GPU allocation, or installs serving
code. Use the existing model-launch/serving workflow, then pass the registered
model ID. HF-shaped files alone do not establish compatibility with stock
vLLM/SGLang. The target server must render the verified native chat format and
support the model architecture. For sycophancy, it must also support completions,
prompt token log probabilities, and the runner's verified tokenizer.

Checkpoint path, revision, tokenizer revision and chat-template revision are
required provenance assertions, **not** filesystem verification by this HTTP
client. Preserve immutable checkpoint directories. Do not put a placeholder
revision or an unverified chat template in a scientific run.

## Artifacts, failures and resume

Native coverage checks flag empty answers, missing MT-Bench turns and reported
generation truncation as incomplete. These stricter run-validity checks do not
replace or alter the native benchmark metrics.

Each run stores `run.json`, `summary.json`, `summary.csv`, native harness results,
native MT-Bench judgments, validated AlpacaEval annotations, and custom per-case
JSONL events (answers, grader text, scores, error phase).

Failed or truncated custom responses remain failures. Rates use successfully
judged cases and always accompany scored/failed/unprocessed counts. A partial
run exits nonzero, never as a successful lower-scoring checkpoint.

`--resume` accepts only matching suite, selected checkpoints/benchmarks, endpoint
model IDs, data hashes, runner source hash, installed relevant package versions,
and harness pin. Completed native benchmarks are skipped; a failed native
benchmark is rerun in full. Custom generation is reused after a judge failure,
and successfully scored custom cases are skipped. This is not native per-request
resumption; use existing harness request-cache facilities for that separately.
Do not run two writers against the same output directory.

Run the focused offline contract/integration tests with:

```bash
python -m unittest discover -s tests -p test_helpfulness.py -v
```

These do not load the large checkpoint or charge an external judge. Real serving
and full-benchmark verification require the staged datasets, compatible endpoints,
tokens, and evaluation environment described above.

The dependency-free tests run without the harness. In the pinned native
environment, IFEval and MT-Bench additionally exercise the real evaluator against
local HTTP fixtures. To enable the real IFBench scorer and AlpacaEval
annotation/GLM tests, explicitly prepare their resources first and set
`HELPFULNESS_TEST_NLTK_DATA` and `HELPFULNESS_TEST_ALPACA_DATA` to the bundle's
`scorer-assets/nltk` and `scorer-assets/alpaca` directories, then run the same
command. Synthetic test credentials are supplied internally; tests never need
real teacher or judge credentials.

## Separate acceptance checks

For Slurm/EDF submission templates and cluster prerequisites, see
[Queue acceptance checks on Clariden](helpfulness-cluster.md).

`python -m scripts.helpfulness.acceptance` implements three independent gates.
They never replace checkpoint benchmark scores or certify an adapted protocol as
an official published evaluation:

| Kind | Required evidence | Acceptance |
| --- | --- | --- |
| `scorer_replay` | Frozen documents, answers and raw judgments/likelihoods; independently captured per-case decisions and aggregates | Every case decision and metric/denominator must reproduce. Boolean/label/count comparisons are exact; declared floating-point tolerance applies only to numeric precision. |
| `reference_regeneration` | Original reference evidence with immutable generation provenance, plus genuinely regenerated outputs | Weights, tokenizer, template, inference and decoding assertions must match. Simultaneous paired cluster-bootstrap difference intervals must lie entirely inside predefined equivalence margins. A similar aggregate or an interval merely containing zero is insufficient. |
| `judge_calibration` | Identical frozen answers and evidence, independently sourced original judgments or human annotations, and new Swiss AI judgments | Predefined agreement, signed-bias and absolute-error gates must pass, including every requested stratum. A passing result can then freeze the judge deployment, rubric, decoding, runtime and policy. |

The serving API normally exposes only a model name. Deployment verification here
is **operator-asserted, not verified server state**, as selected for this workflow.
Export actual serving metadata before the run, not an intended deployment.
All reports/locks retain that limitation and set
`published_protocol_equivalence = false`. Scientific interpretation and source
verification remain necessary even when all three gates pass.

### Frozen evidence and independent oracles

Keep licensed artifacts outside git. Normalize their representations without
changing prompts, output boundaries, case IDs, instruction order, split or
denominators. Preserve the upstream source/revision and the normalization recipe.
The oracle must come from the upstream scorer/judgments or independent human
annotations, **not** the implementation being tested.

Each line of `inputs.jsonl` contains:

```json
{
  "id": "original-case-id",
  "cluster_id": "original-question-or-source-article-id",
  "stratum": "predefined-category-or-bias-cohort",
  "doc": {"question": "original question", "gold": "reference answer"},
  "requests": [{"messages": [{"role": "user", "content": "original question"}]}],
  "answers": ["frozen response"],
  "judgments": ["{\"outcome\":\"CORRECT\"}"]
}
```

This illustrates the adapted SimpleQA representation, not a real benchmark
fixture. `doc` uses the existing adapter's normalized document schema. Keep
human ratings/oracle decisions out of `doc` and model/judge request payloads.
For historical judge-calibration fixtures without original generation requests,
`requests` may identify the frozen evidence instead; this is **not** sufficient
for faithful regeneration.

Use original question IDs as clusters, and source article IDs for SummEval:
multiple summaries, instructions or model answers from one source are not
independent samples. For MT-Bench, use category names as `stratum`; first and
second answers and raw grades are both required.

The separate `oracle.json` has this shape:

```json
{
  "schema_version": 1,
  "inputs_sha256": "<SHA256 of the exact input file bytes>",
  "source": "<independent upstream artifact URL or annotation source>",
  "revision": "<frozen source revision>",
  "protocol": "<reference protocol, not automatically our implemented protocol>",
  "label_origin": "original_judgments",
  "cases": [
    {
      "id": "original-case-id",
      "input_sha256": "<scripts.helpfulness.core.digest of this complete input row>",
      "scores": {"outcome": "CORRECT"}
    }
  ],
  "aggregate": {
    "metrics": {"correct": 1.0, "incorrect": 0.0, "not_attempted": 0.0},
    "denominators": {"correct": 1, "incorrect": 1, "not_attempted": 1}
  }
}
```

For calibration, `label_origin` must be `original_judgments` or
`human_annotations`. Regeneration additionally requires the original
`deployment` and `decoding` objects in the oracle. Missing historical pins or
per-example reference decisions are blockers, not fields to guess.

| Benchmark | Per-case `scores` and replay evidence |
| --- | --- |
| IFEval / IFBench / optional two-turn IFBench | All four `INSTRUCTION_METRICS`; prompt booleans and aligned instruction-boolean lists. Replay uses the actual pinned verifiers; two-turn replay scores the second response. |
| AlpacaEval 2 LC | `preference` in canonical reference-output-1/candidate-output-2 orientation, on the upstream 1–2 scale. Include canonical `annotation` with original global `index`, instruction, outputs and generator names. Raw logprob replay requires one `judgments` choice and `candidate_position` 1 or 2. For released parsed annotations, explicitly use `replay_stage = "annotation"` and `candidate_position = 2`; this does **not** exercise the original logprob parser. Upstream draw 0 normalizes to 1.5. |
| MT-Bench | `turn_1` and `turn_2` grades, each 1–10; two raw textual `judgments`, parsed by the pinned native parser. |
| TruthfulQA generation | Boolean `truthful`, `informative` and their conjunction; one raw adapted JSON judgment for parser replay. |
| SimpleQA | One `outcome`: `CORRECT`, `INCORRECT`, `NOT_ATTEMPTED`; one raw adapted JSON judgment for parser replay. |
| SummEval | All four dimension scores, 1–5; one raw adapted JSON judgment for parser replay. Human dimension means belong in the independent oracle. Identify expert versus crowd annotations; do not silently pool them. |
| Biography sycophancy | Boolean `biography_alignment`; `doc.choices`, matching `doc.target`, aligned total continuation `likelihoods`, and `answers = []`. Normalize the actual native choice order and topic-specific assistant prefix, not a chat-style replacement prompt. |

Legacy TruthfulQA judges are unavailable publicly; the original SimpleQA grader
returns a different representation from our JSON judge. Do not manufacture raw
original judgments or relabel a converted annotation as original parser replay.
Use genuinely available artifacts for the appropriate replay stage and compare
replacement judgments with independently normalized labels in the calibration
lane. A filtered MT-Bench fixture cannot certify the full 80-question headline.
Keep IFBench held-out data separate from RL training data.

### Predeclare, produce, evaluate, freeze

Create one TOML configuration/plan per kind, benchmark and scope. The expert
repository's `evals/acceptance.example.toml` documents required fields and contains
deliberately invalid placeholders, **not** approved tolerances or reference pins.
Protocol must equal the implemented `PROTOCOLS` entry, retaining adapted labels.

`deployment.json` uses this exact schema:

```json
{
  "schema_version": 1,
  "model_id": "<actual fully scoped served model ID>",
  "weights": {"source": "<model repository or immutable checkpoint manifest>", "revision": "<40-hex commit or 64-hex content hash>"},
  "tokenizer": {"source": "<tokenizer source>", "revision": "<40-hex commit or 64-hex content hash>"},
  "chat_template_sha256": "<SHA256 of actual template bytes>",
  "inference": {
    "backend": "<serving implementation>",
    "revision": "<40-hex commit or 64-hex image/content hash>",
    "dtype": "<actual precision>",
    "settings": {"context_length": 16384, "tensor_parallel_size": 1}
  }
}
```

Record every relevant inference setting/default (quantization, reasoning
handling, special tokens, etc.), not just the illustrated two settings. Decoder
settings and request templates must also capture the exact generation protocol;
do not treat hosted per-request seeds as equivalent to a global HF RNG seed.

```bash
python -m scripts.helpfulness.acceptance plan --config acceptance.toml --output plan.json
python -m scripts.helpfulness.acceptance run --plan plan.json --output replay-results
```

Plans seal inputs, oracle, policy, scorer assets and implementation/runtime before
candidate generation/judging. Changing a threshold requires a **new plan and
new candidate artifacts**, not editing a report until it passes. Instructions
require checked NLTK assets; LC replay/regeneration requires the staged
`df_gamed.csv` directory. Configure `[assets.nltk]`/`[assets.alpaca]` with
`directory` and `sha256 = digest(native.asset_hashes(directory))`.

For hosted production, `endpoint.toml` supplies `[endpoint]` with `base_url`,
`model`/`model_env`, `api_key_env`, `deployment_manifest` and optional pacing.
Regeneration of judged benchmarks also supplies `[judge]` with the independent
scoring endpoint and its exported deployment. These exports must match the plan.
Actual calls occur **only** when explicitly running these producer commands:

```bash
python -m scripts.helpfulness.acceptance regenerate --plan reference-plan.json --endpoint endpoint.toml --output regenerated.json
python -m scripts.helpfulness.acceptance judge --plan judge-plan.json --endpoint judge-endpoint.toml --output judged.json
python -m scripts.helpfulness.acceptance run --plan reference-plan.json --candidate regenerated.json --output regeneration-results
python -m scripts.helpfulness.acceptance run --plan judge-plan.json --candidate judged.json --output calibration-results
python -m scripts.helpfulness.acceptance freeze --report calibration-results/report.json --output judge-lock.json
```

The optional generation producer supports literal chat/text-completion requests,
with `{"role":"assistant","answer_from_turn":0}` or a text-prompt segment
`{"answer_from_turn":0}` for genuinely generated multi-turn history. It forwards
supported decoding fields and rejects unsupported options instead of dropping
them. It does not launch models or replace encoder-decoder, likelihood, or
benchmark-specific extraction code. Use the pinned original/native runner for
those cases and supply its output through the same artifact contract.

External candidates contain `schema_version = 1`, the
`acceptance_plan_sha256`, `inputs_sha256`, exact `deployment`/`decoding` objects,
and all case records. Judging adds `rubric_sha256`; judged regeneration adds the
plan's `judge` object. Calibration must preserve frozen answers and documents;
regeneration must preserve documents and symbolic request templates. Each record
must be complete, and reported truncation/failed judging cannot pass.

Hosted judging uses the actual custom rubrics, pinned MT prompts and weighted
Alpaca SDK. Alpaca SDK captures are labeled parsed annotations, not raw parser
replay; preserve original global indices when evaluating subsets. Use a separate
Alpaca calibration plan per candidate system, avoiding duplicate instructions.
Inspect `scoring.rubric(name, model)` and `scoring.judge_decoding(name, model)` to
record the actual rubric hash and semantic decoder settings.

### Uncertainty, bias and enforcing the freeze

Choose scientifically justified policies **before** collecting new outputs.
All benchmark metrics need criteria: do not omit informativeness, abstention,
instruction-level accuracy or turn two. Each predeclared stratum and the overall
group must meet the configured minimum cases/clusters and gates.

Regeneration uses paired prompt/article-cluster percentile bootstrap intervals;
Alpaca LC refits the actual upstream GLM in every draw, using the distinct
reference and regenerated response lengths. This can be computationally
expensive. Numerical failures invalidate uncertainty rather than being dropped.
Intervals receive Bonferroni correction for gated metrics/strata.

Calibration reports case-level agreement, signed bias, MAE, Pearson/Spearman,
categorical confusion matrices and Cohen's kappa where defined. Its conservative
agreement gate uses a Wilson interval for **clusters in which every verdict is
within the declared agreement tolerance**, not the raw case-agreement fraction.
This prevents a tiny all-agreeing fixture from reporting certainty. Bias/MAE use
paired cluster bootstrap intervals, with multiple-comparison correction.
Undefined correlations stay null; a required correlation gate then fails.
Degenerate bootstrap intervals are explicitly flagged and must be interpreted
in light of fixed-benchmark/deterministic versus stochastic inference.

Reports keep per-case differences, aggregates/denominators, coverage, uncertainty,
protocol/source distinctions and evidence hashes. Candidates are copied into the
report directory. Errors/incomplete coverage exit nonzero. Plans, reports,
candidate outputs and locks cannot be overwritten. Freeze rechecks source,
runtime and all artifact hashes and **recomputes calibration**, rather than
trusting a stored `passed` flag.

To require accepted judges in the existing checkpoint runner, configure:

```toml
[judge]
# Existing endpoint/model/credential fields remain unchanged.
require_calibration = true
deployment_manifest = "data/judge-deployment.json"

[judge.calibration_profiles]
simpleqa = "results/simpleqa-calibration/judge-lock.json"
# Add a separate lock for every selected judge-based benchmark.
```

With enforcement enabled, missing locks, changed deployments/models, rubrics,
decoding or scorer runtimes fail before model calls. MT-Bench additionally
requires accepted strata for every selected category. Lock hashes enter run
provenance and resume identity. This validates the exported deployment assertion;
it cannot detect an operator serving different weights behind unchanged metadata.
Existing runs remain possible without enforcement, but do not claim calibration.

Keep the plan, evidence, report and judge lock together. No real reference-model
acceptance, original-judge reproduction or teacher calibration is established by
the synthetic implementation tests.
