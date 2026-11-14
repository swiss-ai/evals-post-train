"""Replay raw benchmark evidence without making generation or judge calls."""

import importlib
from copy import deepcopy

from scripts.helpfulness import adapters, native
from scripts.helpfulness.core import mean, number


class RecordedJudge:
    def __init__(self, response: str):
        self.response = response

    def judge(self, instruction, evidence):
        return self.response


def validate_document(name: str, doc: dict) -> None:
    if name in adapters.open_benchmarks.NAMES | adapters.EXTERNAL_TASKS:
        raise ValueError(
            "New teacher adapters are uncalibrated diagnostics; independent acceptance "
            "replay/producer contracts are not implemented for this protocol"
        )
    fields = {
        "truthfulqa_generation": ("question", "true_answers", "false_answers"),
        "simpleqa": ("question", "gold"),
        "summeval": ("source", "source_id"),
        "sycophancy": ("question",),
    }.get(name, ())
    for field in fields:
        adapters.text(doc, field)
    if name in ("ifeval", "ifbench", "ifbench_multiturn"):
        adapters.text(doc, "prompt")
        ids, kwargs = doc.get("instruction_id_list"), doc.get("kwargs")
        if (
            "key" not in doc or not isinstance(ids, list) or not ids
            or any(not isinstance(item, str) or not item for item in ids)
            or not isinstance(kwargs, list) or len(kwargs) != len(ids)
            or any(not isinstance(item, dict) for item in kwargs)
        ):
            raise ValueError("Instruction documents require original key and aligned verifier IDs/kwargs")
    if name == "mt_bench":
        turns = doc.get("turns")
        if not isinstance(turns, list) or len(turns) != 2 or any(
            not isinstance(turn, str) or not turn.strip() for turn in turns
        ):
            raise ValueError("MT-Bench requires both original user turns")
        adapters.text(doc, "category")
        if "question_id" not in doc:
            raise ValueError("Preserve the original MT-Bench question ID for clustering")
        native.require_harness()
        metric = importlib.import_module("lm_eval.tasks.mt_bench.metric")
        if doc["category"] in metric.NEED_REF_CATS:
            references = doc.get("reference")
            if not isinstance(references, list) or len(references) != 2 or any(
                not isinstance(value, str) or not value.strip() for value in references
            ):
                raise ValueError("Reference-based MT-Bench judging requires both reference answers")


def validate_scores(name: str, scores: dict, doc: dict) -> dict:
    if not isinstance(scores, dict):
        raise ValueError("Per-example scores must be an object")
    if name in ("ifeval", "ifbench", "ifbench_multiturn"):
        if set(scores) != set(adapters.INSTRUCTION_METRICS):
            raise ValueError("Instruction scores require all four strict/loose metrics")
        count = len(doc["instruction_id_list"])
        if not count:
            raise ValueError("Instruction examples require at least one instruction")
        for key, value in scores.items():
            if key.startswith("inst_"):
                if not isinstance(value, list) or len(value) != count or any(
                    type(item) is not bool for item in value
                ):
                    raise ValueError("Instruction verdicts must align with verifier IDs")
            elif type(value) is not bool:
                raise ValueError("Prompt verdicts must be booleans")
    elif name == "truthfulqa_generation":
        if set(scores) != {"truthful", "informative", "truthful_and_informative"} or any(
            type(value) is not bool for value in scores.values()
        ):
            raise ValueError("TruthfulQA scores require three boolean fields")
        if scores["truthful_and_informative"] != (scores["truthful"] and scores["informative"]):
            raise ValueError("TruthfulQA conjunction does not match its component judgments")
    elif name == "simpleqa":
        if set(scores) != {"outcome"} or scores["outcome"] not in (
            "CORRECT", "INCORRECT", "NOT_ATTEMPTED"
        ):
            raise ValueError("SimpleQA requires one of its three outcomes")
    elif name == "summeval":
        if set(scores) != set(adapters.DIMENSIONS):
            raise ValueError("SummEval requires all four dimensions")
        scores = {key: number(value, key, 1, 5) for key, value in scores.items()}
    elif name == "mt_bench":
        if set(scores) != {"turn_1", "turn_2"}:
            raise ValueError("MT-Bench requires separately scored first and second turns")
        scores = {key: number(value, key, 1, 10) for key, value in scores.items()}
    elif name == "alpaca_eval_2_lc":
        if set(scores) != {"preference"}:
            raise ValueError("AlpacaEval requires the canonical output-2 preference")
        scores = {"preference": number(scores["preference"], "preference", 1, 2)}
    elif name == "sycophancy":
        if set(scores) != {"biography_alignment"} or type(scores["biography_alignment"]) is not bool:
            raise ValueError("Sycophancy alignment must be a boolean")
    else:
        adapters.validate_benchmark(name)
    return scores


def replay_case(name: str, row: dict) -> dict:
    doc, answers = row["doc"], row["answers"]
    judgments = row.get("judgments", [])
    if name in ("ifeval", "ifbench", "ifbench_multiturn"):
        native.require_harness()
        family = "ifeval" if name == "ifeval" else "ifbench"
        utils = importlib.import_module(f"lm_eval.tasks.{family}.utils")
        result = utils.process_results(doc, [answers[-1]])
        scores = {key: result[key] for key in adapters.INSTRUCTION_METRICS}
    elif name in ("truthfulqa_generation", "simpleqa", "summeval"):
        if not isinstance(judgments, list) or len(judgments) != 1 or not isinstance(judgments[0], str):
            raise ValueError("JSON-judge replay requires exactly one raw judgment")
        scores, _ = adapters.score_case(name, doc, answers, RecordedJudge(judgments[0]))
    elif name == "mt_bench":
        native.require_harness()
        metric = importlib.import_module("lm_eval.tasks.mt_bench.metric")
        if not isinstance(judgments, list) or len(judgments) != 2 or any(
            not isinstance(item, str) for item in judgments
        ):
            raise ValueError("MT-Bench replay requires two raw judge responses")
        scores = {f"turn_{turn}": metric._parse_score(raw) for turn, raw in enumerate(judgments, 1)}
    elif name == "alpaca_eval_2_lc":
        position = row.get("candidate_position")
        if type(position) is not int or position not in (1, 2):
            raise ValueError("AlpacaEval must declare the candidate's judge-prompt position")
        if row.get("replay_stage") == "annotation":
            annotation = row["annotation"]
            value = annotation["preference"]
            preference = 1.5 if type(value) in (int, float) and value == 0 else number(
                value, "annotated preference", 1, 2
            )
        else:
            from alpaca_eval.completion_parsers import logprob_parser

            if not isinstance(judgments, list) or len(judgments) != 1:
                raise ValueError("AlpacaEval parser replay requires one raw logprob choice")
            choice = judgments[0]
            content = choice.get("logprobs", {}).get("content") if isinstance(choice, dict) else None
            if (
                not isinstance(content, list) or len(content) != 1
                or not isinstance(content[0], dict) or not content[0].get("top_logprobs")
            ):
                raise ValueError("Weighted AlpacaEval needs a single-token top-logprob trace")
            for token in content[0]["top_logprobs"]:
                if not isinstance(token, dict) or not isinstance(token.get("token"), str):
                    raise ValueError("Malformed AlpacaEval token trace")
                number(token.get("logprob"), "token log probability", -1e100, 0)
            preference = number(
                float(logprob_parser(
                    judgments[0], numerator_token="m", denominator_tokens=["m", "M"],
                    is_binarize=False, log_prob_index=0,
                )[0]), "parsed preference", 1, 2,
            )
        scores = {"preference": (
            preference if row.get("replay_stage") == "annotation" or position == 2
            else 3 - preference
        )}
    else:
        choices, target = doc.get("choices"), doc.get("target")
        likelihoods = row.get("likelihoods")
        if (
            not isinstance(choices, list) or len(choices) < 2
            or any(not isinstance(item, str) or not item for item in choices)
            or type(target) is not int or not 0 <= target < len(choices)
            or not isinstance(likelihoods, list) or len(likelihoods) != len(choices)
        ):
            raise ValueError("Sycophancy requires ordered choices, a matching target, and aligned likelihoods")
        values = [number(value, "continuation log likelihood", -1e100, 0) for value in likelihoods]
        scores = {"biography_alignment": max(range(len(values)), key=values.__getitem__) == target}
    return validate_scores(name, scores, doc)


def vectors(name: str, scores: dict) -> dict[str, list[float]]:
    if name == "simpleqa":
        return {
            key.lower(): [float(scores["outcome"] == key)]
            for key in ("CORRECT", "INCORRECT", "NOT_ATTEMPTED")
        }
    if name == "alpaca_eval_2_lc":
        return {"raw_win_rate": [scores["preference"] - 1]}
    result = {
        key: [float(item) for item in value] if isinstance(value, list) else [float(value)]
        for key, value in scores.items()
    }
    if name == "mt_bench":
        result["score"] = result["turn_1"] + result["turn_2"]
    return result


def aggregate(name: str, rows: list[dict], scores: list[dict], *, length_control: bool = True) -> dict:
    if not rows or len(rows) != len(scores):
        raise ValueError("Aggregation requires every selected example")
    collected = {}
    for score in scores:
        for key, values in vectors(name, score).items():
            collected.setdefault(key, []).extend(values)
    metrics = {key: mean(values) for key, values in collected.items()}
    denominators = {key: len(values) for key, values in collected.items()}
    if name == "mt_bench":
        metrics["score"] = (metrics["turn_1"] + metrics["turn_2"]) / 2
        denominators["score"] = 2 * len(rows)
    if name == "alpaca_eval_2_lc" and length_control:
        from alpaca_eval.metrics import get_length_controlled_winrate

        annotations = []
        for row, score in zip(rows, scores):
            annotation = deepcopy(row["annotation"])
            if (
                annotation["output_2"] != row["answers"][0]
                or annotation["output_1"] != row["doc"]["reference_output"]
                or annotation["instruction"] != row["doc"]["instruction"]
                or type(annotation.get("index")) is not int
            ):
                raise ValueError("AlpacaEval annotation does not match the frozen pair/index")
            annotation["preference"] = score["preference"]
            annotations.append(annotation)
        result = get_length_controlled_winrate(annotations, save_weights_dir=None)
        metrics["length_controlled_winrate"] = number(
            float(result["length_controlled_winrate"]), "AlpacaEval LC win rate", 0, 100
        ) / 100
        denominators["length_controlled_winrate"] = len(rows)
    return {"metrics": metrics, "denominators": denominators}


def rubric(name: str, model: str) -> dict:
    if name in ("truthfulqa_generation", "simpleqa", "summeval"):
        return {
            "instruction": {
                "truthfulqa_generation": adapters.TRUTH_INSTRUCTION,
                "simpleqa": adapters.SIMPLE_INSTRUCTION,
                "summeval": adapters.SUMMARY_INSTRUCTION,
            }[name],
            "transport": "scripts.helpfulness.core.ChatClient.judge",
        }
    native.require_harness()
    if name == "mt_bench":
        metric = importlib.import_module("lm_eval.tasks.mt_bench.metric")
        return {"prompts": metric._load_judge_prompts(), "harness_revision": native.harness_pin()}
    if name == "alpaca_eval_2_lc":
        from alpaca_eval import constants
        from lm_eval.tasks.alpaca_eval.metric import _annotator_config

        config = _annotator_config(model)
        return {
            "config": config,
            "templates": {
                key: (constants.EVALUATORS_CONFIG_DIR / value["prompt_template"]).read_text(encoding="utf-8")
                for key, value in config.items()
            },
            "harness_revision": native.harness_pin(),
        }
    raise ValueError(f"{name} does not use a replacement judge")


def judge_decoding(name: str, model: str) -> dict:
    if name == "alpaca_eval_2_lc":
        native.require_harness()
        from lm_eval.tasks.alpaca_eval.metric import _annotator_config

        config = next(iter(_annotator_config(model).values()))["completions_kwargs"]
        return {
            key: value for key, value in config.items()
            if key in ("temperature", "max_tokens", "logprobs", "top_logprobs")
        }
    if name == "mt_bench":
        import os

        return {"temperature": 0, "max_tokens": int(os.environ.get("MT_BENCH_JUDGE_MAX_TOKENS", "2048"))}
    if name in ("truthfulqa_generation", "simpleqa", "summeval"):
        return {"temperature": 0, "max_tokens": 2048}
    raise ValueError(f"{name} does not use a replacement judge")


def confusion(name: str, reference: list[dict], candidate: list[dict]) -> dict:
    if name == "simpleqa":
        labels = ("CORRECT", "INCORRECT", "NOT_ATTEMPTED")
        pairs = [(left["outcome"], right["outcome"]) for left, right in zip(reference, candidate)]
    elif name == "alpaca_eval_2_lc":
        labels = ("reference_wins", "tie", "candidate_wins")

        def label(score):
            value = score["preference"]
            return labels[0] if value < 1.5 else labels[2] if value > 1.5 else labels[1]

        pairs = [(label(left), label(right)) for left, right in zip(reference, candidate)]
    else:
        return {}
    matrix = {left: dict.fromkeys(labels, 0) for left in labels}
    for left, right in pairs:
        matrix[left][right] += 1
    count = len(pairs)
    agreement = sum(matrix[label][label] for label in labels) / count
    chance = sum(
        sum(matrix[label].values()) * sum(matrix[row][label] for row in labels)
        for label in labels
    ) / count**2
    return {
        "reference_rows_candidate_columns": matrix,
        "agreement": agreement,
        "cohen_kappa": (agreement - chance) / (1 - chance) if chance < 1 else None,
    }
