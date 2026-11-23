"""Points tau2-bench's LLM judges at another model, in memory.

Upstream hardcodes both judges to an OpenAI model with no option to change them: the natural-language-assertion
judge (tau2/config.py: DEFAULT_LLM_NL_ASSERTIONS) and the semantic judge of a task's required information
(`communicate_info`, which six airline tasks carry), whose model (DEFAULT_SEMANTIC_COMMUNICATE_MODEL) is
resolved into each task's CommunicateScoring.judge. run_tau2_bench.sh puts this directory on PYTHONPATH of the `tau2` process;
Python imports a `sitecustomize` module at startup. With TAU2_JUDGE_LLM unset nothing happens. With it set (a
litellm model string) and TAU2_JUDGE_LLM_ARGS (a json object of litellm kwargs; the key as
"os.environ/<NAME>", never the key itself), the NL-assertion evaluator's two module-level names are replaced
and the semantic judge's calls are sent to that model, before tau2 runs. Upstream's files are not touched,
and tau2 runs its simulations in threads, so the replacement is seen by every evaluation.

A model judging is less well-behaved than the OpenAI model upstream was written for, in two ways handled here:
  - It may wrap its JSON in a markdown code fence (Chonk does, every time), which json.loads rejects: the fence
    is removed from the reply.
  - Its reply may not parse, or may hold fewer verdicts than there are assertions (Chonk returns an
    empty list for about half of its calls, which upstream counts as "all assertions met": the task would pass without being judged). Either
    way the judge is asked again, up to _ATTEMPTS times; after that the evaluation raises, so the task is redone
    by tau2 and never scored on a missing judgement.
"""
import json
import os
import re

_FENCED = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_ATTEMPTS = 8

if os.environ.get("TAU2_JUDGE_LLM"):
    from tau2.evaluator import evaluator_communicate, evaluator_nl_assertions

    evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS = os.environ["TAU2_JUDGE_LLM"]
    evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS = json.loads(os.environ["TAU2_JUDGE_LLM_ARGS"])

    def _unfenced(generate):
        def generate_unfenced(*args, **kwargs):
            message = generate(*args, **kwargs)
            if isinstance(message.content, str):
                fenced = _FENCED.fullmatch(message.content.strip())
                if fenced:
                    message.content = fenced.group(1)
            return message

        return generate_unfenced

    evaluator_nl_assertions.generate = _unfenced(evaluator_nl_assertions.generate)

    _communicate_generate = _unfenced(evaluator_communicate.generate)

    def _generate_communicate(*, messages, call_name, **_pinned):
        # The model and arguments of the task's CommunicateScoring.judge (`_pinned`, with its `model`) are dropped.
        return _communicate_generate(
            model=os.environ["TAU2_JUDGE_LLM"],
            messages=messages,
            call_name=call_name,
            **json.loads(os.environ["TAU2_JUDGE_LLM_ARGS"]),
        )

    evaluator_communicate.generate = _generate_communicate

    _evaluate = evaluator_nl_assertions.NLAssertionsEvaluator.evaluate_nl_assertions

    def _evaluate_checked(trajectory, nl_assertions):
        for attempt in range(1, _ATTEMPTS + 1):
            try:
                checks = _evaluate(trajectory, nl_assertions)
                if len(checks) == len(nl_assertions):
                    return checks
                problem = f"{len(checks)} verdicts for {len(nl_assertions)} assertions"
            except (TypeError, ValueError, KeyError) as error:
                problem = f"{type(error).__name__}: {error}"
                last_error = error
            print(f"NL-assertion judge reply unusable (attempt {attempt}/{_ATTEMPTS}): {problem}", flush=True)
        raise ValueError(f"NL-assertion judge gave no usable reply in {_ATTEMPTS} attempts ({problem})")

    evaluator_nl_assertions.NLAssertionsEvaluator.evaluate_nl_assertions = staticmethod(_evaluate_checked)
