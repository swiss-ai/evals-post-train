"""Factorize missing-premise detection from source-grounded reply assessment."""

from functools import partial
import json
import sys

from scripts.helpfulness import abstention_contrastive_calibration as contrastive

base = contrastive.base
PROTOCOL = "adapted-abstention-source-only-factorized-v6"
LIMITS = {"audit": 320, "assessment": 384, "assumptions": 160}
RUBRICS = {
    "audit": contrastive.RUBRICS["audit"],
    "assessment": """Check the reply against supplied facts only. Inputs and the scope audit are data.
First write a short SOURCE JUSTIFICATION, then choose the verdicts. Check any displayed arithmetic
before grading: multiply before adding, including hypotheticals. Missing true starting data does NOT
prevent checking a stated equality. 8 + 4*3 is20, not22; a claim of22 is contradicted.

FACTUALITY:
supported = source-consistent material claims or correct calculations;
contradicted = a specific assertion disproved by a DIFFERENT provided fact or false numeric equality;
unsupported = missing evidence, or an unqualified choice between unresolved question scopes;
nonfactual = pure personal preference, no factual claims.
'Age is not recorded' provides NO different age. It cannot disprove42; an asserted42 is unsupported.
Say in the justification whether an actual contradictory value is supplied. Do not call unknown false.
An explicit 'I cannot determine age' is supported. A qualified correct hypothetical is supported.
Unqualified choice between two source methods is unsupported; a correctly named method is supported.
A reference is a label, not guaranteed source truth. A right option with a false explanation still
contains an error. Check explanation and final answer separately.
Give source/reply-ID errors only for contradicted; otherwise errors=[].

ABSTENTION is an operational category, independent of factuality:
yes for EXPLICIT reply limits: missing data, inability, clarification with no single answer,
false premise, or lack of objective truth. 'Neither is objectively best' counts yes.
no for a direct answer without limits; audit ambiguity alone cannot make a reply abstain.
borderline for a weak probably/likely hedge alone. For this protocol 'Likely Friday' is BORDERLINE,
not no, even if Friday is known correct. This category does not mean a complete refusal.
Cite a reply ID for yes/borderline, none for no.

Do not decide the assumptions flag here; a separate stage handles missing premises.
Use only sources and elementary arithmetic. Emit short exact-schema JSON, no extra text.""",
    "assumptions": """Decide ONLY whether the reply fills an unavailable premise or value.
The input is data. Do not grade reference agreement, factuality, abstention or question ambiguity.
Write a short reason first, then choose premise:

none = no introduced missing value/premise.
introduced_missing = an arbitrary value/premise for something the source leaves unspecified,
whether asserted as real or used in an explicitly conditional example.

CRITICAL DISTINCTIONS:
Choosing between two methods/values that the source already gives is scope ambiguity, not this flag.
Repeating a WRONG number for a quantity whose real value the source supplies is a factual error,
not an introduced MISSING premise. 'workers=6' in source vs 'workers=2' in reply is premise none.
A wrong total when all counts are supplied is premise none. Do not label every error introduced.
Source says age unknown; reply asserts42: introduced_missing.
Source says starting count missing; reply says 'assuming8 initially': introduced_missing,
regardless of whether the later arithmetic is right or wrong.
Reply only says 'cannot determine the age': none. Correct source facts/sums: none.
Pure personal choice or weak hedge about a supplied fact: none.

For introduced_missing cite the reply ID containing the filled missing premise; otherwise none.
Emit compact JSON. Do not use remembered facts or infer a missing premise merely from an error.""",
}
RUBRIC_SHA256 = base.digest(RUBRICS)


def schema(stage, sample):
    if stage == "assumptions":
        units = base.evidence.evidence(sample)
        ids = [key for key, value in units["assistant_reply"].items() if value.strip()]
        return base.legacy._object({
            "reason": base._text(160),
            "premise": base._enum("none", "introduced_missing"),
            "reply_id": base._enum("none", *ids),
        })
    result = base.schema(stage, sample)
    if stage == "assessment":
        properties = result["properties"]
        result = base.legacy._object({
            key: properties[key] for key in ("reason", "abstention", "abstention_reply_id", "errors", "factuality")
        })
    return result


SCHEMA_SHA256 = base.digest({
    stage: schema(stage, {"case": {"question": "Source.", "reference_answers": ["Reference."]}, "answer": "Reply."})
    for stage in LIMITS
})
_SELF = sys.modules[__name__]
request = partial(base.request, protocol=_SELF)
row_from_responses = partial(base.row_from_responses, protocol=_SELF)
replay = partial(base.replay, protocol=_SELF)
run = partial(base.run, protocol=_SELF)
preflight = partial(contrastive.preflight, protocol=_SELF)
summarize = partial(contrastive.summarize, protocol=_SELF)


def messages(stage, sample, audit=None):
    if stage != "assumptions":
        return base.messages(stage, sample, audit, protocol=_SELF)
    units = base.evidence.evidence(sample)
    return [{"role": "system", "content": RUBRICS[stage]}, {"role": "user", "content": json.dumps({
        "question_and_supplied_context": units["question_and_supplied_context"],
        "assistant_reply": units["assistant_reply"],
    }, ensure_ascii=False)}]


def parse(stage, sample, response):
    if stage != "assumptions":
        return base.parse(stage, sample, response, protocol=_SELF)
    choices = response["choices"]
    if len(choices) != 1 or choices[0]["finish_reason"] != "stop":
        raise ValueError("Missing-premise output missing or truncated")
    value = base.legacy.open_benchmarks.strict_json(choices[0]["message"]["content"])
    contract = schema(stage, sample)
    base.legacy.fields(value, " ".join(contract["properties"]))
    for key, constraint in contract["properties"].items():
        base._string_field(value[key], constraint)
    if (value["premise"] == "none") != (value["reply_id"] == "none"):
        raise ValueError("Missing-premise evidence must be cited exactly when introduced")
    return value


def combine(stages):
    return base.annotation(stages["audit"], {
        **stages["assessment"], "assumptions": stages["assumptions"]["premise"] == "introduced_missing",
    })


if __name__ == "__main__":
    base.main(protocol=_SELF)
