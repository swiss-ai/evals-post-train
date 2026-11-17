"""MMLU-Pro zero-shot for chat models: lm-eval's mmlu_pro prompt with the subject instruction in the user turn.

lm-eval's mmlu_pro puts the instruction in the task description (a system message under a chat template) and sends
the 5 worked examples as separate user turns. On the Apertus 2 Chonk SFT (rl/chonk/a2_gateway.py) the system message
becomes the system frame's untrained "behavior" field and the model reasons in only ~10% of answers (.25 vs .55 with
reasoning; run 3850664). Here: no few-shot, no description; the instruction opens the user message. Same questions,
options, answer regex and filter as mmlu_pro, so per-subject scores compare.
"""
from functools import partial


CHOICES = "ABCDEFGHIJ"


def process_docs(dataset, subject):
    return dataset.filter(lambda x: x["category"] == subject)


def doc_to_text(doc, subject):
    lines = [f"The following is a multiple choice question about {subject}. Think step by step and then finish your "
             'answer with "the answer is (X)" where X is the correct letter choice.', "", "Question:", doc["question"],
             "Options:"]
    lines += [f"{CHOICES[i]}. {opt.strip()}" for i, opt in enumerate(doc["options"][:len(CHOICES)])]
    return "\n".join(lines)


SUBJECTS = ["biology", "business", "chemistry", "computer science", "economics", "engineering", "health", "history",
            "law", "math", "other", "philosophy", "physics", "psychology"]
for _s in SUBJECTS:
    _k = _s.replace(" ", "_")
    globals()[f"process_{_k}"] = partial(process_docs, subject=_s)
    globals()[f"doc_to_text_{_k}"] = partial(doc_to_text, subject=_s)
