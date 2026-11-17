"""Score medbench's MCQ tasks on the claim frame of an Apertus 2 model, next to lm-eval's reply-based score.

lm-eval grades the reply text with its regex filters (a2_gateway.py returns only the reply). Apertus 2 SFT
models also commit to an answer in the claim frame, the channel RL rewards read; the gateway logs it per request.
This joins lm-eval's logged samples (medbench.sbatch, --log_samples) to the gateway log on the last user message and
reports, per task: n, lm-eval's strict score, claim present, claim accuracy, empty replies and truncated outputs.

usage: python claim_score.py <lm-eval output dir> <gateway.jsonl> [--out summary.json]
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

LETTER = re.compile(r"^\W*([A-J])(?![A-Za-z0-9])")


def last_user(messages):
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content")
            return c if isinstance(c, str) else " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return None


def claim_answer(claim, target):
    """The claim's answer in the target's form: a letter for letter targets, else the lowercased first word."""
    if claim is None:
        return None
    if re.fullmatch(r"[A-J]", target.strip()):
        m = LETTER.match(claim.strip())
        return m.group(1) if m else None
    w = re.findall(r"[a-z]+", claim.lower())
    return w[0] if w else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("lm_eval_dir")
    ap.add_argument("gateway_log")
    ap.add_argument("--out")
    args = ap.parse_args()
    log = {}
    for line in open(args.gateway_log):
        r = json.loads(line)
        log[last_user(r["messages"])] = r
    rows = defaultdict(lambda: defaultdict(int))
    for f in sorted(glob.glob(os.path.join(args.lm_eval_dir, "**", "samples_*.jsonl"), recursive=True)):
        task = re.sub(r"^samples_(.*)_\d{4}-\d\d-\d\dT.*$", r"\1", os.path.basename(f))
        for line in open(f):
            s = json.loads(line)
            if s.get("filter") not in (None, "strict-final-answer", "strict-match", "custom-extract", "none"):
                continue
            target = str(s["target"]).strip()
            try:
                a0 = s["arguments"]["gen_args_0"]["arg_0"]   # API models: [<JSON messages string>]
                msgs = json.loads(a0[0] if isinstance(a0, list) else a0)
            except (json.JSONDecodeError, TypeError, KeyError):
                msgs = None
            g = log.get(last_user(msgs)) if isinstance(msgs, list) else None
            t = rows[task]
            t["n"] += 1
            t["lm_eval"] += float(s.get("exact_match", s.get("acc", 0)) or 0)
            if g is None:
                t["unmatched"] += 1
                continue
            a = claim_answer(g["claim"], target)
            t["claim_present"] += g["claim"] is not None
            t["claim_correct"] += a is not None and a.lower() == target.lower()
            t["empty_reply"] += g["reply"] is None
            t["truncated"] += g["vllm_finish"] == "length"
            t["think_nonempty"] += bool(g["think"])
            t["gen_tokens"] += g["gen_tokens"]
    out = {}
    print(f"{'task':<22}{'n':>6}{'lm-eval':>9}{'claim%':>8}{'claim acc':>11}{'no reply':>10}{'trunc':>7}{'think':>7}{'tok':>7}")
    for task, t in sorted(rows.items()):
        n = t["n"]
        out[task] = {"n": n, "lm_eval_strict": t["lm_eval"] / n, "claim_present": t["claim_present"] / n,
                     "claim_acc": t["claim_correct"] / n, "empty_reply": t["empty_reply"] / n,
                     "truncated": t["truncated"] / n, "think_nonempty": t["think_nonempty"] / n,
                     "mean_gen_tokens": t["gen_tokens"] / max(1, n - t["unmatched"]), "unmatched": t["unmatched"]}
        o = out[task]
        print(f"{task:<22}{n:>6}{o['lm_eval_strict']:>9.3f}{o['claim_present']:>8.2f}{o['claim_acc']:>11.3f}"
              f"{o['empty_reply']:>10.3f}{o['truncated']:>7.3f}{o['think_nonempty']:>7.2f}{o['mean_gen_tokens']:>7.0f}"
              + (f"  unmatched {t['unmatched']}" if t["unmatched"] else ""))
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
