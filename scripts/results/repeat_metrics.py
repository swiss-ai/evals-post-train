"""Resolve repeat means without substituting an oracle pass@k score."""


def mean_key(summary, metric):
    task, sep, score = metric.partition("/")
    if not sep or "_mean_repeats" in score or "_pass@" in score:
        return None
    base, comma, filter_name = score.partition(",")
    key = f"{task}/{base}_mean_repeats"
    if comma:
        key += f",{filter_name}"
    if key in summary:
        return key
    matches = [k for k in summary if not comma and k.startswith(key + ",")]
    return sorted(matches, key=lambda k: ("strict-match" not in k, k))[0] if matches else None


def score_label(summary, metric):
    task = metric.split("/", 1)[0]
    protocol = summary.get("repeat_protocols", {}).get(task)
    if not protocol:
        return ""
    m = protocol["repeats"]
    key = (mean_key(summary, metric) or metric).split("/", 1)[-1]
    contributors = protocol.get("metric_tasks", {}).get(key)
    if contributors:
        counts = sorted({protocol["per_task_repeats"][task] for task in contributors})
        m = counts[0] if len(counts) == 1 else counts
    if "_pass@" in metric:
        k = metric.split("_pass@", 1)[1].split(",", 1)[0]
        return f"pass@{k} (m={m})"
    if mean_key(summary, metric) or "_mean_repeats" in metric:
        return f"mean@{m}"
    return ""
