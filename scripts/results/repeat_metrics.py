"""Resolve repeat means without substituting an oracle pass@k score."""


def first_key(task, keys):
    """The table's pick among matching keys: gsm8k prefers strict-match, then the shortest."""
    keys = sorted(keys, key=lambda k: ("gsm8k" in task.lower() and "strict-match" not in k, len(k)))
    return keys[0] if keys else None


def mean_key(summary, metric):
    task, sep, score = metric.partition("/")
    if (not sep or "_mean_repeats" in score or "_pass@" in score
            or task not in summary.get("repeat_protocols", {})):
        return None
    base, comma, filter_name = score.partition(",")
    key = f"{task}/{base}_mean_repeats"
    if comma:
        key += f",{filter_name}"
    if key in summary:
        return key
    if comma:
        return None
    return first_key(task, [k for k in summary if k.startswith(key + ",")])


def score_label(summary, metric):
    task = metric.split("/", 1)[0]
    protocol = summary.get("repeat_protocols", {}).get(task)
    if not protocol:
        return ""
    m = protocol["repeats"]
    repeated = mean_key(summary, metric)
    key = (repeated or metric).split("/", 1)[-1]
    metric_tasks = protocol.get("metric_tasks", {})
    if "," not in key:
        key = first_key(task, [k for k in metric_tasks if k.startswith(key + ",")]) or key
    contributors = metric_tasks.get(key)
    if contributors:
        counts = sorted({protocol["per_task_repeats"][task] for task in contributors})
        m = counts[0] if len(counts) == 1 else counts
    if "_pass@" in metric:
        k = metric.split("_pass@", 1)[1].split(",", 1)[0]
        label = f"pass@{k} (m={m})"
    elif repeated or "_mean_repeats" in metric:
        label = f"mean@{m}"
    else:
        return ""
    modes = {summary["repeat_protocols"].get(name, {}).get("sampling_enabled")
             for name in contributors or [task]}
    if modes == {False}:
        label += " (greedy)"
    return label


def compute_mgsm_avg(summary):
    vals = []
    protocols = summary.get("repeat_protocols", {})
    for k, v in summary.items():
        metric = "exact_match_mean_repeats" if k.split("/", 1)[0] in protocols else "exact_match"
        if "mgsm" in k and k.endswith(metric) and "stderr" not in k:
            try:
                vals.append(float(v))
            except (ValueError, TypeError):
                pass
    return sum(vals) / len(vals) if vals else None
