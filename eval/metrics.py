"""Aggregate per-(question, permutation) records into accuracy tables and position-bias diagnostics."""
from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, pstdev

from .scorers import wilson

GROUP_KEYS = ("dataset", "category", "subject", "level", "grade", "cultural_sensitivity")


def group_value(item: dict, key: str):
    return item.get(key, item.get("meta", {}).get(key))


def aggregate_items(items: dict[str, dict], records: list[dict]) -> dict[str, dict]:
    """Combine permutations per question: mean of option probabilities (in original order) -> vote."""
    by_uid = defaultdict(list)
    for r in records:
        by_uid[r["uid"]].append(r)
    out = {}
    for uid, rs in by_uid.items():
        n = len(items[uid]["options"])
        rs.sort(key=lambda r: r["perm_k"])
        mean_probs = [mean(r["probs_orig"][j] for r in rs) for j in range(n)]
        if any(sum(r["probs_orig"]) > 0 for r in rs):
            vote = max(range(n), key=lambda j: (mean_probs[j], -j)) + 1
        else:
            vote = None
        preds = [r["pred_orig"] for r in rs]
        out[uid] = {
            "gold": items[uid].get("answer"), "pred": rs[0]["pred_orig"], "vote": vote,
            "consistent": len(set(preds)) == 1 and preds[0] is not None,
            "mean_probs": mean_probs,
        }
    return out


def _acc_block(uids: list[str], agg: dict, records_by_perm: dict[int, dict[str, dict]]) -> dict:
    n = len(uids)
    k = sum(agg[u]["pred"] == agg[u]["gold"] for u in uids)
    kv = sum(agg[u]["vote"] == agg[u]["gold"] for u in uids)
    per_perm = [sum(rp[u]["correct"] for u in uids) / n for rp in records_by_perm.values()]
    lo, hi = wilson(k, n)
    return {
        "n": n, "acc": k / n, "ci95": [lo, hi], "acc_vote": kv / n,
        "acc_perm_mean": mean(per_perm), "acc_perm_std": pstdev(per_perm) if len(per_perm) > 1 else 0.0,
        "consistency": sum(agg[u]["consistent"] for u in uids) / n,
    }


def compute_metrics(items: dict[str, dict], records: list[dict]) -> dict:
    agg = aggregate_items(items, records)
    records_by_perm: dict[int, dict[str, dict]] = defaultdict(dict)
    for r in records:
        records_by_perm[r["perm_k"]][r["uid"]] = r
    uids = sorted(agg)

    m = {"overall": _acc_block(uids, agg, records_by_perm), "groups": {}}
    for key in GROUP_KEYS:
        groups = defaultdict(list)
        for u in uids:
            v = group_value(items[u], key)
            if v is not None:
                groups[str(v)].append(u)
        if len(groups) > 1 or (key in ("dataset", "subject", "category") and groups):
            m["groups"][key] = {g: _acc_block(us, agg, records_by_perm) for g, us in sorted(groups.items())}
    # Macro average over subjects (each subject weighted equally)
    if "subject" in m["groups"]:
        m["overall"]["acc_macro_subject"] = mean(b["acc"] for b in m["groups"]["subject"].values())

    # Position bias over all (question, permutation) records
    n_opt = Counter(len(items[r["uid"]]["options"]) for r in records).most_common(1)[0][0]
    rs = [r for r in records if len(items[r["uid"]]["options"]) == n_opt]
    pred_pos = Counter(r["pred_display"] for r in rs)
    gold_pos = Counter(r["gold_display"] for r in rs)
    acc_by_gold = {p: mean(r["correct"] for r in rs if r["gold_display"] == p) for p in sorted(gold_pos)}
    m["position_bias"] = {
        "n_options": n_opt,
        "pred_dist": {str(p): pred_pos.get(p, 0) / len(rs) for p in range(1, n_opt + 1)},
        "pred_none": pred_pos.get(None, 0) / len(rs),
        "gold_dist": {str(p): gold_pos.get(p, 0) / len(rs) for p in range(1, n_opt + 1)},
        "acc_by_gold_position": {str(p): a for p, a in acc_by_gold.items()},
    }
    if any("label_mass" in r for r in records):
        m["overall"]["label_mass_mean"] = mean(r["label_mass"] for r in records if "label_mass" in r)
    if any(r.get("method") == "generate" for r in records):
        m["overall"]["parse_fail_rate"] = mean(r["pred_display"] is None for r in records)
    return m


def markdown_report(name: str, m: dict) -> str:
    o = m["overall"]
    lines = [f"# {name}", "",
             f"**Accuracy (original order): {o['acc']:.4f}** "
             f"[95% CI {o['ci95'][0]:.3f}-{o['ci95'][1]:.3f}], n={o['n']}  ",
             f"Permutation-vote accuracy: {o['acc_vote']:.4f}  ",
             f"Mean ± std over permutations: {o['acc_perm_mean']:.4f} ± {o['acc_perm_std']:.4f}  ",
             f"Consistency across permutations: {o['consistency']:.3f}  "]
    if "acc_macro_subject" in o:
        lines.append(f"Macro average over subjects: {o['acc_macro_subject']:.4f}  ")
    if "label_mass_mean" in o:
        lines.append(f"Mean probability mass on label tokens: {o['label_mass_mean']:.3f}  ")
    if "parse_fail_rate" in o:
        lines.append(f"Unparseable generations: {o['parse_fail_rate']:.3f}  ")
    for key, groups in m["groups"].items():
        lines += ["", f"## By {key}", "", "| group | n | acc | 95% CI | vote acc | consistency |", "|---|---:|---:|---|---:|---:|"]
        for g, b in groups.items():
            lines.append(f"| {g} | {b['n']} | {b['acc']:.3f} | {b['ci95'][0]:.2f}-{b['ci95'][1]:.2f} | "
                         f"{b['acc_vote']:.3f} | {b['consistency']:.2f} |")
    pb = m["position_bias"]
    lines += ["", "## Position bias (all permutations)", "", "| position | predicted | gold | acc when gold here |", "|---|---:|---:|---:|"]
    for p in pb["pred_dist"]:
        lines.append(f"| {p} | {pb['pred_dist'][p]:.3f} | {pb['gold_dist'][p]:.3f} | "
                     f"{pb['acc_by_gold_position'].get(p, float('nan')):.3f} |")
    return "\n".join(lines) + "\n"
