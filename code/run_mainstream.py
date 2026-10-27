"""Mainstream truth-discovery baselines on the same collapsed rounds.

Majority, Sums, Investment, 2-Estimates, and CATD are the categorical
estimators that truth-discovery papers compare as a block. Decay is the
single-pass temporal vote with a one-epoch half-life. Constants are fixed
before these seeds are scored: Investment growth 1.4, ten iterations, CATD
chi-square level 0.95, decay half-life of one epoch. Copy collapse and the
schema match the held-out operator. Source parameters carry across rounds.
A tied vote does not replace the incumbent.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as rx
import run_fair as fair
import run_heldout as held

REGIMES = ("stationary", "topic", "open_rumor", "sybil")
GROWTH = 1.4
ITERS = 10
HALF = 0.5


def chi2_95(df):
    """Wilson-Hilferty approximation of the 0.95 chi-square quantile."""
    if df <= 0:
        return 0.0
    z = 1.6448536269514722
    term = 2.0 / (9.0 * df)
    return df * (1.0 - term + z * math.sqrt(term)) ** 3


def dedup(claims):
    latest = {}
    for slot, value, issuer in claims:
        latest[(slot, issuer)] = value
    return latest


def unique_argmax(scores):
    if not scores:
        return None
    best = max(scores.values())
    winners = [value for value, score in scores.items() if score == best]
    if len(winners) != 1:
        return None
    return winners[0]


def adopt(world, belief, order, chosen):
    row = list(belief)
    for slot in order:
        schema = fair.schema_value(world, row, slot)
        if slot in chosen and chosen[slot] is not None:
            row[slot] = chosen[slot]
        elif schema is not None:
            row[slot] = schema
    return row


def majority_round(claims):
    rows = dedup(claims)
    buckets = defaultdict(lambda: defaultdict(set))
    for (slot, issuer), value in rows.items():
        buckets[slot][value].add(issuer)
    chosen = {}
    for slot, values in buckets.items():
        scores = {value: len(members) for value, members in values.items()}
        chosen[slot] = unique_argmax(scores)
    return chosen


def sums_round(claims, trust):
    rows = dedup(claims)
    if not rows:
        return {}
    issuers = {issuer for _slot, issuer in rows}
    weight = {issuer: trust.get(issuer, 1.0) for issuer in issuers}
    scores = defaultdict(lambda: defaultdict(float))
    for _ in range(ITERS):
        scores = defaultdict(lambda: defaultdict(float))
        for (slot, issuer), value in rows.items():
            scores[slot][value] += max(weight[issuer], 0.0)
        gained = defaultdict(float)
        for (slot, issuer), value in rows.items():
            gained[issuer] += scores[slot][value]
        peak = max(gained.values()) if gained else 0.0
        if peak <= 0:
            break
        weight = {issuer: gained[issuer] / peak for issuer in gained}
    trust.update(weight)
    return {slot: unique_argmax(dict(values)) for slot, values in scores.items()}


def investment_round(claims, trust):
    rows = dedup(claims)
    by_issuer = defaultdict(list)
    for (slot, issuer), value in rows.items():
        by_issuer[issuer].append((slot, value))
    if not by_issuer:
        return {}
    weight = {issuer: trust.get(issuer, 1.0) for issuer in by_issuer}
    scores = defaultdict(lambda: defaultdict(float))
    for _ in range(ITERS):
        scores = defaultdict(lambda: defaultdict(float))
        share = {}
        for issuer, items in by_issuer.items():
            share[issuer] = max(weight.get(issuer, 1.0), 0.0) / len(items)
            for slot, value in items:
                scores[slot][value] += share[issuer]
        returned = defaultdict(float)
        for issuer, items in by_issuer.items():
            for slot, value in items:
                base = scores[slot][value]
                if base <= 0:
                    continue
                returned[issuer] += (base ** GROWTH) * share[issuer] / base
        weight = returned
    trust.update(weight)
    return {slot: unique_argmax(dict(values)) for slot, values in scores.items()}


def two_estimates_round(claims, trust):
    rows = dedup(claims)
    by_slot = defaultdict(dict)
    for (slot, issuer), value in rows.items():
        by_slot[slot][issuer] = value
    if not by_slot:
        return {}
    issuers = {issuer for votes in by_slot.values() for issuer in votes}
    weight = {issuer: trust.get(issuer, 1.0) for issuer in issuers}
    scores = {}
    for _ in range(ITERS):
        scores = {}
        for slot, votes in by_slot.items():
            totals = defaultdict(float)
            for issuer, value in votes.items():
                totals[value] += weight[issuer]
            mass = sum(totals.values())
            scores[slot] = {value: (2.0 * total - mass) for value, total in totals.items()}
        gained = defaultdict(float)
        count = defaultdict(int)
        for slot, votes in by_slot.items():
            for issuer, value in votes.items():
                gained[issuer] += scores[slot][value]
                count[issuer] += 1
        if not count:
            break
        centered = {issuer: gained[issuer] / count[issuer] for issuer in count}
        center = sum(centered.values()) / len(centered)
        weight = {issuer: value - center for issuer, value in centered.items()}
    trust.update(weight)
    return {slot: unique_argmax(scores[slot]) for slot in scores}


def catd_round(claims):
    rows = dedup(claims)
    by_slot = defaultdict(dict)
    for (slot, issuer), value in rows.items():
        by_slot[slot][issuer] = value
    if not by_slot:
        return {}
    truth = {}
    for slot, votes in by_slot.items():
        tallies = defaultdict(int)
        for value in votes.values():
            tallies[value] += 1
        truth[slot] = unique_argmax(tallies)
    for _ in range(ITERS):
        wrong = defaultdict(float)
        count = defaultdict(int)
        for slot, votes in by_slot.items():
            label = truth.get(slot)
            for issuer, value in votes.items():
                count[issuer] += 1
                if label is None or value != label:
                    wrong[issuer] += 1.0
        weight = {}
        for issuer, n in count.items():
            var = max(wrong[issuer] / n, 1.0 / (n + 1.0))
            weight[issuer] = chi2_95(n) / var
        scores = defaultdict(lambda: defaultdict(float))
        for slot, votes in by_slot.items():
            for issuer, value in votes.items():
                scores[slot][value] += weight[issuer]
        updated = {slot: unique_argmax(dict(values)) for slot, values in scores.items()}
        if updated == truth:
            break
        truth = updated
    return truth


def decay_choose(history, correct, total, epoch_i):
    latest = {}
    for stamp, slot, value, issuer in history:
        latest[(slot, issuer)] = (stamp, value)
    scores = defaultdict(lambda: defaultdict(float))
    for (slot, issuer), (stamp, value) in latest.items():
        n = total.get(issuer, 0)
        weight = (1.0 + correct.get(issuer, 0.0)) / (2.0 + n)
        age = max(0, epoch_i - stamp)
        scores[slot][value] += weight * (HALF ** age)
    return {slot: unique_argmax(dict(values)) for slot, values in scores.items()}


def remember(correct, total, claims, chosen):
    rows = dedup(claims)
    for (slot, issuer), value in rows.items():
        if slot not in chosen or chosen[slot] is None:
            continue
        total[issuer] = total.get(issuer, 0) + 1
        if value == chosen[slot]:
            correct[issuer] = correct.get(issuer, 0.0) + 1.0


def run_world(world):
    names = world.names
    order = list(world.mutable_ids) + list(world.stable_ids) + list(world.derived_ids)
    modes = ("majority", "sums", "invest", "two", "catd", "decay")
    belief = {key: list(world.init_belief[0]) for key in modes}
    state = {key: {} for key in ("sums", "invest", "two")}
    correct, total = {}, {}
    history = []
    answers = {key: [] for key in modes}
    mail = []
    for epoch_i, epoch_docs in enumerate(world.visible):
        mail.extend(epoch_docs)
        find = fair.copy_find(mail, names)
        fresh = fair.collapsed(epoch_docs, find)
        claims, _max_id = held.claims_from(fresh, names)
        for slot, value, issuer in claims:
            history.append((epoch_i, slot, value, issuer))
        chosen = {
            "majority": majority_round(claims),
            "sums": sums_round(claims, state["sums"]),
            "invest": investment_round(claims, state["invest"]),
            "two": two_estimates_round(claims, state["two"]),
            "catd": catd_round(claims),
            "decay": decay_choose(history, correct, total, epoch_i),
        }
        remember(correct, total, claims, chosen["decay"])
        for key in modes:
            belief[key] = adopt(world, belief[key], order, chosen[key])
            answers[key].append([list(belief[key]) for _ in range(rx.N_AGENTS)])
    packed = {}
    for key, grid in answers.items():
        counts = rx.empty_counts()
        rx.consume(counts, world, grid, 0.0)
        packed[key] = rx.pack_metrics(counts)
    return packed


def balanced(metrics):
    return (metrics["changed"] + metrics["stable"] + metrics["derived"]) / 3.0


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "compare"
    seeds = {
        "probe": [800, 801],
        "dev": list(range(800, 816)),
        "compare": list(range(700, 732)),
        "test": list(range(900, 932)),
    }[task]
    modes = ("majority", "sums", "invest", "two", "catd", "decay")
    stored = {mode: {regime: [] for regime in REGIMES} for mode in modes}
    for regime in REGIMES:
        for seed in seeds:
            packed = run_world(fair.load_world(seed, regime))
            bits = []
            for mode in modes:
                stored[mode][regime].append(
                    {key: packed[mode][key] for key in ("sp", "changed", "stable", "derived", "obsolete", "delay", "cfa")}
                )
                bits.append(f"{mode}:{balanced(packed[mode]):.3f}")
            print(regime, seed, " ".join(bits), flush=True)
    summary = {}
    for mode in modes:
        by = {regime: sum(balanced(row) for row in rows) / len(rows) for regime, rows in stored[mode].items()}
        summary[mode] = {"by_regime": by, "mean": sum(by.values()) / len(by)}
    out = rx.ROOT / "results" / f"mainstream_{task}.json"
    out.write_text(json.dumps({"task": task, "summary": summary, "seeds": stored}), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
