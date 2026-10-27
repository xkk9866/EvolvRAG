"""Headroom of the record switch, and whether a separated local record is the one that matches gold.

The count route is the published comparison. The separated route uses the
organization record only when that record clears more sources and its cleared
weights sit at or above 0.60 while its blocked weights sit at or below 0.40.
The oracle picks, for each item, whichever of the two records matches gold.
It is not a method.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_fusion as fusion

ROOT = fusion.ROOT
OUT = ROOT / "results" / "headroom.json"
TAU = fusion.TAU


def separated(weights):
    cleared = [weight for weight in weights.values() if weight + 1e-9 >= TAU]
    blocked = [weight for weight in weights.values() if weight + 1e-9 < TAU]
    if not cleared:
        return False
    if min(cleared) + 1e-9 < 0.60:
        return False
    if blocked and max(blocked) > 0.40 + 1e-9:
        return False
    return True


def speak(weights, pairs):
    best = None
    best_key = None
    for value, issuer in pairs:
        weight = weights.get(issuer, fusion.PRIOR)
        if weight + 1e-9 < TAU:
            continue
        key = (weight, issuer)
        if best_key is None or key > best_key:
            best = value
            best_key = key
    return best


def perm_p(diffs):
    obs = abs(sum(diffs))
    if obs <= 1e-12:
        return 1.0
    wins = sum(diff > 0 for diff in diffs)
    losses = sum(diff < 0 for diff in diffs)
    rng = random.Random(3157)
    count = 1
    for _ in range(99999):
        plus_wins = rng.getrandbits(wins).bit_count() if wins else 0
        plus_losses = rng.getrandbits(losses).bit_count() if losses else 0
        signed = (2 * plus_wins - wins) - (2 * plus_losses - losses)
        if abs(signed) + 1e-9 >= obs:
            count += 1
    return max(count / 100000.0, 1e-5)


def mean_std(values):
    n = len(values)
    mu = sum(values) / n
    var = sum((value - mu) ** 2 for value in values) / (n - 1)
    return mu, var ** 0.5


def analyze(name, days):
    book = fusion.Books()
    scores = {key: [] for key in ("society", "org", "route", "separated", "oracle")}
    switch_hits = {"separated_switch": [], "unseparated_switch": [], "stay": []}
    source_hit = defaultdict(int)
    source_n = defaultdict(int)
    for index, day in enumerate(days):
        if index == 0:
            book.update(day["claims"])
            continue
        by_slot = defaultdict(list)
        org_of = {}
        for slot, value, issuer, org in day["claims"]:
            by_slot[slot].append((value, issuer))
            org_of[slot] = org
            answer = day["gold"].get(slot)
            if answer is None:
                continue
            source_n[issuer] += 1
            source_hit[issuer] += int(value == answer)
        for slot, answer in day["gold"].items():
            pairs = by_slot.get(slot, [])
            org = org_of.get(slot)
            if org is None:
                continue
            society = book.weights(org, "society")
            local = book.weights(org, "org")
            route = book.weights(org, "hier")
            soc_value = speak(society, pairs)
            org_value = speak(local, pairs)
            route_value = speak(route, pairs)
            if soc_value is None:
                soc_value = book.belief["society"].get(slot)
            if org_value is None:
                org_value = book.belief["org"].get(slot)
            if route_value is None:
                route_value = book.belief["hier"].get(slot)
            use_sep = separated(local) and sum(weight + 1e-9 >= TAU for weight in local.values()) >= sum(
                weight + 1e-9 >= TAU for weight in society.values()
            ) + 1
            sep_value = org_value if use_sep else soc_value
            oracle = answer if (soc_value == answer or org_value == answer) else soc_value
            row = {
                "society": int(soc_value == answer),
                "org": int(org_value == answer),
                "route": int(route_value == answer),
                "separated": int(sep_value == answer),
                "oracle": int(oracle == answer),
            }
            for key, hit in row.items():
                scores[key].append(hit)
            switched = route_value != soc_value or (
                sum(weight + 1e-9 >= TAU for weight in local.values())
                >= sum(weight + 1e-9 >= TAU for weight in society.values()) + 1
            )
            if not switched:
                switch_hits["stay"].append(row["society"])
            elif separated(local):
                switch_hits["separated_switch"].append(row["route"])
            else:
                switch_hits["unseparated_switch"].append(row["route"])
            if soc_value is not None:
                book.belief["society"][slot] = soc_value
            if org_value is not None:
                book.belief["org"][slot] = org_value
            if route_value is not None:
                book.belief["hier"][slot] = route_value
        book.update(day["claims"])
        print(name, day["day"], "n", len(scores["society"]), flush=True)
    summary = {}
    for key, hits in scores.items():
        summary[key] = {"em": sum(hits) / len(hits), "n": len(hits)}
    tests = {}
    for left, right in (("separated", "route"), ("separated", "society"), ("oracle", "society"), ("route", "society")):
        diffs = [a - b for a, b in zip(scores[left], scores[right])]
        mu, sd = mean_std(diffs)
        tests[f"{left}_minus_{right}"] = {"mean": mu, "std": sd, "p": perm_p(diffs), "n": len(diffs)}
    groups = {key: {"n": len(hits), "em": (sum(hits) / len(hits) if hits else None)} for key, hits in switch_hits.items()}
    ranked = sorted(source_n, key=lambda issuer: source_n[issuer], reverse=True)
    sources = [
        {"issuer": issuer, "n": source_n[issuer], "em": source_hit[issuer] / source_n[issuer]}
        for issuer in ranked[:12]
    ]
    return {"summary": summary, "tests": tests, "groups": groups, "sources": sources}


def main():
    print("stock", flush=True)
    stock = analyze("stock", fusion.stock_days())
    print("flight", flush=True)
    flight = analyze("flight", fusion.flight_days())
    payload = {"stock": stock, "flight": flight}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
