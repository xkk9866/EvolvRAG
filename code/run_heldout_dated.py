"""Held-out agreement on the dated files.

Copy collapse stays the society containment rule. The answer is still the
highest source at or above 0.55, or the previous belief when nobody clears.
The weight for an item excludes that item: today's other items are allowed
to update the source, and the item being scored is not. The carried record
after the day uses the full day, matching the synthetic schedule.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import run_copy_dated as dated
import run_fusion as fusion


def majority_map(claims):
    by_slot = defaultdict(list)
    for slot, value, issuer, _org in claims:
        by_slot[slot].append((value, issuer))
    truth = {}
    for slot, pairs in by_slot.items():
        buckets = defaultdict(int)
        for value, _issuer in pairs:
            buckets[value] += 1
        truth[slot] = max(buckets, key=lambda value: (buckets[value], value))
    return truth


def additive(hits, count):
    if count < 2:
        return None
    return fusion.additive_from_hits(hits, count)


def run_days(days):
    soc = {}
    belief = {}
    held_belief = {}
    base_hits = []
    held_hits = []
    base_changed = []
    held_changed = []
    for index, day in enumerate(days):
        claims, _folded = dated.collapse(day["claims"], "subset")
        truth = majority_map(claims)
        hits = defaultdict(int)
        counts = defaultdict(int)
        spoke = {}
        for slot, value, issuer, _org in claims:
            if (slot, issuer) in spoke:
                continue
            spoke[(slot, issuer)] = 1 if value == truth[slot] else 0
            counts[issuer] += 1
            hits[issuer] += spoke[(slot, issuer)]
        if index == 0:
            for issuer, weight in fusion.rate_fit(
                [(slot, value, issuer) for slot, value, issuer, _org in claims]
            ).items():
                old = soc.get(issuer, fusion.PRIOR)
                soc[issuer] = (1.0 - fusion.ETA) * old + fusion.ETA * weight
            continue
        previous = days[index - 1]["gold"]
        by_slot = defaultdict(list)
        for slot, value, issuer, _org in claims:
            by_slot[slot].append((value, issuer))
        for slot, answer in day["gold"].items():
            pairs = by_slot.get(slot, [])
            base = pick(pairs, soc, belief, slot)
            local = {}
            for issuer in {issuer for _value, issuer in pairs}:
                old = soc.get(issuer, fusion.PRIOR)
                if (slot, issuer) in spoke:
                    remain_n = counts[issuer] - 1
                    remain_h = hits[issuer] - spoke[(slot, issuer)]
                else:
                    remain_n = counts[issuer]
                    remain_h = hits[issuer]
                today = additive(remain_h, remain_n)
                local[issuer] = old if today is None else (1.0 - fusion.ETA) * old + fusion.ETA * today
            held = pick(pairs, local, held_belief, slot)
            if base is not None:
                belief[slot] = base
            if held is not None:
                held_belief[slot] = held
            base_hits.append(int(base == answer))
            held_hits.append(int(held == answer))
            changed = slot in previous and previous[slot] != answer
            if changed:
                base_changed.append(int(base == answer))
                held_changed.append(int(held == answer))
        for issuer, weight in fusion.rate_fit(
            [(slot, value, issuer) for slot, value, issuer, _org in claims]
        ).items():
            old = soc.get(issuer, fusion.PRIOR)
            soc[issuer] = (1.0 - fusion.ETA) * old + fusion.ETA * weight
        print(index, "base", sum(base_hits) / len(base_hits), "held", sum(held_hits) / len(held_hits), flush=True)
    return {
        "base": rate(base_hits, base_changed),
        "held": rate(held_hits, held_changed),
        "paired_all": paired(held_hits, base_hits),
        "paired_changed": paired(held_changed, base_changed),
    }


def pick(pairs, weights, belief, slot):
    best = None
    best_key = None
    for value, issuer in pairs:
        weight = weights.get(issuer, fusion.PRIOR)
        if weight + 1e-9 < fusion.TAU:
            continue
        key = (weight, issuer)
        if best_key is None or key > best_key:
            best = value
            best_key = key
    if best is None:
        best = belief.get(slot)
    return best


def rate(hits, changed):
    return {
        "em": sum(hits) / len(hits),
        "changed": sum(changed) / len(changed),
        "n": len(hits),
        "changed_n": len(changed),
    }


def paired(left, right):
    deltas = [a - b for a, b in zip(left, right)]
    nonzero = [item for item in deltas if item]
    if not nonzero:
        return {"mean": 0.0, "std": 0.0, "p": 1.0, "n": len(deltas), "nonzero": 0}
    total = sum(nonzero)
    bits = len(nonzero)
    negative = 0
    for index, item in enumerate(nonzero):
        if item < 0:
            negative |= 1 << index
    rng = random.Random(3157)
    extreme = 1
    for _ in range(99999):
        agreed = rng.getrandbits(bits) ^ negative
        draw = 2 * agreed.bit_count() - bits
        if abs(draw) >= abs(total):
            extreme += 1
    mean = sum(deltas) / len(deltas)
    var = sum((item - mean) ** 2 for item in deltas) / max(len(deltas) - 1, 1)
    return {
        "mean": mean,
        "std": var ** 0.5,
        "p": max(extreme / 100000.0, 1e-5),
        "n": len(deltas),
        "nonzero": bits,
    }


def main():
    corpus = sys.argv[1]
    days = fusion.flight_days() if corpus == "flight" else fusion.stock_days()
    payload = run_days(days)
    path = ROOT / "results" / f"heldout_dated_{corpus}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
