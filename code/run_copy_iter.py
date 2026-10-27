"""Copy-collapsed iterated agreement on the dated source files.

The clearance threshold, the mixture rate, and the answer rule stay
those of the dated-file protocol. A copied issuer is one whose
(item, value) set on that day is contained in another issuer's set.
Iteration refines that day's labels from the previous weight vector
before the records are updated. Answers still use the previous day's
weights. This file does not rewrite fusion.json.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import run_fusion as fusion

OUT = ROOT / "results" / "copy_iter.json"
ROUNDS = 4
PRIOR = fusion.PRIOR
TAU = fusion.TAU
ETA = fusion.ETA


def perm_p(deltas):
    nonzero = [item for item in deltas if item]
    if not nonzero:
        return {"mean": 0.0, "std": 0.0, "p": 1.0, "n": len(deltas)}
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
    var = sum((item - mean) ** 2 for item in deltas) / (len(deltas) - 1)
    return {
        "mean": mean,
        "std": var ** 0.5,
        "p": max(extreme / 100000, 1e-5),
        "n": len(deltas),
    }


def collapse(claims):
    grouped = defaultdict(set)
    for claim in claims:
        grouped[claim[2]].add((claim[0], claim[1]))
    order = sorted(grouped, key=lambda issuer: (-len(grouped[issuer]), issuer))
    parent = {}
    for index, issuer in enumerate(order):
        own = grouped[issuer]
        if not own:
            continue
        for other in order[:index]:
            if own <= grouped[other]:
                parent[issuer] = parent.get(other, other)
                break
    folded = 0
    rewritten = []
    for slot, value, issuer, org in claims:
        canon = parent.get(issuer, issuer)
        if canon != issuer:
            folded += 1
        rewritten.append((slot, value, canon, org))
    return rewritten, len(parent)


def label_of(pairs, weights, cleared):
    buckets = defaultdict(float)
    for value, issuer in pairs:
        weight = weights.get(issuer, PRIOR)
        if cleared and weight + 1e-9 < TAU:
            continue
        buckets[value] += weight
    if not buckets and cleared:
        return label_of(pairs, weights, False)
    if not buckets:
        return None
    return max(buckets, key=lambda value: (buckets[value], value))


def fit_epoch(claims, previous):
    by_slot = defaultdict(list)
    for slot, value, issuer, _org in claims:
        by_slot[slot].append((value, issuer))
    weights = dict(previous)
    labels = {}
    for _ in range(ROUNDS):
        labels = {}
        loss = defaultdict(int)
        counts = defaultdict(int)
        for slot, pairs in by_slot.items():
            chosen = label_of(pairs, weights, True)
            if chosen is None:
                continue
            labels[slot] = chosen
            seen = set()
            for value, issuer in pairs:
                if issuer in seen:
                    continue
                seen.add(issuer)
                counts[issuer] += 1
                if value != chosen:
                    loss[issuer] += 1
        weights = {
            issuer: fusion.additive_from_hits(counts[issuer] - loss[issuer], counts[issuer])
            for issuer in counts
        }
    return labels, weights


class Books:
    def __init__(self):
        self.soc = {}
        self.org_hits = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        self.belief = {}

    def org_weight(self, org, issuer):
        hits, count = self.org_hits[org][issuer]
        if count < 2:
            return PRIOR
        return fusion.additive_from_hits(hits, count)

    def active(self, org, issuer, kind):
        if kind == "society":
            return self.soc.get(issuer, PRIOR)
        return self.org_weight(org, issuer)

    def speak(self, pairs, org, kind):
        best_value = None
        best_key = None
        for value, issuer in pairs:
            weight = self.active(org, issuer, kind)
            if weight + 1e-9 < TAU:
                continue
            key = (weight, issuer)
            if best_key is None or key > best_key:
                best_key = key
                best_value = value
        if best_value is None:
            return None
        return best_value

    def vote(self, pairs, org, kind):
        buckets = defaultdict(float)
        for value, issuer in pairs:
            weight = self.active(org, issuer, kind)
            if weight + 1e-9 < TAU:
                continue
            buckets[value] += weight
        if not buckets:
            return None
        return max(buckets, key=lambda value: (buckets[value], value))

    def answer(self, pairs, org, slot, kind, mode):
        spoken = self.vote(pairs, org, kind) if mode == "vote" else self.speak(pairs, org, kind)
        if spoken is None:
            return self.belief.get(slot)
        return spoken

    def update(self, claims, labels):
        epoch = defaultdict(lambda: [0, 0])
        seen = set()
        for slot, value, issuer, org in claims:
            key = (slot, issuer)
            if key in seen or slot not in labels:
                continue
            seen.add(key)
            hit = int(value == labels[slot])
            epoch[issuer][0] += hit
            epoch[issuer][1] += 1
            bucket = self.org_hits[org][issuer]
            bucket[0] += hit
            bucket[1] += 1
        for issuer, (hits, count) in epoch.items():
            fresh = fusion.additive_from_hits(hits, count)
            old = self.soc.get(issuer, PRIOR)
            self.soc[issuer] = (1.0 - ETA) * old + ETA * fresh


def run_corpus(name, days, copy):
    books = Books()
    previous = None
    rows = {key: [] for key in ("society", "org", "vote")}
    folded = []
    changed = {"society": [], "org": [], "vote": []}
    for index, day in enumerate(days):
        claims = day["claims"]
        if copy:
            claims, n_folded = collapse(claims)
            folded.append(n_folded)
        by_slot = defaultdict(list)
        org_of = {}
        seen_pair = set()
        for slot, value, issuer, org in claims:
            key = (slot, issuer)
            if key in seen_pair:
                continue
            seen_pair.add(key)
            by_slot[slot].append((value, issuer))
            org_of[slot] = org
        if index:
            for slot, gold in day["gold"].items():
                if slot not in by_slot:
                    continue
                pairs = by_slot[slot]
                org = org_of[slot]
                pred = {
                    "society": books.answer(pairs, org, slot, "society", "speak"),
                    "org": books.answer(pairs, org, slot, "org", "speak"),
                    "vote": books.answer(pairs, org, slot, "society", "vote"),
                }
                for key, value in pred.items():
                    hit = int(value == gold)
                    rows[key].append(hit)
                    if previous.get(slot) != gold:
                        changed[key].append(hit)
                spoken = pred["society"]
                if spoken is not None:
                    books.belief[slot] = spoken
        labels, _epoch = fit_epoch(claims, books.soc)
        books.update(claims, labels)
        previous = dict(day["gold"])
        print(f"{name} copy={int(copy)} day {index} n={len(rows['society'])}", flush=True)
    summary = {}
    for key in rows:
        summary[key] = {
            "em": sum(rows[key]) / len(rows[key]),
            "changed": sum(changed[key]) / len(changed[key]),
            "n": len(rows[key]),
            "changed_n": len(changed[key]),
        }
    tests = {
        "vote_minus_society": perm_p([vote - base for vote, base in zip(rows["vote"], rows["society"])]),
        "org_minus_society": perm_p([org - base for org, base in zip(rows["org"], rows["society"])]),
    }
    return {"summary": summary, "tests": tests, "issuers_folded": folded}


def main():
    stock = fusion.stock_days()
    flight = fusion.flight_days()
    payload = {
        "rounds": ROUNDS,
        "stock": run_corpus("stock", stock, False),
        "stock_copy": run_corpus("stock", stock, True),
        "flight": run_corpus("flight", flight, False),
        "flight_copy": run_corpus("flight", flight, True),
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key]["summary"] for key in payload if key != "rounds"}, indent=2))


if __name__ == "__main__":
    main()
