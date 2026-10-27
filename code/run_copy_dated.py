"""Dated files with the same copy collapse used in the society.

Labels and weights stay on the additive agreement count. Answers use
the previous day's weights and the same 0.55 cutoff. Collapse is applied
before the fit and the decision. exact keeps an issuer only when another
issuer has the same (item, value) set. subset also folds a proper subset
into the larger issuer, which is the society rule. Neither variant is
written back into fusion.json.
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

OUT = ROOT / "results" / "copy_dated.json"


def collapse(claims, mode):
    if mode == "none":
        return claims, 0
    grouped = defaultdict(set)
    for slot, value, issuer, _org in claims:
        grouped[issuer].add((slot, value))
    order = sorted(grouped, key=lambda issuer: (-len(grouped[issuer]), issuer))
    parent = {}
    for index, issuer in enumerate(order):
        own = grouped[issuer]
        if not own:
            continue
        for other in order[:index]:
            contained = own <= grouped[other]
            if not contained:
                continue
            if mode == "exact" and own != grouped[other]:
                continue
            parent[issuer] = parent.get(other, other)
            break
    rewritten = []
    seen = set()
    for slot, value, issuer, org in claims:
        canon = parent.get(issuer, issuer)
        key = (slot, canon)
        if key in seen:
            continue
        seen.add(key)
        rewritten.append((slot, value, canon, org))
    return rewritten, len(parent)


def paired(left, right):
    deltas = [a - b for a, b in zip(left, right)]
    nonzero = [item for item in deltas if item]
    total = sum(nonzero)
    bits = len(nonzero)
    if bits == 0:
        return {"mean": 0.0, "std": 0.0, "p": 1.0, "n": len(deltas)}
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
        "p": max(extreme / 100000, 1e-5),
        "n": len(deltas),
    }


class Slim:
    """Agreement-count records only. The log-score fit is not needed here."""

    def __init__(self):
        self.soc = {}
        self.org_hits = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        self.belief = {name: {} for name in ("majority", "society", "org", "hier")}
        self.used_org = 0

    def org_weight(self, org, issuer):
        hits, count = self.org_hits[org][issuer]
        return fusion.additive_from_hits(hits, count)

    def weights(self, org, kind):
        if kind == "society":
            return self.soc
        local = {
            issuer: self.org_weight(org, issuer) for issuer in self.org_hits[org]
        }
        if kind == "org":
            return local
        soc_high = sum(weight + 1e-9 >= fusion.TAU for weight in self.soc.values())
        org_high = sum(weight + 1e-9 >= fusion.TAU for weight in local.values())
        if org_high >= soc_high + 1:
            self.used_org += 1
            return local
        return self.soc

    def update(self, claims):
        rows = [(slot, value, issuer) for slot, value, issuer, _org in claims]
        for issuer, weight in fusion.rate_fit(rows).items():
            old = self.soc.get(issuer, fusion.PRIOR)
            self.soc[issuer] = (1.0 - fusion.ETA) * old + fusion.ETA * weight
        grouped = defaultdict(list)
        for slot, value, issuer, org in claims:
            grouped[org].append((slot, value, issuer))
        for org, fresh in grouped.items():
            by_slot = defaultdict(list)
            for slot, value, issuer in fresh:
                by_slot[slot].append((value, issuer))
            truth = {}
            for slot, pairs in by_slot.items():
                buckets = defaultdict(int)
                for value, _issuer in pairs:
                    buckets[value] += 1
                truth[slot] = max(buckets, key=lambda value: (buckets[value], value))
            seen = set()
            for slot, value, issuer in fresh:
                if (slot, issuer) in seen:
                    continue
                seen.add((slot, issuer))
                self.org_hits[org][issuer][1] += 1
                if value == truth[slot]:
                    self.org_hits[org][issuer][0] += 1

    def answer(self, claims, gold, previous_gold):
        by_slot = defaultdict(list)
        org_of = {}
        for slot, value, issuer, org in claims:
            by_slot[slot].append((value, issuer))
            org_of[slot] = org
        item = {name: [] for name in self.belief}
        changed_item = {name: [] for name in self.belief}
        scores = {
            name: {"hit": 0, "n": 0, "changed_hit": 0, "changed_n": 0}
            for name in self.belief
        }
        for slot, answer in gold.items():
            pairs = by_slot.get(slot, [])
            org = org_of.get(slot)
            majority = None
            if pairs:
                buckets = defaultdict(set)
                for value, issuer in pairs:
                    buckets[value].add(issuer)
                majority = max(buckets, key=lambda value: (len(buckets[value]), value))
            for name in self.belief:
                if name == "majority":
                    best = majority
                else:
                    weights = self.weights(org, name) if org is not None else self.soc
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
                        best = self.belief[name].get(slot)
                if best is not None:
                    self.belief[name][slot] = best
                hit = int(best == answer)
                scores[name]["n"] += 1
                scores[name]["hit"] += hit
                item[name].append(hit)
                changed = (
                    previous_gold is not None
                    and slot in previous_gold
                    and previous_gold[slot] != answer
                )
                if changed:
                    scores[name]["changed_n"] += 1
                    scores[name]["changed_hit"] += hit
                    changed_item[name].append(hit)
        scores["_item"] = item
        return scores


def run_corpus(days, mode):
    book = Slim()
    totals = {
        key: {"hit": 0, "n": 0, "changed_hit": 0, "changed_n": 0} for key in book.belief
    }
    items = {key: [] for key in ("society", "hier")}
    folded = []
    for index, day in enumerate(days):
        claims, n_folded = collapse(day["claims"], mode)
        folded.append(n_folded)
        if index == 0:
            book.update(claims)
            continue
        scores = book.answer(claims, day["gold"], days[index - 1]["gold"])
        book.update(claims)
        for key in totals:
            totals[key]["hit"] += scores[key]["hit"]
            totals[key]["n"] += scores[key]["n"]
            totals[key]["changed_hit"] += scores[key]["changed_hit"]
            totals[key]["changed_n"] += scores[key]["changed_n"]
        for key in items:
            items[key].extend(scores["_item"][key])
        print(
            f"{mode} day {index} society {totals['society']['hit'] / totals['society']['n']:.4f}",
            flush=True,
        )
    summary = {}
    for key, row in totals.items():
        summary[key] = {
            "em": row["hit"] / row["n"],
            "changed": row["changed_hit"] / row["changed_n"],
            "n": row["n"],
            "changed_n": row["changed_n"],
            "org_decisions": book.used_org,
        }
    return {"summary": summary, "items": items, "issuers_folded": folded}


def main():
    import gc

    corpus = sys.argv[1]
    modes = (sys.argv[2],) if len(sys.argv) > 2 else ("none", "exact", "subset")
    days = fusion.stock_days() if corpus == "stock" else fusion.flight_days()
    payload = {}
    stored = {}
    for mode in modes:
        result = run_corpus(days, mode)
        stored[mode] = result["items"]
        payload[mode] = {
            "summary": result["summary"],
            "issuers_folded": result["issuers_folded"],
        }
        print(corpus, mode, json.dumps(result["summary"]), flush=True)
        checkpoint = ROOT / "results" / f"copy_dated_{corpus}_{mode}.json"
        checkpoint.write_text(
            json.dumps(
                {
                    "summary": result["summary"],
                    "issuers_folded": result["issuers_folded"],
                    "society": result["items"]["society"],
                    "hier": result["items"]["hier"],
                }
            ),
            encoding="utf-8",
        )
        gc.collect()
    tests = {}
    if "none" in stored:
        for mode in ("exact", "subset"):
            if mode not in stored:
                continue
            tests[f"{mode}_minus_none"] = paired(
                stored[mode]["society"], stored["none"]["society"]
            )
            tests[f"{mode}_hier_minus_society"] = paired(
                stored[mode]["hier"], stored[mode]["society"]
            )
    payload["tests"] = tests
    suffix = "" if len(modes) == 3 else f"_{modes[0]}"
    path = ROOT / "results" / f"copy_dated_{corpus}{suffix}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
