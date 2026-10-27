"""Revision by corroborated accuracy.

A copied issuer is the same observation. A claim updates accuracy only when
another issuer reports the same value; a source cannot certify itself.
Accuracy in an organization adds that organization's corroborated claims to
the shared posterior. The source may revise there when this posterior is
above one half, which is the point at which it is more likely right than
wrong. The revision is the latest testimony among such sources, and the
higher excess accuracy breaks a tie in time. Derived facts with no such
testimony follow the parent schema. Development seeds 600-615 are read
before test seeds 700-731. This file does not write hier.json or fusion.json.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import run_experiment as rx
import run_fair as fair
import run_fusion as fusion

OUT = ROOT / "results" / "evidence.json"
ROUNDS = 4
DEV = list(range(600, 616))
TEST = list(range(700, 732))
REGIMES = ("stationary", "topic", "open_rumor", "sybil")


def alpha_of(hits, count):
    return (1.0 + hits) / (2.0 + count)


def excess_of(hits, count):
    return max(2.0 * alpha_of(hits, count) - 1.0, 0.0)


def latest_pairs(docs, names):
    best = {}
    org_of = {}
    for doc in docs:
        slot = fair.entity_of_text(doc["text"], names)
        issuer = doc["issuer"]
        if slot is None or not isinstance(issuer, int) or issuer < 0:
            continue
        key = (doc["time"], doc.get("seq", 0))
        prev = best.get((slot, issuer))
        if prev is None or key > prev[0]:
            best[(slot, issuer)] = (key, doc["value"])
            org_of[slot] = slot % len(rx.ORGS)
    grouped = defaultdict(list)
    for (slot, issuer), (_key, value) in best.items():
        grouped[slot].append((value, issuer))
    return grouped, org_of


class Record:
    """Shared posterior, plus each organization's extra corroborated claims."""

    def __init__(self, n_org):
        self.n_org = n_org
        self.g_hits = defaultdict(int)
        self.g_n = defaultdict(int)
        self.o_hits = [defaultdict(int) for _ in range(n_org)]
        self.o_n = [defaultdict(int) for _ in range(n_org)]

    def alpha(self, issuer, org):
        return alpha_of(
            self.g_hits[issuer] + self.o_hits[org][issuer],
            self.g_n[issuer] + self.o_n[org][issuer],
        )

    def excess(self, issuer, org):
        return max(2.0 * self.alpha(issuer, org) - 1.0, 0.0)

    def update(self, grouped):
        for slot, pairs in grouped.items():
            org = slot % self.n_org
            support = defaultdict(set)
            seen = set()
            unique = []
            for value, issuer in pairs:
                if issuer in seen:
                    continue
                seen.add(issuer)
                unique.append((value, issuer))
                support[value].add(issuer)
            corroborated = {value: names for value, names in support.items() if len(names) >= 2}
            if not corroborated:
                continue
            label = max(corroborated, key=lambda value: (len(corroborated[value]), value))
            runner = max(
                (len(names) for value, names in support.items() if value != label),
                default=0,
            )
            if len(corroborated[label]) < runner + 2:
                continue
            for value, issuer in unique:
                self.g_n[issuer] += 1
                self.o_n[org][issuer] += 1
                if value == label:
                    self.g_hits[issuer] += 1
                    self.o_hits[org][issuer] += 1


def run_world(world):
    names = world.names
    belief = list(world.init_belief[0])
    book = Record(len(rx.ORGS))
    mail = []
    answers = []
    for epoch_docs in world.visible:
        mail.extend(epoch_docs)
        find = fair.copy_find(mail, names)
        usable = fair.collapsed(mail, find)
        fresh = fair.collapsed(epoch_docs, find)
        fresh_pairs, _org = latest_pairs(fresh, names)
        row = list(belief)
        best = {}
        for doc in usable:
            slot = fair.entity_of_text(doc["text"], names)
            issuer = doc["issuer"]
            if slot is None or not isinstance(issuer, int) or issuer < 0:
                continue
            org = slot % len(rx.ORGS)
            excess = book.excess(issuer, org)
            if excess <= 0.0:
                continue
            key = (doc["time"], doc.get("seq", 0), excess, issuer)
            current = best.get(slot)
            if current is None or key > current[0]:
                best[slot] = (key, doc["value"])
        for slot, (_key, value) in best.items():
            if slot not in world.derived_ids:
                row[slot] = value
        for slot in world.derived_ids:
            if slot in best:
                row[slot] = best[slot][1]
            else:
                schema = fair.schema_value(world, row, slot)
                if schema is not None:
                    row[slot] = schema
        belief = row
        answers.append([list(row) for _ in range(rx.N_AGENTS)])
        book.update(fresh_pairs)
    counts = rx.empty_counts()
    rx.consume(counts, world, answers, 0.0)
    return rx.pack_metrics(counts)


def balanced(metrics):
    return (metrics["changed"] + metrics["stable"] + metrics["derived"]) / 3.0


def eval_seeds(seeds, regimes):
    hier_cfg = fair.RULES[4]
    out = {}
    for regime in regimes:
        rows = []
        base = []
        for seed in seeds:
            world = fair.load_world(seed, regime)
            metrics = run_world(world)
            other = fair.run_cfg(world, hier_cfg)
            rows.append(metrics)
            base.append(other)
            print(
                regime,
                seed,
                f"evidence {balanced(metrics):.3f}",
                f"hier {balanced(other):.3f}",
                f"sp {metrics['sp']:.3f}/{other['sp']:.3f}",
                flush=True,
            )
        cell = fair.cell_of(rows)
        base_cell = fair.cell_of(base)
        out[regime] = {
            "evidence_balanced": balanced(cell),
            "hier_balanced": balanced(base_cell),
            "evidence": {key: cell[key] for key in ("sp", "changed", "stable", "derived")},
            "hier": {key: base_cell[key] for key in ("sp", "changed", "stable", "derived")},
        }
    means = {
        "evidence": sum(out[regime]["evidence_balanced"] for regime in regimes) / len(regimes),
        "hier": sum(out[regime]["hier_balanced"] for regime in regimes) / len(regimes),
    }
    return {"regimes": out, "mean_balanced": means}


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
    var = sum((item - mean) ** 2 for item in deltas) / max(len(deltas) - 1, 1)
    return {"mean": mean, "std": var ** 0.5, "p": max(extreme / 100000, 1e-5), "n": len(deltas)}


class StreamBook:
    """The same excess-accuracy comparison on one dated file."""

    def __init__(self):
        self.g_hits = defaultdict(int)
        self.g_n = defaultdict(int)
        self.o_hits = defaultdict(lambda: defaultdict(int))
        self.o_n = defaultdict(lambda: defaultdict(int))
        self.belief = {}

    def excess(self, org, issuer, local):
        if local:
            return excess_of(self.o_hits[org][issuer], self.o_n[org][issuer])
        return excess_of(self.g_hits[issuer], self.g_n[issuer])

    def choose(self, pairs, org, slot, local):
        buckets = defaultdict(float)
        for value, issuer in pairs:
            buckets[value] += self.excess(org, issuer, local)
        if not buckets or max(buckets.values()) <= 0.0:
            return self.belief.get(slot)
        belief = self.belief.get(slot)
        chosen = max(buckets, key=lambda value: (buckets[value], value == belief, value))
        return chosen

    def prefer_local(self, org, by_slot):
        local_mass = 0.0
        global_mass = 0.0
        for pairs in by_slot.values():
            local_buckets = defaultdict(float)
            global_buckets = defaultdict(float)
            for value, issuer, slot_org in pairs:
                if slot_org != org:
                    continue
                local_buckets[value] += self.excess(org, issuer, True)
                global_buckets[value] += self.excess(org, issuer, False)
            if local_buckets:
                local_mass += max(local_buckets.values())
            if global_buckets:
                global_mass += max(global_buckets.values())
        return local_mass > global_mass

    def update(self, claims, local_org=None):
        grouped = defaultdict(list)
        for slot, value, issuer, org in claims:
            if local_org is not None and org != local_org:
                continue
            grouped[slot].append((value, issuer))
        if not grouped:
            return
        hits = self.o_hits[local_org] if local_org is not None else self.g_hits
        counts = self.o_n[local_org] if local_org is not None else self.g_n
        prev_h = dict(hits)
        prev_n = dict(counts)
        epoch_h = defaultdict(int)
        epoch_n = defaultdict(int)
        working_h = dict(prev_h)
        working_n = dict(prev_n)
        for _ in range(ROUNDS):
            epoch_h = defaultdict(int)
            epoch_n = defaultdict(int)
            for pairs in grouped.values():
                buckets = defaultdict(float)
                seen = set()
                for value, issuer in pairs:
                    if issuer in seen:
                        continue
                    seen.add(issuer)
                    buckets[value] += alpha_of(working_h.get(issuer, 0), working_n.get(issuer, 0))
                if not buckets:
                    continue
                label = max(buckets, key=lambda value: (buckets[value], value))
                seen = set()
                for value, issuer in pairs:
                    if issuer in seen:
                        continue
                    seen.add(issuer)
                    epoch_n[issuer] += 1
                    if value == label:
                        epoch_h[issuer] += 1
            working_h = {issuer: prev_h.get(issuer, 0) + epoch_h[issuer] for issuer in epoch_n}
            working_n = {issuer: prev_n.get(issuer, 0) + epoch_n[issuer] for issuer in epoch_n}
        for issuer, count in epoch_n.items():
            hits[issuer] += epoch_h[issuer]
            counts[issuer] += count


def run_stream(days, mode):
    import run_copy_dated as dated

    book = StreamBook()
    hits = []
    changed = []
    for index, day in enumerate(days):
        claims, _folded = dated.collapse(day["claims"], mode)
        by_slot = defaultdict(list)
        org_of = {}
        seen = set()
        for slot, value, issuer, org in claims:
            if (slot, issuer) in seen:
                continue
            seen.add((slot, issuer))
            by_slot[slot].append((value, issuer, org))
            org_of[slot] = org
        if index:
            previous = days[index - 1]["gold"]
            orgs_today = set(org_of.values())
            use_local = {org: book.prefer_local(org, by_slot) for org in orgs_today}
            for slot, gold in day["gold"].items():
                pairs = by_slot.get(slot, [])
                org = org_of.get(slot)
                if org is None:
                    continue
                pred = book.choose(
                    [(value, issuer) for value, issuer, _org in pairs],
                    org,
                    slot,
                    use_local[org],
                )
                if pred is not None:
                    book.belief[slot] = pred
                hit = int(pred == gold)
                hits.append(hit)
                if slot in previous and previous[slot] != gold:
                    changed.append(hit)
        book.update(claims, None)
        orgs = {org for _slot, _value, _issuer, org in claims}
        for org in orgs:
            book.update(claims, org)
        if index:
            print(mode, index, f"{sum(hits) / len(hits):.4f}", flush=True)
    return {
        "em": sum(hits) / len(hits),
        "changed": sum(changed) / len(changed),
        "n": len(hits),
        "changed_n": len(changed),
        "hits": hits,
    }


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else "probe"
    if command == "probe":
        payload = eval_seeds([600, 601], REGIMES)
    elif command == "dev":
        payload = eval_seeds(DEV, REGIMES)
    elif command == "test":
        payload = eval_seeds(TEST, REGIMES)
    elif command == "stock":
        payload = {
            "subset": run_stream(fusion.stock_days(), "subset"),
            "exact": run_stream(fusion.stock_days(), "exact"),
        }
        for key in payload:
            payload[key].pop("hits", None)
    elif command == "flight":
        payload = {
            "subset": run_stream(fusion.flight_days(), "subset"),
            "exact": run_stream(fusion.flight_days(), "exact"),
        }
        for key in payload:
            payload[key].pop("hits", None)
    else:
        raise SystemExit(command)
    print(json.dumps(payload, indent=2, default=str)[:4000])
    if command in ("dev", "test", "stock", "flight"):
        path = ROOT / "results" / f"evidence_{command}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        print("wrote", path)


if __name__ == "__main__":
    main()
