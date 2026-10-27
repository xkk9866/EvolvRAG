"""Newest corroborated cohort.

Source records stay the published agreement fit. A copied issuer is one
observation. Sources that are not better than the clearance already used by
that fit do not revise the belief. Among the sources that do, only the
newest time is current, and the value is the one named by the most of those
sources. A later sequence number is not a second vote. Derived facts with
no current cohort follow the parent. Development seeds are scored before
any test seed. This file does not rewrite hier.json or fusion.json.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import run_experiment as rx
import run_fair as fair
import run_fusion as fusion

DEV = list(range(600, 616))
REGIMES = ("stationary", "topic", "open_rumor", "sybil")


def cohort_value(hits, belief, trust, cutoff):
    eligible = []
    for hit in hits:
        weight = fair.weight_of(trust, hit["issuer"])
        if weight is None or weight + 1e-9 < cutoff:
            continue
        eligible.append((hit, weight))
    if not eligible:
        return belief
    newest = max(hit["time"] for hit, _weight in eligible)
    buckets = defaultdict(lambda: [0, 0.0])
    for hit, weight in eligible:
        if hit["time"] != newest:
            continue
        buckets[hit["value"]][0] += 1
        buckets[hit["value"]][1] += weight
    return max(buckets, key=lambda value: (buckets[value][0], buckets[value][1], value))


def run_world(world, cfg):
    names = world.names
    belief = list(world.init_belief[0])
    book = fair.TrustBook(cfg)
    mail = []
    answers = []
    order = list(world.mutable_ids) + list(world.stable_ids) + list(world.derived_ids)
    for epoch_docs in world.visible:
        mail.extend(epoch_docs)
        find = fair.copy_find(mail, names) if cfg.get("copy") else None
        usable = fair.collapsed(mail, find)
        fresh = fair.collapsed(epoch_docs, find)
        buckets = fair.index_hits(usable, names)
        row = list(belief)
        for slot in order:
            schema_v = fair.schema_value(world, row, slot) if cfg.get("schema") else None
            hits = buckets.get(slot, [])
            trust = book.for_slot(slot)
            chosen = cohort_value(hits, row[slot], trust, cfg.get("cutoff", 0.55))
            saw = chosen != row[slot] or any(
                fair.weight_of(trust, hit["issuer"]) is not None
                and fair.weight_of(trust, hit["issuer"]) + 1e-9 >= cfg.get("cutoff", 0.55)
                for hit in hits
            )
            if cfg.get("schema") and schema_v is not None and not any(
                (weight := fair.weight_of(trust, hit["issuer"])) is not None
                and weight + 1e-9 >= cfg.get("cutoff", 0.55)
                for hit in hits
            ):
                row[slot] = schema_v
            else:
                row[slot] = chosen
            del saw
        belief = row
        answers.append([list(row) for _ in range(rx.N_AGENTS)])
        book.update(fresh, usable, names)
    counts = rx.empty_counts()
    rx.consume(counts, world, answers, 0.0)
    return rx.pack_metrics(counts)


def balanced(metrics):
    return (metrics["changed"] + metrics["stable"] + metrics["derived"]) / 3.0


def eval_seeds(seeds):
    cfg = fair.RULES[4]
    out = {}
    for regime in REGIMES:
        rows = []
        base = []
        for seed in seeds:
            world = fair.load_world(seed, regime)
            metrics = run_world(world, cfg)
            other = fair.run_cfg(world, cfg)
            rows.append(metrics)
            base.append(other)
            print(
                regime,
                seed,
                f"cohort {balanced(metrics):.3f}",
                f"hier {balanced(other):.3f}",
                flush=True,
            )
        cell = fair.cell_of(rows)
        base_cell = fair.cell_of(base)
        out[regime] = {
            "cohort": balanced(cell),
            "hier": balanced(base_cell),
            "cohort_sp": cell["sp"],
            "hier_sp": base_cell["sp"],
            "cohort_changed": cell["changed"],
            "cohort_stable": cell["stable"],
            "cohort_derived": cell["derived"],
        }
    out["mean"] = {
        "cohort": sum(out[regime]["cohort"] for regime in REGIMES) / len(REGIMES),
        "hier": sum(out[regime]["hier"] for regime in REGIMES) / len(REGIMES),
    }
    return out


def fusion_cohort(days, mode):
    import run_copy_dated as dated

    book = dated.Slim()
    hits = []
    changed_hits = []
    for index, day in enumerate(days):
        claims, _folded = dated.collapse(day["claims"], mode)
        if index == 0:
            book.update(claims)
            continue
        by_slot = defaultdict(list)
        org_of = {}
        for slot, value, issuer, org in claims:
            by_slot[slot].append((value, issuer))
            org_of[slot] = org
        previous = days[index - 1]["gold"]
        for slot, gold in day["gold"].items():
            pairs = by_slot.get(slot, [])
            org = org_of.get(slot)
            weights = book.weights(org, "society") if org is not None else book.soc
            buckets = defaultdict(lambda: [0, 0.0])
            for value, issuer in pairs:
                weight = weights.get(issuer, fusion.PRIOR)
                if weight + 1e-9 < fusion.TAU:
                    continue
                buckets[value][0] += 1
                buckets[value][1] += weight
            if buckets:
                pred = max(buckets, key=lambda value: (buckets[value][0], buckets[value][1], value))
            else:
                pred = book.belief["society"].get(slot)
            if pred is not None:
                book.belief["society"][slot] = pred
            hit = int(pred == gold)
            hits.append(hit)
            if slot in previous and previous[slot] != gold:
                changed_hits.append(hit)
        book.update(claims)
        print(mode, index, f"{sum(hits) / len(hits):.4f}", flush=True)
    return {
        "em": sum(hits) / len(hits),
        "changed": sum(changed_hits) / len(changed_hits),
        "n": len(hits),
        "changed_n": len(changed_hits),
    }


def main():
    command = sys.argv[1]
    if command == "probe":
        payload = eval_seeds([600, 601])
    elif command == "dev":
        payload = eval_seeds(DEV)
        path = ROOT / "results" / "cohort_dev.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        print("wrote", path)
    elif command == "stock":
        days = fusion.stock_days()
        payload = {"subset": fusion_cohort(days, "subset"), "exact": fusion_cohort(days, "exact")}
        path = ROOT / "results" / "cohort_stock.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        print("wrote", path)
    elif command == "flight":
        days = fusion.flight_days()
        payload = {"subset": fusion_cohort(days, "subset"), "exact": fusion_cohort(days, "exact")}
        path = ROOT / "results" / "cohort_flight.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        print("wrote", path)
    else:
        raise SystemExit(command)
    print(json.dumps(payload, indent=2)[:3000])


if __name__ == "__main__":
    main()
