"""Does a single-passage posterior beat latest-cleared on the same records?

Development seeds 800 and 801. Every mode uses the count-switch held-out
record. The only change is which passage or value that record supports.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as rx
import run_fair as fair
import run_heldout as held

REGIMES = ("stationary", "topic", "open_rumor", "sybil")
SEEDS = [800, 801]


def window_best(hits, trust, radius, floor, how):
    usable = [h for h in hits if isinstance(h["issuer"], int) and h["issuer"] >= 0]
    if not usable:
        return None, 0.0
    tmax = max(h["time"] for h in usable)
    scores = defaultdict(float)
    best = None
    for h in usable:
        if radius is not None and h["time"] < tmax - radius:
            continue
        w = fair.weight_of(trust, h["issuer"])
        if w is None or w < floor:
            continue
        scores[h["value"]] += w
        key = (w, h["time"], h.get("seq", 0))
        if best is None or key > best[0]:
            best = (key, h["value"], w)
    if how == "issuer":
        if best is None:
            return None, 0.0
        return best[1], best[2]
    if not scores:
        return None, 0.0
    value = max(scores, key=lambda v: scores[v])
    return value, scores[value]


def choose(hits, belief, trust, schema, radius, floor, how):
    cleared, _mass = window_best(hits, trust, None, 0.55, "issuer")
    if how == "abstain":
        if cleared is not None:
            return fair.answer_slot(
                {"rule": "hier", "cutoff": 0.55, "schema": True, "margin": 1},
                hits,
                belief,
                trust,
                schema,
            )
        value, _mass = window_best(hits, trust, radius, floor, "issuer")
        if value is None:
            return schema if schema is not None else belief
        return value
    if how == "cleared_max":
        value, _mass = window_best(hits, trust, None, 0.55, "issuer")
        if value is None:
            value, _mass = window_best(hits, trust, radius, floor, "issuer")
        if value is None:
            return schema if schema is not None else belief
        return value
    value, _mass = window_best(hits, trust, radius, floor, how)
    if value is None:
        return schema if schema is not None else belief
    return value


def run_world(world):
    cfg = next(item for item in fair.RULES if item["name"] == "hier")
    specs = {
        "loo": None,
        "abs1": (1, 0.40 + 1e-6, "abstain"),
        "absAll": (None, 0.40 + 1e-6, "abstain"),
        "abs50": (1, 0.50, "abstain"),
        "cmax": (1, 0.40 + 1e-6, "cleared_max"),
    }
    belief = {key: list(world.init_belief[0]) for key in specs}
    answers = {key: [] for key in specs}
    mail = []
    society = held.rag_blank()
    n_org = len(rx.ORGS)
    eta = cfg.get("eta", 0.5)
    order = list(world.mutable_ids) + list(world.stable_ids) + list(world.derived_ids)
    for epoch_docs in world.visible:
        mail.extend(epoch_docs)
        find = fair.copy_find(mail, names) if False else fair.copy_find(mail, world.names)
        usable = fair.collapsed(mail, find)
        fresh_c = fair.collapsed(epoch_docs, find)
        hist_rows, hist_max = held.claims_from(usable, world.names)
        fresh_rows, fresh_max = held.claims_from(fresh_c, world.names)
        by_org = [[] for _ in range(n_org)]
        org_slots = [set() for _ in range(n_org)]
        for slot, value, issuer in hist_rows:
            by_org[slot % n_org].append((slot, value, issuer))
            org_slots[slot % n_org].add(slot)
        epoch_slots = {slot for slot, _v, _i in fresh_rows}
        buckets = fair.index_hits(usable, world.names)
        pre = society
        blank = fair.pad(held.rag_blank(), hist_max + 1)
        soc_full = held.fit_vector(fresh_rows, pre, eta, max(len(pre) - 1, fresh_max))
        org_full = [held.fit_vector(by_org[g], blank, 1.0, hist_max) for g in range(n_org)]
        row = {key: list(item) for key, item in belief.items()}
        for slot in order:
            g = slot % n_org
            if slot in epoch_slots:
                kept = [item for item in fresh_rows if item[0] != slot]
                ids = [fresh_max] + [issuer for _s, _v, issuer in kept]
                glob = held.fit_vector(kept, pre, eta, max(len(pre) - 1, max(ids)))
            else:
                glob = soc_full
            if slot in org_slots[g]:
                kept_org = [item for item in by_org[g] if item[0] != slot]
                org_vec = held.fit_vector(kept_org, blank, 1.0, hist_max)
            else:
                org_vec = org_full[g]
            hits = buckets.get(slot, [])
            trust = held.trust_for(glob, org_vec, cfg, False)
            for key, spec in specs.items():
                schema = fair.schema_value(world, row[key], slot)
                if spec is None:
                    row[key][slot] = fair.answer_slot(cfg, hits, row[key][slot], trust, schema)
                else:
                    row[key][slot] = choose(hits, row[key][slot], trust, schema, *spec)
        for key in specs:
            belief[key] = row[key]
            answers[key].append([list(row[key]) for _ in range(rx.N_AGENTS)])
        society = soc_full
    packed = {}
    for key, grid in answers.items():
        counts = rx.empty_counts()
        rx.consume(counts, world, grid, 0.0)
        packed[key] = rx.pack_metrics(counts)
    return packed


def balanced(m):
    return (m["changed"] + m["stable"] + m["derived"]) / 3.0


def main():
    totals = defaultdict(lambda: defaultdict(list))
    for regime in REGIMES:
        for seed in SEEDS:
            packed = run_world(fair.load_world(seed, regime))
            bits = []
            for key, metrics in packed.items():
                score = balanced(metrics)
                totals[key][regime].append(score)
                bits.append(
                    f"{key}:{score:.3f}/{metrics['changed']:.3f}/{metrics['stable']:.3f}/{metrics['derived']:.3f}"
                )
            print(regime, seed, " ".join(bits), flush=True)
    print("---", flush=True)
    for key, by in totals.items():
        means = {r: sum(v) / len(v) for r, v in by.items()}
        print(key, {r: round(v, 3) for r, v in means.items()}, "ALL", round(sum(means.values()) / len(means), 3), flush=True)


if __name__ == "__main__":
    main()
