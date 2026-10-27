"""Incumbent-prior posterior on the count-switch held-out record.

Inside a clock window, a value's score is the sum of held-out source
probabilities above the uninformed prior. The incumbent, or the entailed
schema, carries pseudo-weight 0.55. Development seeds 800 and 801.
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
PRIOR = 0.40
ANCHOR = 0.55


def posterior(hits, belief, trust, schema, radius):
    usable = [h for h in hits if isinstance(h["issuer"], int) and h["issuer"] >= 0]
    scores = defaultdict(float)
    if usable:
        tmax = max(h["time"] for h in usable)
        best = {}
        for h in usable:
            if radius is not None and h["time"] < tmax - radius:
                continue
            w = fair.weight_of(trust, h["issuer"])
            if w is None or w <= PRIOR + 1e-9:
                continue
            key = (h["time"], h.get("seq", 0))
            prev = best.get(h["issuer"])
            if prev is None or key > prev[0]:
                best[h["issuer"]] = (key, h["value"], w)
        for _issuer, (_key, value, w) in best.items():
            scores[value] += w
    anchor = schema if schema is not None else belief
    if anchor is not None:
        scores[anchor] += ANCHOR
    if not scores:
        return belief
    return max(scores, key=lambda value: (scores[value], value == anchor))


def run_world(world):
    cfg = next(item for item in fair.RULES if item["name"] == "hier")
    modes = ("loo", "bayes1", "bayesAll")
    belief = {key: list(world.init_belief[0]) for key in modes}
    answers = {key: [] for key in modes}
    mail = []
    society = held.rag_blank()
    n_org = len(rx.ORGS)
    order = list(world.mutable_ids) + list(world.stable_ids) + list(world.derived_ids)
    eta = cfg.get("eta", 0.5)
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
        row = {key: list(belief[key]) for key in modes}
        for slot in order:
            g = slot % n_org
            if slot in epoch_slots:
                kept = [item for item in fresh_rows if item[0] != slot]
                ids = [fresh_max] + [issuer for _s, _v, issuer in kept]
                glob = held.fit_vector(kept, pre, eta, max(len(pre) - 1, max(ids)))
            else:
                glob = soc_full
            if slot in org_slots[g]:
                org_vec = held.fit_vector([item for item in by_org[g] if item[0] != slot], blank, 1.0, hist_max)
            else:
                org_vec = org_full[g]
            hits = buckets.get(slot, [])
            trust = held.trust_for(glob, org_vec, cfg, False)
            for key in modes:
                schema = fair.schema_value(world, row[key], slot)
                if key == "loo":
                    row[key][slot] = fair.answer_slot(cfg, hits, row[key][slot], trust, schema)
                else:
                    radius = 1 if key == "bayes1" else None
                    row[key][slot] = posterior(hits, row[key][slot], trust, schema, radius)
        for key in modes:
            belief[key] = row[key]
            answers[key].append([list(row[key]) for _ in range(rx.N_AGENTS)])
        society = soc_full
    out = {}
    for key, grid in answers.items():
        counts = rx.empty_counts()
        rx.consume(counts, world, grid, 0.0)
        out[key] = rx.pack_metrics(counts)
    return out


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
                bits.append(f"{key}:{score:.3f}/{metrics['changed']:.3f}/{metrics['stable']:.3f}/{metrics['derived']:.3f}")
            print(regime, seed, " ".join(bits), flush=True)
    for key, by in totals.items():
        means = {r: sum(v) / len(v) for r, v in by.items()}
        print(key, {r: round(v, 3) for r, v in means.items()}, "ALL", round(sum(means.values()) / len(means), 3), flush=True)


if __name__ == "__main__":
    main()
