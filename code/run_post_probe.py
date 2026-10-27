"""Probe revision posteriors on the same held-out records as the count switch.

Development-only. Seeds 800 and 801. The script compares the locked count
switch with a categorical posterior that uses those records as source
probabilities, so a gap here is a gap in the revision distribution rather
than in the reliability fit.
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as rx
import run_fair as fair
import run_heldout as held

REGIMES = ("stationary", "topic", "open_rumor", "sybil")
SEEDS = [800, 801]
CLIP = 1e-3


def loglik(groups, trust, cand):
    """One latest claim per issuer. Uniform noise over the other candidates."""
    wrong = max(1, len(cand) - 1)
    scores = {value: 0.0 for value in cand}
    for value, issuer in groups:
        w = fair.weight_of(trust, issuer)
        w = min(1.0 - CLIP, max(CLIP, w if w is not None else rx.PRIOR))
        hit = math.log(w)
        miss = math.log((1.0 - w) / wrong)
        for hyp in cand:
            scores[hyp] += hit if hyp == value else miss
    return scores


def latest_groups(hits, radius):
    usable = [h for h in hits if isinstance(h["issuer"], int) and h["issuer"] >= 0]
    if not usable:
        return [], None
    tmax = max(h["time"] for h in usable)
    best = {}
    for h in usable:
        if radius is not None and h["time"] < tmax - radius:
            continue
        key = (h["time"], h.get("seq", 0))
        prev = best.get(h["issuer"])
        if prev is None or key > prev[0]:
            best[h["issuer"]] = (key, h["value"])
    return [(value, issuer) for issuer, (_key, value) in best.items()], tmax


def mode_of(scores, belief):
    return max(scores, key=lambda value: (scores[value], value == belief))


def posterior(hits, belief, trust, schema_v, radius, persist):
    groups, _tmax = latest_groups(hits, radius)
    cand = {value for value, _issuer in groups}
    if belief is not None:
        cand.add(belief)
    if schema_v is not None:
        cand.add(schema_v)
    if not cand:
        return belief
    scores = loglik(groups, trust, cand)
    if persist > 0.0 and belief in scores:
        scores[belief] += math.log(persist)
        others = [value for value in scores if value != belief]
        if others:
            share = math.log(max(CLIP, 1.0 - persist) / len(others))
            for value in others:
                scores[value] += share
    if schema_v is not None and schema_v not in {value for value, _i in groups} and schema_v in scores:
        # schema is a deterministic child, not a witness; give it the prior mass of one pseudo-claim
        pass
    return mode_of(scores, belief)


def run_world(world):
    cfg = next(item for item in fair.RULES if item["name"] == "hier")
    names = world.names
    order = list(world.mutable_ids) + list(world.stable_ids) + list(world.derived_ids)
    modes = (
        "loo",
        "soc",
        "org",
        "post0",
        "post1",
        "post1p",
        "oracle",
    )
    belief = {key: list(world.init_belief[0]) for key in modes}
    answers = {key: [] for key in modes}
    mail = []
    society = held.rag_blank()
    n_org = len(rx.ORGS)
    eta = cfg.get("eta", 0.5)
    for epoch_i, epoch_docs in enumerate(world.visible):
        mail.extend(epoch_docs)
        find = fair.copy_find(mail, names)
        usable = fair.collapsed(mail, find)
        fresh_c = fair.collapsed(epoch_docs, find)
        hist_rows, hist_max = held.claims_from(usable, names)
        fresh_rows, fresh_max = held.claims_from(fresh_c, names)
        by_org = [[] for _ in range(n_org)]
        org_slots = [set() for _ in range(n_org)]
        for slot, value, issuer in hist_rows:
            by_org[slot % n_org].append((slot, value, issuer))
            org_slots[slot % n_org].add(slot)
        epoch_slots = {slot for slot, _v, _i in fresh_rows}
        buckets = fair.index_hits(usable, names)
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
                kept = [item for item in by_org[g] if item[0] != slot]
                org_vec = held.fit_vector(kept, blank, 1.0, hist_max)
            else:
                org_vec = org_full[g]
            hits = buckets.get(slot, [])
            truth = world.timeline[epoch_i][slot]
            for key in modes:
                schema_v = fair.schema_value(world, row[key], slot)
                if key == "loo":
                    trust = held.trust_for(glob, org_vec, cfg, False)
                    row[key][slot] = fair.answer_slot(cfg, hits, row[key][slot], trust, schema_v)
                elif key == "soc":
                    row[key][slot] = fair.answer_slot(cfg, hits, row[key][slot], held.trust_for(glob, org_vec, cfg, True), schema_v)
                elif key == "org":
                    row[key][slot] = fair.answer_slot(cfg, hits, row[key][slot], (lambda issuer, vec=org_vec: fair.weight_of(vec, issuer)), schema_v)
                elif key == "post0":
                    trust = held.trust_for(glob, org_vec, cfg, False)
                    row[key][slot] = posterior(hits, row[key][slot], trust, schema_v, 0, 0.5)
                elif key == "post1":
                    trust = held.trust_for(glob, org_vec, cfg, False)
                    row[key][slot] = posterior(hits, row[key][slot], trust, schema_v, 1, 0.0)
                elif key == "post1p":
                    trust = held.trust_for(glob, org_vec, cfg, False)
                    row[key][slot] = posterior(hits, row[key][slot], trust, schema_v, 1, 0.84)
                elif key == "oracle":
                    a = fair.answer_slot(cfg, hits, row[key][slot], held.trust_for(glob, org_vec, cfg, True), schema_v)
                    b = fair.answer_slot(cfg, hits, row[key][slot], (lambda issuer, vec=org_vec: fair.weight_of(vec, issuer)), schema_v)
                    row[key][slot] = a if a == truth else b
        for key in modes:
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
                bits.append(f"{key}:{score:.3f}/{metrics['changed']:.3f}/{metrics['stable']:.3f}/{metrics['derived']:.3f}")
            print(regime, seed, " ".join(bits), flush=True)
    print("--- mean balanced ---", flush=True)
    for key, by in totals.items():
        means = {r: sum(v) / len(v) for r, v in by.items()}
        print(key, {r: round(v, 3) for r, v in means.items()}, "ALL", round(sum(means.values()) / len(means), 3), flush=True)


if __name__ == "__main__":
    main()
