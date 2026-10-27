"""Ceiling for choosing a record inside one shared memory.

The choice at each entity may use the truth. That is an upper bound, not a method.
Development seeds 800 and 801.
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


def run_world(world):
    cfg = next(item for item in fair.RULES if item["name"] == "hier")
    names = world.names
    order = list(world.mutable_ids) + list(world.stable_ids) + list(world.derived_ids)
    modes = ("loo", "soc", "org", "step")
    belief = {key: list(world.init_belief[0]) for key in modes}
    answers = {key: [] for key in modes}
    mail = []
    society = held.rag_blank()
    n_org = len(rx.ORGS)
    eta = cfg.get("eta", 0.5)
    disagree = defaultdict(int)
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
                org_vec = held.fit_vector([item for item in by_org[g] if item[0] != slot], blank, 1.0, hist_max)
            else:
                org_vec = org_full[g]
            hits = buckets.get(slot, [])
            truth = world.timeline[epoch_i][slot]
            for key, society_only, vec in (
                ("loo", False, None),
                ("soc", True, glob),
                ("org", True, org_vec),
            ):
                if key == "loo":
                    trust = held.trust_for(glob, org_vec, cfg, False)
                else:
                    trust = (lambda issuer, vec=vec: fair.weight_of(vec, issuer)) if key == "org" else held.trust_for(glob, org_vec, cfg, True)
                schema = fair.schema_value(world, row[key], slot)
                row[key][slot] = fair.answer_slot(cfg, hits, row[key][slot], trust, schema)
            # one shared belief: the step-oracle memory
            schema = fair.schema_value(world, row["step"], slot)
            soc_v = fair.answer_slot(cfg, hits, row["step"][slot], held.trust_for(glob, org_vec, cfg, True), schema)
            org_v = fair.answer_slot(cfg, hits, row["step"][slot], (lambda issuer, vec=org_vec: fair.weight_of(vec, issuer)), schema)
            if soc_v != org_v:
                disagree["n"] += 1
                disagree["soc"] += int(soc_v == truth)
                disagree["org"] += int(org_v == truth and soc_v != truth)
                disagree["neither"] += int(soc_v != truth and org_v != truth)
            row["step"][slot] = soc_v if soc_v == truth else org_v
        for key in modes:
            belief[key] = row[key]
            answers[key].append([list(row[key]) for _ in range(rx.N_AGENTS)])
        society = soc_full
    packed = {}
    for key, grid in answers.items():
        counts = rx.empty_counts()
        rx.consume(counts, world, grid, 0.0)
        packed[key] = rx.pack_metrics(counts)
    packed["_d"] = dict(disagree)
    return packed


def balanced(m):
    return (m["changed"] + m["stable"] + m["derived"]) / 3.0


def main():
    totals = defaultdict(lambda: defaultdict(list))
    disc = defaultdict(int)
    for regime in REGIMES:
        for seed in SEEDS:
            packed = run_world(fair.load_world(seed, regime))
            for k, v in packed["_d"].items():
                disc[f"{regime}_{k}"] += v
            bits = []
            for key in ("loo", "soc", "org", "step"):
                score = balanced(packed[key])
                totals[key][regime].append(score)
                bits.append(f"{key}:{score:.3f}")
            print(regime, seed, " ".join(bits), packed["_d"], flush=True)
    for key, by in totals.items():
        means = {r: sum(v) / len(v) for r, v in by.items()}
        print(key, {r: round(v, 3) for r, v in means.items()}, "ALL", round(sum(means.values()) / len(means), 3), flush=True)
    print("disagree", dict(disc), flush=True)


if __name__ == "__main__":
    main()
