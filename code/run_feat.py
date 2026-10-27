"""Which observable features separate the record that is right?

Development seeds 800 and 801. Counts are disagreements of the two records
inside one shared memory.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as rx
import run_fair as fair
import run_heldout as held

REGIMES = ("topic", "stationary")
SEEDS = [800, 801]


def chosen(hits, vec):
    best = None
    best_key = None
    for hit in hits:
        w = fair.weight_of(vec, hit["issuer"])
        if w is None or w < 0.55:
            continue
        key = (hit["time"], hit.get("seq", 0))
        if best_key is None or key > best_key:
            best, best_key = hit, key
    return best


def run(regime, seed):
    world = fair.load_world(seed, regime)
    cfg = next(item for item in fair.RULES if item["name"] == "hier")
    names = world.names
    order = list(world.mutable_ids) + list(world.stable_ids) + list(world.derived_ids)
    belief = list(world.init_belief[0])
    mail = []
    society = held.rag_blank()
    n_org = len(rx.ORGS)
    stats = defaultdict(int)
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
        soc_full = held.fit_vector(fresh_rows, pre, 0.5, max(len(pre) - 1, fresh_max))
        org_full = [held.fit_vector(by_org[g], blank, 1.0, hist_max) for g in range(n_org)]
        new_belief = list(belief)
        for slot in order:
            g = slot % n_org
            if slot in epoch_slots:
                kept = [item for item in fresh_rows if item[0] != slot]
                ids = [fresh_max] + [issuer for _s, _v, issuer in kept]
                glob = held.fit_vector(kept, pre, 0.5, max(len(pre) - 1, max(ids)))
            else:
                glob = soc_full
            if slot in org_slots[g]:
                org_vec = held.fit_vector([item for item in by_org[g] if item[0] != slot], blank, 1.0, hist_max)
            else:
                org_vec = org_full[g]
            hits = buckets.get(slot, [])
            schema = fair.schema_value(world, belief, slot)
            soc_v = fair.answer_slot(cfg, hits, belief[slot], held.trust_for(glob, org_vec, cfg, True), schema)
            org_trust = lambda issuer, vec=org_vec: fair.weight_of(vec, issuer)
            org_v = fair.answer_slot(cfg, hits, belief[slot], org_trust, schema)
            truth = world.timeline[epoch_i][slot]
            if soc_v != org_v:
                kind = "soc" if soc_v == truth and org_v != truth else "org" if org_v == truth and soc_v != truth else "none"
                stats[kind] += 1
                op = chosen(hits, org_vec)
                sp = chosen(hits, glob)
                if op is not None:
                    sw = fair.weight_of(glob, op["issuer"])
                    stats[f"{kind}_org_src_soc_clears"] += int(sw is not None and sw >= 0.55)
                    stats[f"{kind}_org_adopts"] += 1
                else:
                    stats[f"{kind}_org_abstain"] += 1
                if sp is not None:
                    ow = fair.weight_of(org_vec, sp["issuer"])
                    stats[f"{kind}_soc_src_org_clears"] += int(ow is not None and ow >= 0.55)
                    stats[f"{kind}_soc_adopts"] += 1
                else:
                    stats[f"{kind}_soc_abstain"] += 1
                stats[f"{kind}_epoch{epoch_i+1}"] += 1
            pick = soc_v if soc_v == truth else org_v
            new_belief[slot] = pick
        belief = new_belief
        society = soc_full
    return stats


def main():
    for regime in REGIMES:
        total = defaultdict(int)
        for seed in SEEDS:
            stats = run(regime, seed)
            for k, v in stats.items():
                total[k] += v
        print(regime, dict(total), flush=True)


if __name__ == "__main__":
    main()
