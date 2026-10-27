"""Print topic disagreements where the society record is right and the org record is wrong."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as rx
import run_fair as fair
import run_heldout as held


def adopt(hits, vec):
    best = None
    best_key = None
    for hit in hits:
        w = fair.weight_of(vec, hit["issuer"])
        if w is None or w < 0.55:
            continue
        key = (hit["time"], hit.get("seq", 0), w)
        if best_key is None or key > best_key:
            best, best_key = hit, key
    if best is None:
        return None
    return best["value"], best["issuer"], round(best_key[2], 3), best["time"]


def main():
    world = fair.load_world(800, "topic")
    cfg = next(item for item in fair.RULES if item["name"] == "hier")
    names = world.names
    order = list(world.mutable_ids) + list(world.stable_ids) + list(world.derived_ids)
    belief = list(world.init_belief[0])
    mail = []
    society = held.rag_blank()
    n_org = len(rx.ORGS)
    shown = 0
    by_org = defaultdict(lambda: [0, 0])
    for epoch_i, epoch_docs in enumerate(world.visible):
        mail.extend(epoch_docs)
        find = fair.copy_find(mail, names)
        usable = fair.collapsed(mail, find)
        fresh_c = fair.collapsed(epoch_docs, find)
        hist_rows, hist_max = held.claims_from(usable, names)
        fresh_rows, fresh_max = held.claims_from(fresh_c, names)
        org_rows = [[] for _ in range(n_org)]
        org_slots = [set() for _ in range(n_org)]
        for slot, value, issuer in hist_rows:
            org_rows[slot % n_org].append((slot, value, issuer))
            org_slots[slot % n_org].add(slot)
        epoch_slots = {slot for slot, _v, _i in fresh_rows}
        buckets = fair.index_hits(usable, names)
        pre = society
        blank = fair.pad(held.rag_blank(), hist_max + 1)
        soc_full = held.fit_vector(fresh_rows, pre, 0.5, max(len(pre) - 1, fresh_max))
        org_full = [held.fit_vector(org_rows[g], blank, 1.0, hist_max) for g in range(n_org)]
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
                org_vec = held.fit_vector([item for item in org_rows[g] if item[0] != slot], blank, 1.0, hist_max)
            else:
                org_vec = org_full[g]
            hits = buckets.get(slot, [])
            schema = fair.schema_value(world, belief, slot)
            soc_v = fair.answer_slot(cfg, hits, belief[slot], held.trust_for(glob, org_vec, cfg, True), schema)
            org_v = fair.answer_slot(cfg, hits, belief[slot], (lambda issuer, vec=org_vec: fair.weight_of(vec, issuer)), schema)
            truth = world.timeline[epoch_i][slot]
            if soc_v != org_v and soc_v == truth and org_v != truth:
                by_org[g][0] += 1
                if shown < 12 and slot in world.mutable_ids:
                    shown += 1
                    print(
                        "epoch", epoch_i + 1,
                        "org", g,
                        "slot", slot,
                        "truth", truth,
                        "soc", soc_v, adopt(hits, glob),
                        "org", org_v, adopt(hits, org_vec),
                        "belief", belief[slot],
                        "schema", schema,
                        flush=True,
                    )
            elif soc_v != org_v and org_v == truth and soc_v != truth:
                by_org[g][1] += 1
            new_belief[slot] = soc_v if soc_v == truth else org_v
        belief = new_belief
        society = soc_full
    print("per org [soc_wins, org_wins]", {k: by_org[k] for k in range(n_org)}, flush=True)


if __name__ == "__main__":
    main()
