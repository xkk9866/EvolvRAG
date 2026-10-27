"""Why the abstention fallback breaks the sybil world on seed 800."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as rx
import run_fair as fair
import run_heldout as held


def analyze(regime):
    world = fair.load_world(800, regime)
    cfg = next(item for item in fair.RULES if item["name"] == "hier")
    names = world.names
    order = list(world.mutable_ids) + list(world.stable_ids) + list(world.derived_ids)
    belief = list(world.init_belief[0])
    mail = []
    society = held.rag_blank()
    n_org = len(rx.ORGS)
    eta = 0.5
    bins = defaultdict(int)
    for epoch_i, epoch_docs in enumerate(world.visible):
        mail.extend(epoch_docs)
        find = fair.copy_find(mail, names)
        usable = fair.collapsed(mail, find)
        fresh_c = fair.collapsed(epoch_docs, find)
        hist_rows, hist_max = held.claims_from(usable, names)
        fresh_rows, fresh_max = held.claims_from(fresh_c, names)
        n_clone = sum(1 for d in usable if isinstance(d["issuer"], int) and d["issuer"] >= rx.N_REL + rx.N_NOISY)
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
            schema = fair.schema_value(world, belief, slot)
            loo_v = fair.answer_slot(cfg, hits, belief[slot], trust, schema)
            weights = []
            for h in hits:
                w = fair.weight_of(trust, h["issuer"])
                if w is not None:
                    weights.append((w, h["issuer"], h["value"], h["time"]))
            cleared = [item for item in weights if item[0] >= 0.55]
            truth = world.timeline[epoch_i][slot]
            if not cleared:
                above = [item for item in weights if item[0] > 0.40 + 1e-9]
                pick = max(above) if above else None
                fallback = pick[2] if pick is not None else (schema if schema is not None else belief[slot])
                bins["abstain"] += 1
                bins["loo_ok"] += int(loo_v == truth)
                bins["fb_ok"] += int(fallback == truth)
                if pick is not None:
                    bins["has_mid"] += 1
                    bins["mid_is_truth"] += int(pick[2] == truth)
                    bins["mid_issuer_ge10"] += int(pick[1] >= 10)
                    band = round(pick[0], 2)
                    bins[f"w{band:.2f}"] += 1
            belief[slot] = loo_v
        society = soc_full
        print(regime, "epoch", epoch_i + 1, "collapsed_clone_docs", n_clone, "mail", len(mail), flush=True)
    print(regime, {k: bins[k] for k in sorted(bins)}, flush=True)


if __name__ == "__main__":
    analyze("stationary")
    analyze("sybil")
