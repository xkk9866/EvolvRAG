"""Where the society record and the organization record disagree with the truth.

Development seeds 800 and 801 only. Counts are decision events, not scores.
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


def adopt(hits, vec):
    best = None
    best_key = None
    best_w = None
    for hit in hits:
        w = fair.weight_of(vec, hit["issuer"])
        if w is None or w + 1e-9 < 0.55:
            continue
        key = (hit["time"], hit.get("seq", 0))
        if best_key is None or key > best_key:
            best, best_key, best_w = hit, key, w
    if best is None:
        return None, None
    return best["value"], best_w


def run_world(world, regime):
    cfg = next(item for item in fair.RULES if item["name"] == "hier")
    names = world.names
    order = list(world.mutable_ids) + list(world.stable_ids) + list(world.derived_ids)
    belief_s = list(world.init_belief[0])
    belief_o = list(world.init_belief[0])
    mail = []
    society = held.rag_blank()
    n_org = len(rx.ORGS)
    eta = cfg.get("eta", 0.5)
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
        soc_full = held.fit_vector(fresh_rows, pre, eta, max(len(pre) - 1, fresh_max))
        org_full = [held.fit_vector(by_org[g], blank, 1.0, hist_max) for g in range(n_org)]
        for slot in order:
            g = slot % n_org
            kind = "derived" if slot in world.derived_ids else "stable" if slot in world.stable_ids else "changed"
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
            schema_s = fair.schema_value(world, belief_s, slot)
            schema_o = fair.schema_value(world, belief_o, slot)
            soc_v = fair.answer_slot(cfg, hits, belief_s[slot], held.trust_for(glob, org_vec, cfg, True), schema_s)
            org_v = fair.answer_slot(cfg, hits, belief_o[slot], (lambda issuer, vec=org_vec: fair.weight_of(vec, issuer)), schema_o)
            belief_s[slot] = soc_v
            belief_o[slot] = org_v
            truth = world.timeline[epoch_i][slot]
            changed_now = bool(world.history_at[epoch_i][slot]) and truth != "EXPIRED"
            if kind == "stable":
                track = True
            elif changed_now:
                track = True
            else:
                track = False
            if not track:
                continue
            stats["n"] += 1
            sv, sw = adopt(hits, glob)
            ov, ow = adopt(hits, org_vec)
            soc_ok = soc_v == truth
            org_ok = org_v == truth
            if soc_ok and org_ok:
                stats["both"] += 1
            elif soc_ok:
                stats["soc_only"] += 1
                stats[f"soc_only_{kind}"] += 1
            elif org_ok:
                stats["org_only"] += 1
                stats[f"org_only_{kind}"] += 1
            else:
                stats["neither"] += 1
                stats[f"neither_{kind}"] += 1
            if not soc_ok and not org_ok:
                truth_w_s = max((fair.weight_of(glob, h["issuer"]) for h in hits if h["value"] == truth), default=None)
                truth_w_o = max((fair.weight_of(org_vec, h["issuer"]) for h in hits if h["value"] == truth), default=None)
                present = any(h["value"] == truth for h in hits)
                if not present:
                    stats["miss_absent"] += 1
                elif (truth_w_s or 0) < 0.55 and (truth_w_o or 0) < 0.55:
                    stats["miss_below"] += 1
                elif sv == ov:
                    stats["miss_same_latest"] += 1
                else:
                    stats["miss_split"] += 1
            if sv != ov:
                stats["raw_disagree"] += 1
                if sv == truth and ov != truth:
                    stats["raw_soc"] += 1
                elif ov == truth and sv != truth:
                    stats["raw_org"] += 1
                elif sv != truth and ov != truth:
                    stats["raw_neither"] += 1
                else:
                    stats["raw_both_wrong_wait"] += 1
            if sw is not None and ow is not None and sv != ov:
                if ow > sw and ov == truth and sv != truth:
                    stats["weight_would_fix"] += 1
                if ow > sw and sv == truth and ov != truth:
                    stats["weight_would_hurt"] += 1
        society = soc_full
    return stats


def main():
    grand = defaultdict(int)
    for regime in REGIMES:
        acc = defaultdict(int)
        for seed in SEEDS:
            stats = run_world(fair.load_world(seed, regime), regime)
            for k, v in stats.items():
                acc[k] += v
                grand[k] += v
        n = max(1, acc["n"])
        print(
            regime,
            "n", acc["n"],
            "both", round(acc["both"] / n, 3),
            "soc_only", round(acc["soc_only"] / n, 3),
            "org_only", round(acc["org_only"] / n, 3),
            "neither", round(acc["neither"] / n, 3),
            "kinds",
            {k: acc[k] for k in acc if "only_" in k or k.startswith("neither_")},
            "raw",
            {k: acc[k] for k in ("raw_disagree", "raw_soc", "raw_org", "raw_neither", "weight_would_fix", "weight_would_hurt", "miss_absent", "miss_below", "miss_same_latest", "miss_split")},
            flush=True,
        )
    print("ALL", {k: grand[k] for k in ("n", "both", "soc_only", "org_only", "neither")}, flush=True)


if __name__ == "__main__":
    main()
