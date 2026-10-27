"""Organization competence from contemporaneous testimony.

A static fit treats every past value of an entity as one label, so a source
that tracked a real change is scored as inconsistent. The temporal record
fits the current round inside the organization, holds out the entity being
revised, and mixes that fit into the organization's previous record with the
same rate used for the society record. The count switch, threshold, copy
collapse, and schema are unchanged. Development seeds 800 and 801.
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
    modes = ("loo", "temp", "temp_only")
    belief = {key: list(world.init_belief[0]) for key in modes}
    answers = {key: [] for key in modes}
    mail = []
    society = held.rag_blank()
    n_org = len(rx.ORGS)
    org_state = [held.rag_blank() for _ in range(n_org)]
    eta = cfg.get("eta", 0.5)
    recent = []
    for epoch_docs in world.visible:
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
        fresh_org = [[] for _ in range(n_org)]
        for slot, value, issuer in fresh_rows:
            fresh_org[slot % n_org].append((slot, value, issuer))
        recent.append(fresh_rows)
        window_rows = recent[-2:]
        window_org = [[] for _ in range(n_org)]
        for rows in window_rows:
            for slot, value, issuer in rows:
                window_org[slot % n_org].append((slot, value, issuer))
        epoch_slots = {slot for slot, _v, _i in fresh_rows}
        buckets = fair.index_hits(usable, names)
        pre = society
        blank = fair.pad(held.rag_blank(), hist_max + 1)
        soc_full = held.fit_vector(fresh_rows, pre, eta, max(len(pre) - 1, fresh_max))
        org_full = [held.fit_vector(by_org[g], blank, 1.0, hist_max) for g in range(n_org)]
        org_now = [
            held.fit_vector(fresh_org[g], org_state[g], eta, max(len(org_state[g]) - 1, fresh_max))
            for g in range(n_org)
        ]
        row = {key: list(belief[key]) for key in modes}
        for slot in order:
            g = slot % n_org
            if slot in epoch_slots:
                kept = [item for item in fresh_rows if item[0] != slot]
                ids = [fresh_max] + [issuer for _s, _v, issuer in kept]
                glob = held.fit_vector(kept, pre, eta, max(len(pre) - 1, max(ids)))
                kept_g = [item for item in fresh_org[g] if item[0] != slot]
                local = held.fit_vector(kept_g, org_state[g], eta, max(len(org_state[g]) - 1, fresh_max, hist_max))
            else:
                glob = soc_full
                local = org_now[g]
            if slot in org_slots[g]:
                org_vec = held.fit_vector([item for item in by_org[g] if item[0] != slot], blank, 1.0, hist_max)
            else:
                org_vec = org_full[g]
            hits = buckets.get(slot, [])
            for key in modes:
                schema = fair.schema_value(world, row[key], slot)
                if key == "loo":
                    trust = held.trust_for(glob, org_vec, cfg, False)
                elif key == "temp":
                    trust = held.trust_for(glob, local, cfg, False)
                else:
                    trust = held.trust_for(glob, local, cfg, True)
                    trust = (lambda issuer, vec=local: fair.weight_of(vec, issuer))
                row[key][slot] = fair.answer_slot(cfg, hits, row[key][slot], trust, schema)
        for key in modes:
            belief[key] = row[key]
            answers[key].append([list(row[key]) for _ in range(rx.N_AGENTS)])
        society = soc_full
        org_state = org_now
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
