"""Choose the record whose cleared sources agree, not the record that clears more.

A record explains an organization when the sources it clears give one value
on that organization's other entities. Development seeds 800 and 801.
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
TAU = 0.55


def agreement(vec, rows_by_slot):
    cleared = {i for i in range(max(len(vec), 1)) if fair.weight_of(vec, i) >= TAU}
    if len(cleared) < 2:
        return 0.0, 0
    total = 0.0
    n = 0
    for rows in rows_by_slot.values():
        vals = [value for value, issuer in rows if issuer in cleared]
        if len(vals) < 2:
            continue
        best = max(vals.count(value) for value in set(vals))
        total += best / len(vals)
        n += 1
    if n == 0:
        return 0.0, 0
    return total / n, n


def run_world(world):
    cfg = next(item for item in fair.RULES if item["name"] == "hier")
    names = world.names
    order = list(world.mutable_ids) + list(world.stable_ids) + list(world.derived_ids)
    modes = ("loo", "agree")
    belief = {key: list(world.init_belief[0]) for key in modes}
    answers = {key: [] for key in modes}
    mail = []
    society = held.rag_blank()
    n_org = len(rx.ORGS)
    eta = cfg.get("eta", 0.5)
    picks = defaultdict(int)
    for epoch_docs in world.visible:
        mail.extend(epoch_docs)
        find = fair.copy_find(mail, names)
        usable = fair.collapsed(mail, find)
        fresh_c = fair.collapsed(epoch_docs, find)
        hist_rows, hist_max = held.claims_from(usable, names)
        fresh_rows, fresh_max = held.claims_from(fresh_c, names)
        by_org = [[] for _ in range(n_org)]
        org_slots = [set() for _ in range(n_org)]
        hist_by_slot = defaultdict(list)
        for slot, value, issuer in hist_rows:
            by_org[slot % n_org].append((slot, value, issuer))
            org_slots[slot % n_org].add(slot)
            hist_by_slot[slot].append((value, issuer))
        epoch_slots = {slot for slot, _v, _i in fresh_rows}
        buckets = fair.index_hits(usable, names)
        pre = society
        blank = fair.pad(held.rag_blank(), hist_max + 1)
        soc_full = held.fit_vector(fresh_rows, pre, eta, max(len(pre) - 1, fresh_max))
        org_full = [held.fit_vector(by_org[g], blank, 1.0, hist_max) for g in range(n_org)]
        glob_of, orgv_of = {}, {}
        for slot in order:
            g = slot % n_org
            if slot in epoch_slots:
                kept = [item for item in fresh_rows if item[0] != slot]
                ids = [fresh_max] + [issuer for _s, _v, issuer in kept]
                glob_of[slot] = held.fit_vector(kept, pre, eta, max(len(pre) - 1, max(ids)))
            else:
                glob_of[slot] = soc_full
            if slot in org_slots[g]:
                orgv_of[slot] = held.fit_vector([item for item in by_org[g] if item[0] != slot], blank, 1.0, hist_max)
            else:
                orgv_of[slot] = org_full[g]
        row = {key: list(belief[key]) for key in modes}
        for slot in order:
            g = slot % n_org
            hits = buckets.get(slot, [])
            glob, org_vec = glob_of[slot], orgv_of[slot]
            others = {s: rows for s, rows in hist_by_slot.items() if s != slot and s % n_org == g}
            schema = {}
            for key in modes:
                schema[key] = fair.schema_value(world, row[key], slot)
                if key == "loo":
                    trust = held.trust_for(glob, org_vec, cfg, False)
                else:
                    soc_a, soc_n = agreement(glob, others)
                    org_a, org_n = agreement(org_vec, others)
                    use_org = org_n >= 2 and org_a > soc_a + 1e-12
                    picks["org" if use_org else "soc"] += 1
                    trust = (lambda issuer, vec=org_vec: fair.weight_of(vec, issuer)) if use_org else held.trust_for(glob, org_vec, cfg, True)
                row[key][slot] = fair.answer_slot(cfg, hits, row[key][slot], trust, schema[key])
        for key in modes:
            belief[key] = row[key]
            answers[key].append([list(row[key]) for _ in range(rx.N_AGENTS)])
        society = soc_full
    packed = {}
    for key, grid in answers.items():
        counts = rx.empty_counts()
        rx.consume(counts, world, grid, 0.0)
        packed[key] = rx.pack_metrics(counts)
    packed["_picks"] = dict(picks)
    return packed


def balanced(m):
    return (m["changed"] + m["stable"] + m["derived"]) / 3.0


def main():
    totals = defaultdict(lambda: defaultdict(list))
    for regime in REGIMES:
        picks = defaultdict(int)
        for seed in SEEDS:
            packed = run_world(fair.load_world(seed, regime))
            for k, v in packed["_picks"].items():
                picks[k] += v
            bits = []
            for key in ("loo", "agree"):
                score = balanced(packed[key])
                totals[key][regime].append(score)
                bits.append(f"{key}:{score:.3f}/{packed[key]['sp']:.3f}")
            print(regime, seed, " ".join(bits), flush=True)
        print(regime, "org_share", round(picks["org"] / max(1, picks["org"] + picks["soc"]), 3), flush=True)
    for key, by in totals.items():
        means = {r: sum(v) / len(v) for r, v in by.items()}
        print(key, {r: round(v, 3) for r, v in means.items()}, "ALL", round(sum(means.values()) / len(means), 3), flush=True)


if __name__ == "__main__":
    main()
