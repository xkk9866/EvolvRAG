"""Reliability is contemporaneous accuracy, pooled across rounds.

Each round has its own consensus. A source is scored by whether it matched
that round's consensus, and only on entities other than the one being revised.
Accuracy is shrunk toward the prior with the same four pseudo-claims the CRH
fit already uses. The count switch then chooses the organization pool or the
society pool. Development seeds 800 and 801.
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
ALPHA = 4.0


def round_truth(kept):
    """Unique plurality. A tie is not a consensus, so it does not train anyone."""
    seen = {}
    for slot, value, issuer in kept:
        seen[(slot, issuer)] = value
    buckets = defaultdict(lambda: defaultdict(set))
    for (slot, issuer), value in seen.items():
        buckets[slot][value].add(issuer)
    truth = {}
    for slot, vals in buckets.items():
        ranked = sorted((len(members), value) for value, members in vals.items())
        if len(ranked) == 1 or ranked[-1][0] > ranked[-2][0]:
            truth[slot] = ranked[-1][1]
    return truth, seen


def accuracy_vector(epoch_rows, max_id, exclude):
    correct = defaultdict(float)
    total = defaultdict(int)
    for rows in epoch_rows:
        kept = [row for row in rows if row[0] != exclude]
        if len(kept) < 2:
            continue
        truth, seen = round_truth(kept)
        for (slot, issuer), value in seen.items():
            if slot not in truth:
                continue
            total[issuer] += 1
            if truth[slot] == value:
                correct[issuer] += 1.0
    weights = [rx.PRIOR for _ in range(max_id + 1)]
    for issuer, n in total.items():
        if issuer < 0 or issuer > max_id:
            continue
        raw = correct[issuer] / n
        weights[issuer] = (rx.PRIOR * ALPHA + raw * n) / (ALPHA + n)
    return weights


def run_world(world):
    cfg = next(item for item in fair.RULES if item["name"] == "hier")
    names = world.names
    order = list(world.mutable_ids) + list(world.stable_ids) + list(world.derived_ids)
    modes = ("loo", "acc", "acc_only")
    belief = {key: list(world.init_belief[0]) for key in modes}
    answers = {key: [] for key in modes}
    mail = []
    society = held.rag_blank()
    n_org = len(rx.ORGS)
    eta = cfg.get("eta", 0.5)
    epoch_all = []
    epoch_org = []
    for epoch_docs in world.visible:
        mail.extend(epoch_docs)
        find = fair.copy_find(mail, names)
        usable = fair.collapsed(mail, find)
        fresh_c = fair.collapsed(epoch_docs, find)
        hist_rows, hist_max = held.claims_from(usable, names)
        fresh_rows, fresh_max = held.claims_from(fresh_c, names)
        epoch_all.append(fresh_rows)
        by_g = [[] for _ in range(n_org)]
        for slot, value, issuer in fresh_rows:
            by_g[slot % n_org].append((slot, value, issuer))
        epoch_org.append(by_g)
        by_org = [[] for _ in range(n_org)]
        org_slots = [set() for _ in range(n_org)]
        for slot, value, issuer in hist_rows:
            by_org[slot % n_org].append((slot, value, issuer))
            org_slots[slot % n_org].add(slot)
        epoch_slots = {slot for slot, _v, _i in fresh_rows}
        buckets = fair.index_hits(usable, names)
        pre = society
        blank = fair.pad(held.rag_blank(), hist_max + 1)
        width = max(len(pre) - 1, fresh_max, hist_max)
        soc_full = held.fit_vector(fresh_rows, pre, eta, width)
        org_full = [held.fit_vector(by_org[g], blank, 1.0, hist_max) for g in range(n_org)]
        row = {key: list(belief[key]) for key in modes}
        org_epochs = [[epoch_org[t][g] for t in range(len(epoch_org))] for g in range(n_org)]
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
            local = accuracy_vector(org_epochs[g], width, slot)
            hits = buckets.get(slot, [])
            for key in modes:
                schema = fair.schema_value(world, row[key], slot)
                if key == "loo":
                    trust = held.trust_for(glob, org_vec, cfg, False)
                elif key == "acc":
                    trust = held.trust_for(glob, local, cfg, False)
                else:
                    trust = lambda issuer, vec=local: fair.weight_of(vec, issuer)
                row[key][slot] = fair.answer_slot(cfg, hits, row[key][slot], trust, schema)
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
                bits.append(f"{key}:{score:.3f}/{metrics['sp']:.3f}")
            print(regime, seed, " ".join(bits), flush=True)
    for key, by in totals.items():
        means = {r: sum(v) / len(v) for r, v in by.items()}
        print(key, {r: round(v, 3) for r, v in means.items()}, "ALL", round(sum(means.values()) / len(means), 3), flush=True)


if __name__ == "__main__":
    main()
