"""Clearance from the perfectly consistent component of the agreement graph.

Power iteration ranks sources by agreement on entities other than the one
being revised. A component is usable when those sources never disagreed on a
shared entity. Clearance weights are then the ordinary CRH fit on that
component's claims, so the numerical scale stays the one already used for
the 0.55 rule. If no such component exists, the society record is used.
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


def perfect_component(epoch_rows, exclude):
    hit = defaultdict(lambda: defaultdict(float))
    co = defaultdict(lambda: defaultdict(int))
    for rows in epoch_rows:
        by_slot = defaultdict(dict)
        for slot, value, issuer in rows:
            if slot == exclude or not isinstance(issuer, int) or issuer < 0:
                continue
            by_slot[slot][issuer] = value
        for claims in by_slot.values():
            items = list(claims.items())
            for i, (left, left_v) in enumerate(items):
                for right, right_v in items[i + 1 :]:
                    co[left][right] += 1
                    co[right][left] += 1
                    if left_v == right_v:
                        hit[left][right] += 1.0
                        hit[right][left] += 1.0
    nodes = {issuer for issuer, partners in co.items() if partners}
    best = None
    best_rate = -1.0
    guard = 0
    while len(nodes) >= 2 and guard < 6:
        guard += 1
        idx = sorted(nodes)
        pi = {issuer: 1.0 for issuer in idx}
        total = 0.0
        for _ in range(40):
            nxt = {
                issuer: sum((hit[issuer][other] / co[issuer][other]) * pi[other] for other in idx if co[issuer][other])
                for issuer in idx
            }
            total = sum(nxt.values())
            if total <= 1e-12:
                nodes = set()
                break
            pi = {issuer: nxt[issuer] / total for issuer in idx}
        if total <= 1e-12:
            break
        peak = max(pi.values())
        support = [issuer for issuer in idx if pi[issuer] >= peak * 0.25 and pi[issuer] > 1e-8]
        if len(support) < 2:
            break
        num = den = 0.0
        for i, left in enumerate(support):
            for right in support[i + 1 :]:
                if co[left][right]:
                    den += co[left][right]
                    num += hit[left][right]
        rate = num / den if den else 0.0
        if den > 0 and rate > best_rate:
            best_rate = rate
            best = (set(support), rate, den)
        nodes -= set(support)
    if best is None or best[1] < 1.0 - 1e-12:
        return None
    return best[0]


def run_world(world):
    cfg = next(item for item in fair.RULES if item["name"] == "hier")
    names = world.names
    order = list(world.mutable_ids) + list(world.stable_ids) + list(world.derived_ids)
    modes = ("loo", "comp", "comp_only")
    belief = {key: list(world.init_belief[0]) for key in modes}
    answers = {key: [] for key in modes}
    mail = []
    society = held.rag_blank()
    n_org = len(rx.ORGS)
    eta = cfg.get("eta", 0.5)
    history = []
    for epoch_docs in world.visible:
        mail.extend(epoch_docs)
        find = fair.copy_find(mail, names)
        usable = fair.collapsed(mail, find)
        fresh_c = fair.collapsed(epoch_docs, find)
        hist_rows, hist_max = held.claims_from(usable, names)
        fresh_rows, fresh_max = held.claims_from(fresh_c, names)
        history.append(fresh_rows)
        by_org = [[] for _ in range(n_org)]
        org_slots = [set() for _ in range(n_org)]
        for slot, value, issuer in hist_rows:
            by_org[slot % n_org].append((slot, value, issuer))
            org_slots[slot % n_org].add(slot)
        epoch_slots = {slot for slot, _v, _i in fresh_rows}
        buckets = fair.index_hits(usable, names)
        pre = society
        width = max(len(pre) - 1, fresh_max, hist_max)
        soc_full = held.fit_vector(fresh_rows, pre, eta, width)
        blank = fair.pad(held.rag_blank(), hist_max + 1)
        org_full = [held.fit_vector(by_org[g], blank, 1.0, hist_max) for g in range(n_org)]
        grouped = [[[row for row in rows if row[0] % n_org == g] for rows in history] for g in range(n_org)]
        row = {key: list(belief[key]) for key in modes}
        for slot in order:
            g = slot % n_org
            if slot in epoch_slots:
                kept = [item for item in fresh_rows if item[0] != slot]
                ids = [issuer for _s, _v, issuer in kept] or [0]
                glob = held.fit_vector(kept, pre, eta, max(len(pre) - 1, max(ids)))
            else:
                glob = soc_full
            if slot in org_slots[g]:
                static_org = held.fit_vector([item for item in by_org[g] if item[0] != slot], blank, 1.0, hist_max)
            else:
                static_org = org_full[g]
            members = perfect_component(grouped[g], slot)
            local = list(blank)
            if members:
                claims = [item for item in by_org[g] if item[0] != slot and item[2] in members]
                local = held.fit_vector(claims, blank, 1.0, hist_max)
            hits = buckets.get(slot, [])
            for key in modes:
                schema = fair.schema_value(world, row[key], slot)
                if key == "loo":
                    trust = held.trust_for(glob, static_org, cfg, False)
                elif key == "comp":
                    trust = held.trust_for(glob, local if members else glob, cfg, False)
                else:
                    trust = held.trust_for(glob, local, cfg, True) if not members else (
                        lambda issuer, vec=local: fair.weight_of(vec, issuer)
                    )
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
