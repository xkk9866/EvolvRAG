"""Source weight is the fixed point of contemporaneous agreement.

An edge is the rate at which two sources report the same value in rounds
where both speak about an entity other than the one being revised. Power
iteration finds a component that agrees with itself; deflation finds the
next. A source's weight is its agreement inside its component, shrunk toward
the prior by four pseudo-counts. The count switch can still prefer the
society record when that fixed point clears fewer sources. Development seeds
800 and 801.
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


def agreement_counts(epoch_rows, exclude):
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
    return hit, co


def spectral_vector(epoch_rows, max_id, exclude):
    hit, co = agreement_counts(epoch_rows, exclude)
    weights = [rx.PRIOR for _ in range(max_id + 1)]
    nodes = {issuer for issuer, partners in co.items() if partners}
    guard = 0
    while len(nodes) >= 2 and guard < 8:
        guard += 1
        idx = sorted(nodes)
        pi = {issuer: 1.0 for issuer in idx}
        total = 0.0
        for _ in range(40):
            nxt = {}
            for issuer in idx:
                nxt[issuer] = sum(
                    (hit[issuer][other] / co[issuer][other]) * pi[other]
                    for other in idx
                    if co[issuer][other]
                )
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
        support_set = set(support)
        for issuer in support:
            num = sum(hit[issuer][other] for other in support_set if other != issuer)
            den = sum(co[issuer][other] for other in support_set if other != issuer)
            if den > 0 and issuer <= max_id:
                raw = num / den
                weights[issuer] = (rx.PRIOR * ALPHA + raw * den) / (ALPHA + den)
        nodes -= support_set
    return weights


def run_world(world):
    cfg = next(item for item in fair.RULES if item["name"] == "hier")
    names = world.names
    order = list(world.mutable_ids) + list(world.stable_ids) + list(world.derived_ids)
    modes = ("loo", "spec", "spec_only")
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
        grouped = []
        for g in range(n_org):
            grouped.append([[row for row in rows if row[0] % n_org == g] for rows in history])
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
            local = spectral_vector(grouped[g], width, slot)
            hits = buckets.get(slot, [])
            for key in modes:
                schema = fair.schema_value(world, row[key], slot)
                if key == "loo":
                    trust = held.trust_for(glob, static_org, cfg, False)
                elif key == "spec":
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
                bits.append(f"{key}:{score:.3f}/{metrics['changed']:.3f}/{metrics['stable']:.3f}/{metrics['derived']:.3f}")
            print(regime, seed, " ".join(bits), flush=True)
    for key, by in totals.items():
        means = {r: sum(v) / len(v) for r, v in by.items()}
        print(key, {r: round(v, 3) for r, v in means.items()}, "ALL", round(sum(means.values()) / len(means), 3), flush=True)


if __name__ == "__main__":
    main()
