"""Competence is the connected component of perfect contemporaneous agreement.

Two sources are linked when, on every other entity they both addressed in
the same round, they reported the same value. A source may revise the held-out
entity only as a member of such a component. The revision is that component's
latest passage. If the component is silent, the society held-out record is
used. Development seeds 800 and 801 decide whether this operator proceeds.
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
LARGEST = True


class UnionFind:
    def __init__(self):
        self.parent = {}

    def add(self, node):
        self.parent.setdefault(node, node)

    def find(self, node):
        self.add(node)
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def consistent_members(epoch_rows, exclude):
    """Sources that sit in a size-at-least-2 block of perfect agreement."""
    co = defaultdict(int)
    hit = defaultdict(int)
    seen_issuers = set()
    for rows in epoch_rows:
        by_slot = defaultdict(dict)
        for slot, value, issuer in rows:
            if slot == exclude or not isinstance(issuer, int) or issuer < 0:
                continue
            by_slot[slot][issuer] = value
            seen_issuers.add(issuer)
        for claims in by_slot.values():
            items = list(claims.items())
            for i, (left, left_v) in enumerate(items):
                for right, right_v in items[i + 1 :]:
                    key = (left, right) if left < right else (right, left)
                    co[key] += 1
                    if left_v == right_v:
                        hit[key] += 1
    graph = UnionFind()
    linked = set()
    for key, n in co.items():
        if n > 0 and hit[key] == n:
            graph.union(*key)
            linked.add(key[0])
            linked.add(key[1])
    groups = defaultdict(set)
    for issuer in linked:
        groups[graph.find(issuer)].add(issuer)
    groups = [group for group in groups.values() if len(group) >= 2]
    if not groups:
        return set()
    groups.sort(key=len)
    if LARGEST:
        return groups[-1]
    return set().union(*groups)


def latest_member(hits, members):
    best = None
    best_key = None
    for hit in hits:
        issuer = hit["issuer"]
        if issuer not in members:
            continue
        key = (hit["time"], hit.get("seq", 0))
        if best_key is None or key > best_key:
            best = hit
            best_key = key
    if best is None:
        return None
    return best["value"]


def round_mode(fresh_pairs, members):
    counts = defaultdict(int)
    for value, issuer in fresh_pairs:
        if issuer in members:
            counts[value] += 1
    if not counts:
        return None
    ranked = sorted(counts.items(), key=lambda item: item[1])
    if len(ranked) == 1 or ranked[-1][1] > ranked[-2][1]:
        return ranked[-1][0]
    return None


def run_world(world):
    cfg = next(item for item in fair.RULES if item["name"] == "hier")
    names = world.names
    order = list(world.mutable_ids) + list(world.stable_ids) + list(world.derived_ids)
    modes = ("loo", "latest", "mode", "only")
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
        _hist_rows, hist_max = held.claims_from(usable, names)
        fresh_rows, fresh_max = held.claims_from(fresh_c, names)
        history.append(fresh_rows)
        fresh_by_slot = defaultdict(list)
        for slot, value, issuer in fresh_rows:
            fresh_by_slot[slot].append((value, issuer))
        grouped = [[[row for row in rows if row[0] % n_org == g] for rows in history] for g in range(n_org)]
        epoch_slots = set(fresh_by_slot)
        buckets = fair.index_hits(usable, names)
        pre = society
        width = max(len(pre) - 1, fresh_max, hist_max)
        soc_full = held.fit_vector(fresh_rows, pre, eta, width)
        row = {key: list(belief[key]) for key in modes}
        for slot in order:
            g = slot % n_org
            if slot in epoch_slots:
                kept = [item for item in fresh_rows if item[0] != slot]
                ids = [issuer for _s, _v, issuer in kept] or [0]
                glob = held.fit_vector(kept, pre, eta, max(len(pre) - 1, max(ids)))
            else:
                glob = soc_full
            members = consistent_members(grouped[g], slot)
            hits = buckets.get(slot, [])
            society_trust = held.trust_for(glob, glob, cfg, True)
            for key in modes:
                schema = fair.schema_value(world, row[key], slot)
                base = fair.answer_slot(cfg, hits, row[key][slot], society_trust, schema)
                if key == "loo":
                    # rebuilt below from the count switch; placeholder replaced after this loop
                    row[key][slot] = base
                elif key == "latest":
                    chosen = latest_member(hits, members) if members else None
                    row[key][slot] = chosen if chosen is not None else base
                elif key == "mode":
                    chosen = round_mode(fresh_by_slot.get(slot, []), members) if members else None
                    row[key][slot] = chosen if chosen is not None else base
                else:
                    chosen = latest_member(hits, members) if members else None
                    if chosen is not None:
                        row[key][slot] = chosen
                    elif schema is not None:
                        row[key][slot] = schema
                    else:
                        row[key][slot] = row[key][slot]
        # The count-switch baseline needs the static organization record.
        # It is computed in a second, cheaper pass over the same fits below.
        for key in modes:
            belief[key] = row[key]
            answers[key].append([list(row[key]) for _ in range(rx.N_AGENTS)])
        society = soc_full
    # loo was filled with the society record. Recompute it with the locked operator.
    loo = held.run_loo(world)["loo"]
    out = {"loo": loo}
    for key, grid in answers.items():
        if key == "loo":
            continue
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
