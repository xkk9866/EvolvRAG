"""Missing reviewer cells on seeds 800-801 only. Test seeds stay unread.

additive: the dated agreement-count weight, with the same hold-out,
count switch, copy collapse, and latest eligible passage as EvolvRAG.
truthfinder: streaming TruthFinder on each round's collapsed claims,
with source trust carried across rounds.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as rx
import run_fair as fair
import run_fusion as fusion
import run_heldout as held

REGIMES = ("stationary", "topic", "open_rumor", "sybil")


def additive_vector(claims, max_id):
    by_slot = defaultdict(dict)
    for slot, value, issuer in claims:
        by_slot[slot][issuer] = value
    hits = defaultdict(int)
    counts = defaultdict(int)
    for votes in by_slot.values():
        tally = defaultdict(int)
        for value in votes.values():
            tally[value] += 1
        label = max(tally.items(), key=lambda item: (item[1], str(item[0])))[0]
        for issuer, value in votes.items():
            counts[issuer] += 1
            hits[issuer] += int(value == label)
    vec = [rx.PRIOR] * (max_id + 1)
    for issuer, count in counts.items():
        if 0 <= issuer <= max_id and count >= 2:
            vec[issuer] = fusion.additive_from_hits(hits[issuer], count)
    return vec


def mix(old, new, eta):
    width = max(len(old), len(new))
    out = []
    for index in range(width):
        left = old[index] if index < len(old) else rx.PRIOR
        right = new[index] if index < len(new) else rx.PRIOR
        out.append((1.0 - eta) * left + eta * right)
    return out


def run_additive(world):
    cfg = next(item for item in fair.RULES if item["name"] == "hier")
    names = world.names
    order = list(world.mutable_ids) + list(world.stable_ids) + list(world.derived_ids)
    belief = list(world.init_belief[0])
    mail = []
    society = [rx.PRIOR] * (rx.N_REL + rx.N_NOISY)
    answers = []
    n_org = len(rx.ORGS)
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
            org = slot % n_org
            by_org[org].append((slot, value, issuer))
            org_slots[org].add(slot)
        epoch_slots = {slot for slot, _value, _issuer in fresh_rows}
        buckets = fair.index_hits(usable, names)
        width = max(hist_max, fresh_max, len(society) - 1)
        soc_full = mix(society, additive_vector(fresh_rows, width), 0.5)
        org_full = [additive_vector(by_org[org], width) for org in range(n_org)]
        row = list(belief)
        for slot in order:
            org = slot % n_org
            if slot in epoch_slots:
                kept = [item for item in fresh_rows if item[0] != slot]
                glob = mix(society, additive_vector(kept, width), 0.5)
            else:
                glob = soc_full
            if slot in org_slots[org]:
                kept_org = [item for item in by_org[org] if item[0] != slot]
                org_vec = additive_vector(kept_org, width)
            else:
                org_vec = org_full[org]
            hits = buckets.get(slot, [])
            schema_v = fair.schema_value(world, row, slot)
            row[slot] = fair.answer_slot(
                cfg, hits, row[slot], held.trust_for(glob, org_vec, cfg, False), schema_v
            )
        belief = row
        answers.append([list(row) for _ in range(rx.N_AGENTS)])
        society = soc_full
    counts = rx.empty_counts()
    rx.consume(counts, world, answers, 0.0)
    return rx.pack_metrics(counts)


def truthfinder_round(claims, trust):
    by_slot = defaultdict(dict)
    for slot, value, issuer in claims:
        by_slot[slot][issuer] = value
    issuers = {issuer for votes in by_slot.values() for issuer in votes}
    current = {issuer: trust.get(issuer, 0.8) for issuer in issuers}
    fact_score = {}
    for _ in range(6):
        fact_score = {}
        for slot, votes in by_slot.items():
            groups = defaultdict(list)
            for issuer, value in votes.items():
                groups[value].append(issuer)
            for value, members in groups.items():
                remain = 1.0
                for issuer in members:
                    remain *= 1.0 - min(0.99, max(0.01, current[issuer]))
                fact_score[(slot, value)] = 1.0 - remain
        updated = {}
        for issuer in issuers:
            scores = [
                fact_score[(slot, votes[issuer])]
                for slot, votes in by_slot.items()
                if issuer in votes
            ]
            updated[issuer] = sum(scores) / len(scores) if scores else current[issuer]
        current = updated
    chosen = {}
    for slot, votes in by_slot.items():
        best_value = None
        best_score = -1.0
        for value in set(votes.values()):
            score = fact_score[(slot, value)]
            if score > best_score or (score == best_score and str(value) < str(best_value)):
                best_value = value
                best_score = score
        chosen[slot] = best_value
    trust.update(current)
    return chosen


def run_truthfinder(world):
    names = world.names
    order = list(world.mutable_ids) + list(world.stable_ids) + list(world.derived_ids)
    belief = list(world.init_belief[0])
    trust = {}
    mail = []
    answers = []
    for epoch_docs in world.visible:
        mail.extend(epoch_docs)
        find = fair.copy_find(mail, names)
        fresh = fair.collapsed(epoch_docs, find)
        claims, _max_id = held.claims_from(fresh, names)
        chosen = truthfinder_round(claims, trust)
        row = list(belief)
        for slot in order:
            schema_v = fair.schema_value(world, row, slot)
            if slot in chosen:
                row[slot] = chosen[slot]
            elif schema_v is not None:
                row[slot] = schema_v
        belief = row
        answers.append([list(row) for _ in range(rx.N_AGENTS)])
    counts = rx.empty_counts()
    rx.consume(counts, world, answers, 0.0)
    return rx.pack_metrics(counts)


def balanced(metrics):
    return (metrics["changed"] + metrics["stable"] + metrics["derived"]) / 3.0


def main():
    stored = {name: {regime: [] for regime in REGIMES} for name in ("additive", "truthfinder")}
    for regime in REGIMES:
        for seed in (800, 801):
            world = fair.load_world(seed, regime)
            packed = {"additive": run_additive(world), "truthfinder": run_truthfinder(world)}
            bits = []
            for name in stored:
                stored[name][regime].append(packed[name])
                bits.append(
                    f"{name}:{balanced(packed[name]):.3f}/{packed[name]['sp']:.3f}"
                )
            print(regime, seed, " ".join(bits), flush=True)
    summary = {}
    for name, regimes in stored.items():
        by = {regime: sum(balanced(row) for row in rows) / len(rows) for regime, rows in regimes.items()}
        summary[name] = {"by_regime": by, "mean": sum(by.values()) / len(by)}
    out = rx.ROOT / "results" / "unified_probe.json"
    slim = {
        name: {
            regime: [
                {key: row[key] for key in ("sp", "changed", "stable", "derived", "obsolete", "delay")}
                for row in rows
            ]
            for regime, rows in regimes.items()
        }
        for name, regimes in stored.items()
    }
    out.write_text(json.dumps({"summary": summary, "seeds": slim}), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
