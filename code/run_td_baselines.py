"""Published truth-discovery baselines on the existing test seeds 700-731.

TruthFinder carries source trust and scores each round's collapsed claims.
One-coin Dawid--Skene does the same with a latent value and a per-source
accuracy. The agreement-count weight uses EvolvRAG's hold-out and count
switch, so the only change from the selected operator is the weight formula.
Results are written to a new file.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as rx
import run_fair as fair
import run_unified_probe as uni

REGIMES = ("stationary", "topic", "open_rumor", "sybil")
SEEDS = list(range(700, 732))


def dawid_skene_round(claims, accuracy):
    by_slot = defaultdict(dict)
    for slot, value, issuer in claims:
        by_slot[slot][issuer] = value
    issuers = {issuer for votes in by_slot.values() for issuer in votes}
    acc = {issuer: accuracy.get(issuer, 0.6) for issuer in issuers}
    truth = {}
    for slot, votes in by_slot.items():
        tally = defaultdict(int)
        for value in votes.values():
            tally[value] += 1
        truth[slot] = max(tally.items(), key=lambda item: (item[1], str(item[0])))[0]
    for _ in range(8):
        correct = defaultdict(float)
        total = defaultdict(float)
        for slot, votes in by_slot.items():
            for issuer, value in votes.items():
                total[issuer] += 1.0
                if value == truth[slot]:
                    correct[issuer] += 1.0
        for issuer in issuers:
            acc[issuer] = (1.0 + correct[issuer]) / (2.0 + total[issuer])
        updated = {}
        for slot, votes in by_slot.items():
            values = list(set(votes.values()))
            classes = max(len(values), 2)
            best_value = None
            best_score = None
            for value in values:
                score = 0.0
                for issuer, said in votes.items():
                    prob = min(0.99, max(0.01, acc[issuer]))
                    if said == value:
                        score += math.log(prob)
                    else:
                        score += math.log((1.0 - prob) / (classes - 1))
                if best_score is None or score > best_score:
                    best_value = value
                    best_score = score
            updated[slot] = best_value
        truth = updated
    accuracy.update(acc)
    return truth


def run_dawid(world):
    names = world.names
    order = list(world.mutable_ids) + list(world.stable_ids) + list(world.derived_ids)
    belief = list(world.init_belief[0])
    accuracy = {}
    mail = []
    answers = []
    for epoch_docs in world.visible:
        mail.extend(epoch_docs)
        find = fair.copy_find(mail, names)
        fresh = fair.collapsed(epoch_docs, find)
        claims, _max_id = uni.held.claims_from(fresh, names)
        chosen = dawid_skene_round(claims, accuracy)
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
    names = ("truthfinder", "dawid", "additive")
    stored = {name: {regime: [] for regime in REGIMES} for name in names}
    runners = {
        "truthfinder": uni.run_truthfinder,
        "dawid": run_dawid,
        "additive": uni.run_additive,
    }
    for regime in REGIMES:
        for seed in SEEDS:
            world = fair.load_world(seed, regime)
            bits = []
            for name in names:
                metrics = runners[name](world)
                stored[name][regime].append(
                    {key: metrics[key] for key in ("sp", "changed", "stable", "derived", "obsolete", "delay", "cfa")}
                )
                bits.append(f"{name}:{balanced(metrics):.3f}")
            print(regime, seed, " ".join(bits), flush=True)
    summary = {}
    for name, regimes in stored.items():
        by = {regime: sum(balanced(row) for row in rows) / len(rows) for regime, rows in regimes.items()}
        summary[name] = {"by_regime": by, "mean": sum(by.values()) / len(by)}
    out = rx.ROOT / "results" / "td_baselines_test.json"
    out.write_text(json.dumps({"summary": summary, "seeds": stored}), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
