"""Forward the eligible passage. The recipient list is only the delivery constraint.

Each agent reads the documents addressed to it and runs clearance routing.
It then sends the fresh eligible passage it adopted, with the issuer and the
time left intact, to every other agent. The passage arrives on the next
epoch. No weight vector is sent, and no new threshold is introduced.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_decentral as dec
import run_experiment as rx
import run_fair as fair

ROOT = rx.ROOT
OUT = ROOT / "results" / "relay.json"
TEST = list(range(700, 732))
REGIMES = dec.REGIMES
HIER = dec.HIER
TAU = 0.55


def run_relay(world):
    names = world.names
    order = list(world.mutable_ids) + list(world.stable_ids) + list(world.derived_ids)
    beliefs = [list(row) for row in world.init_belief]
    books = [fair.TrustBook(HIER) for _ in range(rx.N_AGENTS)]
    mail = [[] for _ in range(rx.N_AGENTS)]
    pending = [[] for _ in range(rx.N_AGENTS)]
    answers = []
    held = 0
    asked = 0
    n_sent = 0
    for epoch_docs in world.visible:
        fresh = [[] for _ in range(rx.N_AGENTS)]
        for agent in range(rx.N_AGENTS):
            for doc in pending[agent]:
                mail[agent].append(doc)
                fresh[agent].append(doc)
            pending[agent] = []
        for doc in epoch_docs:
            for agent in doc["receivers"]:
                if 0 <= agent < rx.N_AGENTS:
                    mail[agent].append(doc)
                    fresh[agent].append(doc)
        grid = []
        sent = []
        for agent in range(rx.N_AGENTS):
            find = fair.copy_find(mail[agent], names)
            usable = fair.collapsed(mail[agent], find)
            fresh_c = fair.collapsed(fresh[agent], find)
            buckets = fair.index_hits(usable, names)
            fresh_buckets = fair.index_hits(fresh_c, names)
            row = list(beliefs[agent])
            adopted = []
            for slot in order:
                asked += 1
                if buckets.get(slot):
                    held += 1
                schema_v = fair.schema_value(world, row, slot)
                trust = books[agent].for_slot(slot)
                row[slot] = fair.answer_slot(HIER, buckets.get(slot, []), row[slot], trust, schema_v)
                picked, saw = fair.trusted_pick(fresh_buckets.get(slot, []), None, trust, TAU)
                if saw and picked == row[slot]:
                    for hit in fresh_buckets.get(slot, []):
                        weight = fair.weight_of(trust, hit["issuer"])
                        if hit["value"] == picked and weight is not None and weight + 1e-9 >= TAU:
                            adopted.append(hit)
                            break
            beliefs[agent] = row
            grid.append(row)
            books[agent].update(fresh_c, usable, names)
            sent.append(adopted)
        answers.append(grid)
        for agent, docs in enumerate(sent):
            n_sent += len(docs)
            for doc in docs:
                relay = {
                    "slot": doc["slot"],
                    "value": doc["value"],
                    "time": doc["time"],
                    "issuer": doc["issuer"],
                    "text": doc["text"],
                    "seq": doc.get("seq", 0),
                    "receivers": [],
                }
                for other in range(rx.N_AGENTS):
                    if other != agent:
                        pending[other].append(relay)
    counts = rx.empty_counts()
    rx.consume(counts, world, answers, 0.0)
    packed = rx.pack_metrics(counts)
    packed["coverage"] = held / float(asked)
    packed["messages"] = n_sent / float(rx.N_AGENTS * len(world.visible))
    return packed


def main():
    stored = {name: {regime: [] for regime in REGIMES} for name in ("central", "private", "relay")}
    for regime in REGIMES:
        for seed in TEST:
            world = fair.load_world(seed, regime)
            stored["central"][regime].append(dec.metrics_of(world, HIER))
            stored["private"][regime].append(dec.run_private(world, dec.PRIVATE))
            stored["relay"][regime].append(run_relay(world))
            print(
                regime,
                seed,
                " ".join(f"{name}:{dec.balanced(stored[name][regime][-1]):.3f}" for name in stored),
                flush=True,
            )
    summary = {}
    tests = []
    for name in stored:
        flat = []
        per = {}
        for regime in REGIMES:
            values = [dec.balanced(row) for row in stored[name][regime]]
            mu, sd = rx.mean_std(values)
            per[regime] = {
                "balanced": mu,
                "std": sd,
                "sp": rx.mean_std([row["sp"] for row in stored[name][regime]])[0],
                "coverage": rx.mean_std([row.get("coverage", 0.0) for row in stored[name][regime]])[0],
            }
            flat.extend(values)
        mu, sd = rx.mean_std(flat)
        summary[name] = {"balanced": mu, "std": sd, "per_regime": per}
    for regime in REGIMES:
        for other in ("private", "central"):
            tests.append(
                {
                    "regime": regime,
                    "other": other,
                    **dec.paired(
                        [dec.balanced(row) for row in stored["relay"][regime]],
                        [dec.balanced(row) for row in stored[other][regime]],
                    ),
                }
            )
    payload = {"summary": summary, "tests": tests, "seeds": TEST, "rule": "published clearance, eligible fresh passage forwarded"}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
