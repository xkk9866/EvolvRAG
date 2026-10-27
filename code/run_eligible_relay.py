"""Eligible testimony is the message. Probe on seeds 800-801 before any test read.

Each agent revises only from documents it holds. After the decision it may
forward passages its own record has already cleared, with the original issuer
and time. The one-passage relay forwards only the passage just adopted. The
corpus relay forwards every cleared passage that agent has not forwarded
before. No weight vector and no recipient list is sent. Tau, margin, copy,
and schema stay fixed. Seeds 900-931 are not read here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_decentral as dec
import run_experiment as rx
import run_fair as fair
import run_relay as relay

TAU = 0.55
REGIMES = ("stationary", "topic", "open_rumor", "sybil")


def run_corpus(world):
    names = world.names
    order = list(world.mutable_ids) + list(world.stable_ids) + list(world.derived_ids)
    beliefs = [list(row) for row in world.init_belief]
    books = [fair.TrustBook(dec.HIER) for _ in range(rx.N_AGENTS)]
    mail = [[] for _ in range(rx.N_AGENTS)]
    pending = [[] for _ in range(rx.N_AGENTS)]
    forwarded = [set() for _ in range(rx.N_AGENTS)]
    answers = []
    held = 0
    asked = 0
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
            row = list(beliefs[agent])
            packet = []
            for slot in order:
                asked += 1
                hits = buckets.get(slot, [])
                if hits:
                    held += 1
                schema_v = fair.schema_value(world, row, slot)
                trust = books[agent].for_slot(slot)
                row[slot] = fair.answer_slot(dec.HIER, hits, row[slot], trust, schema_v)
                for hit in hits:
                    weight = fair.weight_of(trust, hit["issuer"])
                    if weight is None or weight + 1e-9 < TAU:
                        continue
                    key = (hit["issuer"], hit["value"], hit["time"], hit["text"])
                    if key in forwarded[agent]:
                        continue
                    forwarded[agent].add(key)
                    packet.append(hit)
            beliefs[agent] = row
            grid.append(row)
            books[agent].update(fresh_c, usable, names)
            sent.append(packet)
        answers.append(grid)
        for agent, docs in enumerate(sent):
            for doc in docs:
                relay_doc = {
                    "slot": doc.get("slot"),
                    "value": doc["value"],
                    "time": doc["time"],
                    "issuer": doc["issuer"],
                    "text": doc["text"],
                    "seq": doc.get("seq", 0),
                    "receivers": [],
                }
                for other in range(rx.N_AGENTS):
                    if other != agent:
                        pending[other].append(relay_doc)
    counts = rx.empty_counts()
    rx.consume(counts, world, answers, 0.0)
    packed = rx.pack_metrics(counts)
    packed["coverage"] = held / float(asked)
    return packed


def balanced(metrics):
    return (metrics["changed"] + metrics["stable"] + metrics["derived"]) / 3.0


def main():
    seeds = [800, 801]
    stored = {name: {regime: [] for regime in REGIMES} for name in ("central", "private", "relay", "corpus")}
    for regime in REGIMES:
        for seed in seeds:
            world = fair.load_world(seed, regime)
            stored["central"][regime].append(dec.metrics_of(world, dec.HIER))
            stored["private"][regime].append(dec.run_private(world, dec.PRIVATE))
            stored["relay"][regime].append(relay.run_relay(world))
            stored["corpus"][regime].append(run_corpus(world))
            print(
                regime,
                seed,
                " ".join(
                    f"{name}:{balanced(stored[name][regime][-1]):.3f}/c{stored[name][regime][-1].get('coverage', 0):.3f}"
                    for name in stored
                ),
                flush=True,
            )
    summary = {}
    for name in stored:
        by = {
            regime: sum(balanced(row) for row in rows) / len(rows)
            for regime, rows in stored[name].items()
        }
        summary[name] = {"by_regime": by, "mean": sum(by.values()) / len(by)}
    out = rx.ROOT / "results" / "eligible_probe.json"
    out.write_text(json.dumps({"summary": summary}, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
