"""Select the retrieval system on seeds 100-115, then test seeds 200-231.

The score is the minimum of stationary, topic-split, and open-rumor
stability-plasticity. Open-rumor addresses a lie with the same receiver draw
as a truth update, so a unicast filter cannot see the generator's rumor mark.
Within 0.005 of the best minimum, the simpler candidate wins. Simpler means a
local corpus before a corroborated index, before a union with local rumors,
before an index of every document; and a distinct-issuer majority before
whole-set recency, the oldest passage in the top BM25 cluster, the newest
passage in that cluster, ungated CRH, and the incumbent gate. Derived
accuracy does not add a component. Stickiness is not refit.

Corpus rules, fixed before any test seed is read:
  local: documents delivered to that agent.
  corroborated: documents whose recipient list has size at least 2.
  union: local deliveries plus corroborated documents addressed to others.
  all: every generated document, including a unicast rumor.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as rx
import run_rag as rag

ROOT = rx.ROOT
OUT = ROOT / "results" / "rag_v2.json"
DEV = rag.DEV
TEST = rag.TEST
RATE = rag.RATE
POBS = rag.POBS
CORPORA = ("local", "corroborated", "union", "all")
RULES = [
    {"rule": "majority", "eta": 1.0, "topic": False, "schema": False},
    {"rule": "recency", "eta": 1.0, "topic": False, "schema": False},
    {"rule": "cluster_oldest", "eta": 1.0, "topic": False, "schema": False},
    {"rule": "cluster_newest", "eta": 1.0, "topic": False, "schema": False},
    {"rule": "crh", "eta": 1.0, "topic": False, "schema": False},
    {"rule": "gate", "eta": 0.25, "topic": True, "schema": True},
]
RULE_COST = {"majority": 0, "recency": 1, "cluster_oldest": 2, "cluster_newest": 3, "crh": 4, "gate": 5}
REGIMES = ("stationary", "topic", "open_rumor")
CORPUS_COST = {"local": 0, "corroborated": 1, "union": 2, "all": 3}


def evaluate(seeds, regime):
    bucket = {(corpus, i): [] for corpus in CORPORA for i in range(len(RULES))}
    coverage_rows = {corpus: [] for corpus in CORPORA}
    for seed in seeds:
        world = rx.generate_trajectory(seed, RATE, POBS, regime=regime)
        parts = []
        for corpus in CORPORA:
            order, epochs, coverage = rag.collect_trace(world, corpus)
            coverage_rows[corpus].append(coverage)
            for i, cfg in enumerate(RULES):
                metrics = rag.score_trace(world, order, epochs, coverage, cfg)
                bucket[(corpus, i)].append(metrics)
            parts.append(f"{corpus}:{bucket[(corpus, 0)][-1]['sp']:.3f}")
        print(regime, seed, " ".join(parts), flush=True)
    cells = {f"{corpus}|{i}": rag.cell_of(bucket[(corpus, i)]) for corpus in CORPORA for i in range(len(RULES))}
    coverage = {}
    for corpus, rows in coverage_rows.items():
        coverage[corpus] = {
            key: sum(row[key] for row in rows) / float(len(rows))
            for key in ("recall5", "entity_recall", "held_entity", "held_value", "published_docs", "published_bytes")
        }
    return cells, coverage


def simplicity(corpus, cfg):
    return (
        CORPUS_COST[corpus],
        RULE_COST[cfg["rule"]],
        bool(cfg.get("schema", False)),
        bool(cfg.get("topic", False)),
        cfg.get("window", 0),
    )


def select(dev_cells):
    ranked = []
    for corpus in CORPORA:
        for i, cfg in enumerate(RULES):
            key = f"{corpus}|{i}"
            scores = {regime: dev_cells[regime][key]["sp"] for regime in REGIMES}
            ranked.append((min(scores.values()), simplicity(corpus, cfg), corpus, i, cfg, scores))
    best_min = max(row[0] for row in ranked)
    near = [row for row in ranked if best_min - row[0] <= 0.005]
    near.sort(key=lambda row: (row[1], -row[0]))
    chosen = near[0]
    return {
        "corpus": chosen[2],
        "rule_index": chosen[3],
        "cfg": chosen[4],
        "dev_min_sp": chosen[0],
        "dev_sp": chosen[5],
        "band": 0.005,
        "grid": [
            {
                "corpus": corpus,
                "cfg": cfg,
                "sp": {regime: dev_cells[regime][f"{corpus}|{i}"]["sp"] for regime in REGIMES},
                "min_sp": min(dev_cells[regime][f"{corpus}|{i}"]["sp"] for regime in REGIMES),
                "derived": {regime: dev_cells[regime][f"{corpus}|{i}"]["derived"] for regime in REGIMES},
            }
            for corpus in CORPORA
            for i, cfg in enumerate(RULES)
        ],
    }


def paired(cells, left, right, metric="sp"):
    diffs = [a - b for a, b in zip(cells[left][metric + "_seeds"], cells[right][metric + "_seeds"])]
    mu, sd = rx.mean_std(diffs)
    return {"left": left, "right": right, "metric": metric, "mean_diff": mu, "std_diff": sd, "p": rag.perm_p(diffs), "n": len(diffs)}


def slim(cell):
    keep = ("sp", "changed", "stable", "derived", "obsolete", "unnec", "delay", "cfa")
    out = {}
    for key, value in cell.items():
        if key.endswith("_seeds"):
            if key.split("_seeds")[0] in keep or key in ("sp_seeds", "changed_seeds", "stable_seeds", "derived_seeds"):
                out[key] = value
            continue
        if key.endswith("_std") and key[: -len("_std")] not in keep and not key.startswith("sp"):
            base = key[: -len("_std")]
            if base not in keep:
                continue
        if key in keep or key.endswith("_std") or key in ("recall5", "entity_recall", "held_entity", "held_value"):
            out[key] = value
    return out


def main():
    started = time.time()
    dev_cells = {}
    dev_cov = {}
    for regime in REGIMES:
        print("DEV", regime, flush=True)
        dev_cells[regime], dev_cov[regime] = evaluate(DEV, regime)
    chosen = select(dev_cells)
    label = f"{chosen['corpus']}|{chosen['rule_index']}"
    print("SELECTED", label, json.dumps(chosen["cfg"]), "min", round(chosen["dev_min_sp"], 4), flush=True)
    test_cells = {}
    test_cov = {}
    for regime in REGIMES:
        print("TEST", regime, flush=True)
        test_cells[regime], test_cov[regime] = evaluate(TEST, regime)
    tests = []
    for regime, cells in test_cells.items():
        for corpus in CORPORA:
            for i in range(len(RULES)):
                other = f"{corpus}|{i}"
                if other == label:
                    continue
                if corpus != chosen["corpus"] and i != chosen["rule_index"]:
                    continue
                for metric in ("sp", "changed", "stable", "derived"):
                    tests.append({"regime": regime, **paired(cells, label, other, metric)})
    payload = {
        "protocol": {
            "dev_seeds": DEV,
            "test_seeds": TEST,
            "selection": "max min(stationary, topic, open_rumor SP); within 0.005 prefer the simpler corpus and rule",
            "selected": chosen,
            "rules": RULES,
            "corpora": CORPORA,
            "regimes": REGIMES,
        },
        "dev_coverage": dev_cov,
        "test_coverage": test_cov,
        "test": {regime: {key: slim(value) for key, value in cells.items()} for regime, cells in test_cells.items()},
        "tests": tests,
        "seconds": time.time() - started,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("WROTE", OUT, flush=True)
    for row in tests:
        if row["metric"] == "sp" and row["regime"] == "stationary":
            print("SP", row["left"], "minus", row["right"], f"{row['mean_diff']:+.4f}", f"p={row['p']:.6f}", flush=True)


if __name__ == "__main__":
    main()
