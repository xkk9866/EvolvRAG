"""Matched-evidence comparisons for the shared mailbox.

The shared board reads every generated document. Local majority reads only
the documents delivered to that agent. This runner scores decision rules on
the same document list, then repeats the comparison under topic-split
issuers, mid-run reliability swap, and heavier copying. Thresholds stay at
the values locked on the original development seeds. Nothing here retunes them.

Seeds 5-36 are the stationary test. Seeds 5-20 are the stress tests.
A paired sign-flip is exact through 16 seeds and Monte Carlo beyond that.
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as rx

ROOT = rx.ROOT
OUT = ROOT / "results" / "causal.json"
RATE = rx.DEFAULT_RATE
POBS = rx.DEFAULT_POBS
STATIONARY_SEEDS = list(range(5, 37))
STRESS_SEEDS = list(range(5, 21))
SCALE_SEEDS = list(range(5, 13))
MATCHED = [
    "Majority",
    "PoolMajority",
    "PoolCount",
    "PoolCRH",
    "PoolUniform",
    "PoolOpen",
    "PoolWeight",
    "NoShrink",
    "Indep",
    "OrgTrust",
    "Routed",
    "Central",
    "EvolvRAG",
]
STRESS_METHODS = ["Majority", "PoolMajority", "PoolCRH", "Central", "Indep", "OrgTrust", "Routed"]
SCALE_METHODS = ["Majority", "PoolMajority", "Central", "Routed"]


def perm_p(diffs, draws=100000, seed=3157):
    n = len(diffs)
    if n == 0:
        raise RuntimeError("empty paired sample")
    if n <= 16:
        return rx.signflip_p(diffs)
    obs = abs(sum(diffs))
    if obs <= 1e-12:
        return 1.0
    rng = random.Random(seed)
    count = 1
    for _ in range(draws - 1):
        signed = 0.0
        for diff in diffs:
            signed += diff if rng.randrange(2) else -diff
        if abs(signed) + 1e-9 >= obs:
            count += 1
    return count / draws


def cell_from(store):
    cell = {}
    for name, values in store.items():
        mu, sd = rx.mean_std(values)
        cell[name] = mu
        cell[name + "_std"] = sd
        if name in ("sp", "changed", "stable", "cfa", "comm", "derived"):
            cell[name + "_seeds"] = values
    return cell


def run_seeds(seeds, regime, methods):
    bucket = {method: defaultdict(list) for method in methods}
    worlds = []
    for seed in seeds:
        world = rx.generate_trajectory(seed, RATE, POBS, regime=regime)
        worlds.append(world)
        for method in methods:
            counts, _answers = rx.run_method(world, method, seed)
            metrics = rx.pack_metrics(counts)
            for name, value in metrics.items():
                if name == "epoch_cfa":
                    continue
                bucket[method][name].append(value)
        print(regime, seed, " ".join(f"{method}={bucket[method]['sp'][-1]:.3f}" for method in methods), flush=True)
    return {method: cell_from(store) for method, store in bucket.items()}, worlds


def evidence_account(worlds):
    distinct_n = 0
    naive_n = 0
    distinct_bytes = 0
    naive_bytes = 0
    receiver_frac = []
    covered = 0
    docs_n = 0
    true_docs = 0
    true_covered_agents = []
    for world in worlds:
        for epoch_i, docs in enumerate(world.visible):
            seen = set()
            for doc in docs:
                docs_n += 1
                receivers = doc["receivers"]
                frac = len(receivers) / float(rx.N_AGENTS)
                receiver_frac.append(frac)
                if receivers:
                    covered += 1
                payload = len(doc["text"].encode("utf-8")) + 8
                naive_n += len(receivers)
                naive_bytes += payload * len(receivers)
                key = (doc["slot"], doc["value"], doc["time"], doc["issuer"])
                if key in seen or not receivers:
                    continue
                seen.add(key)
                distinct_n += 1
                distinct_bytes += payload
                truth = world.timeline[epoch_i][doc["slot"]]
                if doc["value"] == truth and doc["slot"] in world.mutable_ids:
                    true_docs += 1
                    true_covered_agents.append(frac)
    denom = float(rx.N_AGENTS * rx.EPOCHS * len(worlds))
    return {
        "docs": docs_n,
        "fraction_with_a_receiver": covered / float(docs_n),
        "mean_receiver_fraction": sum(receiver_frac) / float(len(receiver_frac)),
        "union_coverage_bound": 1.0 - (1.0 - POBS) ** rx.N_AGENTS,
        "true_update_docs": true_docs,
        "true_update_mean_receiver_fraction": sum(true_covered_agents) / float(len(true_covered_agents)),
        "distinct_uploads_per_agent_epoch": distinct_n / denom,
        "naive_uploads_per_agent_epoch": naive_n / denom,
        "distinct_upload_bytes_per_agent_epoch": distinct_bytes / denom,
        "naive_upload_bytes_per_agent_epoch": naive_bytes / denom,
        "belief_message_bytes": 16,
    }


def selective_filter(world):
    epochs = []
    uploads = 0
    for docs in world.visible:
        boxes = defaultdict(list)
        for doc in docs:
            for agent in doc["receivers"]:
                boxes[agent].append(doc)
        kept = set()
        for packet in boxes.values():
            first = {}
            for doc in packet:
                key = (doc["slot"], doc["value"], doc["time"], doc["issuer"])
                if doc["slot"] not in first:
                    first[doc["slot"]] = doc["value"]
                    kept.add(key)
                    uploads += 1
                elif doc["value"] != first[doc["slot"]]:
                    if key not in kept:
                        uploads += 1
                    kept.add(key)
        epochs.append(
            [doc for doc in docs if (doc["slot"], doc["value"], doc["time"], doc["issuer"]) in kept or not doc["receivers"]]
        )
    return epochs, uploads


def paired(cells, left, right, metric="sp"):
    a = cells[left][metric + "_seeds"]
    b = cells[right][metric + "_seeds"]
    diffs = [x - y for x, y in zip(a, b)]
    mu, sd = rx.mean_std(diffs)
    return {
        "left": left,
        "right": right,
        "metric": metric,
        "mean_diff": mu,
        "std_diff": sd,
        "p": perm_p(diffs),
        "n": len(diffs),
    }


def lambda_curve(cells):
    grid = [i / 10 for i in range(11)]
    out = {}
    for method, cell in cells.items():
        changed = cell["changed_seeds"]
        stable = cell["stable_seeds"]
        rows = []
        for lam in grid:
            scores = [lam * c + (1.0 - lam) * s for c, s in zip(changed, stable)]
            mu, sd = rx.mean_std(scores)
            rows.append({"lambda": lam, "score": mu, "std": sd})
        out[method] = rows
    return out


def check_identity():
    payload = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))
    stored = None
    for cell in payload["cells"]:
        if cell["method"] == "Central" and abs(cell["rate"] - RATE) < 1e-9 and abs(cell["p_obs"] - POBS) < 1e-9:
            stored = cell["sp_seeds"]
            break
    if stored is None:
        raise RuntimeError("stored Central seeds missing")
    fresh = []
    for seed in rx.SEEDS_TEST:
        world = rx.generate_trajectory(seed, RATE, POBS)
        counts, _answers = rx.run_method(world, "Central", seed)
        fresh.append(rx.pack_metrics(counts)["sp"])
    gaps = [abs(a - b) for a, b in zip(fresh, stored)]
    if max(gaps) > 1e-9:
        raise RuntimeError(f"Central trajectory changed: max gap {max(gaps)}")
    print("IDENTITY ok", flush=True)


def with_globals(agents, mutable, fn):
    saved = (rx.N_AGENTS, rx.N_MUTABLE, rx.N_DERIVED)
    rx.N_AGENTS = agents
    rx.N_MUTABLE = mutable
    rx.N_DERIVED = min(saved[2], mutable)
    try:
        return fn()
    finally:
        rx.N_AGENTS, rx.N_MUTABLE, rx.N_DERIVED = saved


def main():
    check_identity()
    stationary, worlds = run_seeds(STATIONARY_SEEDS, "stationary", MATCHED)
    tests = []
    for metric in ("sp", "changed", "stable", "cfa"):
        for left in ("Central", "Routed", "PoolWeight", "Indep", "OrgTrust", "PoolCRH", "PoolUniform", "PoolOpen", "PoolCount", "NoShrink"):
            tests.append(paired(stationary, left, "PoolMajority", metric))
        tests.append(paired(stationary, "Routed", "Central", metric))
        tests.append(paired(stationary, "Central", "Majority", metric))
    account = evidence_account(worlds)
    # Belief broadcasts are the existing revision counts. Uploads are shared
    # by every full-pool rule because they read the same document list.
    for method, cell in stationary.items():
        if method in ("Majority", "EvolvRAG"):
            cell["upload_distinct"] = 0.0
            cell["upload_naive"] = 0.0
            cell["total_distinct"] = cell["comm"]
            cell["total_naive"] = cell["comm"]
        else:
            cell["upload_distinct"] = account["distinct_uploads_per_agent_epoch"]
            cell["upload_naive"] = account["naive_uploads_per_agent_epoch"]
            cell["total_distinct"] = cell["comm"] + cell["upload_distinct"]
            cell["total_naive"] = cell["comm"] + cell["upload_naive"]
            cell["total_distinct_bytes"] = (
                cell["comm"] * account["belief_message_bytes"]
                + account["distinct_upload_bytes_per_agent_epoch"]
            )
            cell["total_naive_bytes"] = (
                cell["comm"] * account["belief_message_bytes"]
                + account["naive_upload_bytes_per_agent_epoch"]
            )
    selective = defaultdict(list)
    selective_uploads = []
    saved_visible = []
    for world in worlds:
        filtered, uploads = selective_filter(world)
        selective_uploads.append(uploads / float(rx.N_AGENTS * rx.EPOCHS))
        saved_visible.append(world.visible)
        world.visible = filtered
    for seed, world in zip(STATIONARY_SEEDS, worlds):
        for method in ("PoolMajority", "Central", "Routed"):
            counts, _answers = rx.run_method(world, method, seed)
            metrics = rx.pack_metrics(counts)
            selective[method].append(metrics)
    for world, visible in zip(worlds, saved_visible):
        world.visible = visible
    selective_cells = {}
    for method, rows in selective.items():
        store = defaultdict(list)
        for metrics in rows:
            for name, value in metrics.items():
                if name != "epoch_cfa":
                    store[name].append(value)
        selective_cells[method] = cell_from(store)
        selective_cells[method]["upload_policy"] = sum(selective_uploads) / float(len(selective_uploads))
    stress = {}
    for regime in ("topic", "drift", "collude"):
        cells, _unused = run_seeds(STRESS_SEEDS, regime, STRESS_METHODS)
        stress[regime] = {
            "cells": cells,
            "tests": [
                paired(cells, left, "PoolMajority", "sp")
                for left in ("Central", "Indep", "OrgTrust", "Routed", "PoolCRH")
            ]
            + [paired(cells, "Routed", "Central", "sp"), paired(cells, "OrgTrust", "Central", "sp"), paired(cells, "Indep", "Central", "sp")],
        }
    scale = []
    for agents in (8, 16, 32):
        def run_agents(agents=agents):
            return run_seeds(SCALE_SEEDS, "stationary", SCALE_METHODS)
        cells, _unused = with_globals(agents, rx.N_MUTABLE, run_agents)
        scale.append({"agents": agents, "mutable": rx.N_MUTABLE, "cells": cells})
    key_scale = []
    for mutable in (80, 320):
        def run_keys(mutable=mutable):
            return run_seeds(SCALE_SEEDS, "stationary", SCALE_METHODS)
        cells, _unused = with_globals(rx.N_AGENTS, mutable, run_keys)
        key_scale.append({"agents": rx.N_AGENTS, "mutable": mutable, "derived": min(40, mutable), "cells": cells})
    # Drop seed vectors from the bulky stress/scale copies after tests are computed.
    def slim(cells):
        slimmed = {}
        for method, cell in cells.items():
            slimmed[method] = {
                key: value
                for key, value in cell.items()
                if not key.endswith("_seeds") or key in ("sp_seeds", "changed_seeds", "stable_seeds")
            }
        return slimmed
    payload = {
        "protocol": {
            "rate": RATE,
            "p_obs": POBS,
            "stationary_seeds": STATIONARY_SEEDS,
            "stress_seeds": STRESS_SEEDS,
            "scale_seeds": SCALE_SEEDS,
            "thresholds_locked": True,
            "primary": "sp",
            "pool_reads": "every generated document, not a local mailbox",
        },
        "account": account,
        "stationary": slim(stationary),
        "tests": tests,
        "lambda": lambda_curve(stationary),
        "selective": {
            "rule": "each agent uploads the first document it received on a key and any later document whose value differs",
            "cells": slim(selective_cells),
            "mean_uploads_per_agent_epoch": sum(selective_uploads) / float(len(selective_uploads)),
        },
        "stress": {
            regime: {"cells": slim(block["cells"]), "tests": block["tests"]}
            for regime, block in stress.items()
        },
        "scale_agents": [{"agents": row["agents"], "mutable": row["mutable"], "cells": slim(row["cells"])} for row in scale],
        "scale_keys": key_scale and [
            {"agents": row["agents"], "mutable": row["mutable"], "derived": row["derived"], "cells": slim(row["cells"])}
            for row in key_scale
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("WROTE", OUT, flush=True)
    for row in tests:
        if row["metric"] == "sp" and row["right"] in ("PoolMajority", "Central", "Majority"):
            print(
                "TEST",
                row["left"],
                "minus",
                row["right"],
                f"diff={row['mean_diff']:.4f}",
                f"sd={row['std_diff']:.4f}",
                f"p={row['p']:.4f}",
                f"n={row['n']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
