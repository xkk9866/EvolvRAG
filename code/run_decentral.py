"""Private mailboxes and a reliability broadcast, with the same clearance rule.

The central operator reads every document and does not read recipient lists.
Here each agent is delivered only the documents addressed to that agent. The
decision still does not read who else was addressed. Beliefs start from each
agent's own initial belief. The clearance floor, margin, schema, and copy
collapse are the already selected ones. Nothing is retuned.

The broadcast policy sends, once per epoch, each agent's society weight, its
organization weights, and the claim counts behind them. Recipients pool an
issuer by those counts. An agent with no claims does not move the pool. The
message does not contain a recipient list. A belief still changes only when
the agent holds a passage.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as rx
import run_fair as fair

ROOT = rx.ROOT
OUT = ROOT / "results" / "decentral.json"
REGIMES = fair.SELECT_REGIMES
TEST = list(fair.TEST)
HIER = next(cfg for cfg in fair.RULES if cfg["name"] == "hier")
NEWEST = next(cfg for cfg in fair.RULES if cfg["name"] == "newest")
PRIVATE = dict(HIER, corpus="local")
PRIVATE_NEW = dict(NEWEST, corpus="local")


def metrics_of(world, cfg):
    return fair.run_cfg(world, cfg)


def balanced(metrics):
    return (metrics["changed"] + metrics["stable"] + metrics["derived"]) / 3.0


def pool_issuer(entries):
    num = 0.0
    den = 0
    width = 0
    for weights, counts in entries:
        width = max(width, len(weights), (max(counts) + 1) if counts else 0)
    pooled = [rx.PRIOR] * width
    for issuer in range(width):
        num = 0.0
        den = 0
        for weights, counts in entries:
            n = counts.get(issuer, 0) if isinstance(counts, dict) else 0
            if n <= 0:
                continue
            num += n * fair.weight_of(weights, issuer)
            den += n
        if den:
            pooled[issuer] = num / float(den)
    return pooled


def run_share(world):
    names = world.names
    order = list(world.mutable_ids) + list(world.stable_ids) + list(world.derived_ids)
    beliefs = [list(row) for row in world.init_belief]
    books = [fair.TrustBook(HIER) for _ in range(rx.N_AGENTS)]
    mail = [[] for _ in range(rx.N_AGENTS)]
    answers = []
    held = 0
    asked = 0
    for epoch_docs in world.visible:
        fresh = [[] for _ in range(rx.N_AGENTS)]
        for doc in epoch_docs:
            for agent in doc["receivers"]:
                if 0 <= agent < rx.N_AGENTS:
                    mail[agent].append(doc)
                    fresh[agent].append(doc)
        grid = []
        usable = []
        for agent in range(rx.N_AGENTS):
            find = fair.copy_find(mail[agent], names)
            usable_agent = fair.collapsed(mail[agent], find)
            usable.append(usable_agent)
            buckets = fair.index_hits(usable_agent, names)
            row = list(beliefs[agent])
            for slot in order:
                asked += 1
                if buckets.get(slot):
                    held += 1
                schema_v = fair.schema_value(world, row, slot)
                row[slot] = fair.answer_slot(
                    HIER, buckets.get(slot, []), row[slot], books[agent].for_slot(slot), schema_v
                )
            beliefs[agent] = row
            grid.append(row)
            books[agent].update(fair.collapsed(fresh[agent], find), usable_agent, names)
        answers.append(grid)
        society = []
        for agent in range(rx.N_AGENTS):
            _rows, counts = fair.claim_rows(usable[agent], names)
            society.append((books[agent].plain, counts))
        pooled_society = pool_issuer(society)
        pooled_org = []
        for org in range(len(rx.ORGS)):
            entries = []
            for agent in range(rx.N_AGENTS):
                org_vec = books[agent].org[org] if books[agent].org is not None else []
                org_counts = books[agent].org_n[org] if books[agent].org_n is not None else {}
                entries.append((org_vec, org_counts))
            pooled_org.append(pool_issuer(entries))
        for agent in range(rx.N_AGENTS):
            books[agent].plain = list(pooled_society)
            books[agent].org = [list(row) for row in pooled_org]
    counts = rx.empty_counts()
    rx.consume(counts, world, answers, 0.0)
    packed = rx.pack_metrics(counts)
    packed["coverage"] = held / float(asked)
    return packed


def run_private(world, cfg):
    names = world.names
    order = list(world.mutable_ids) + list(world.stable_ids) + list(world.derived_ids)
    beliefs = [list(row) for row in world.init_belief]
    books = [fair.TrustBook(cfg) for _ in range(rx.N_AGENTS)]
    mail = [[] for _ in range(rx.N_AGENTS)]
    answers = []
    held = 0
    asked = 0
    for epoch_docs in world.visible:
        fresh = [[] for _ in range(rx.N_AGENTS)]
        for doc in epoch_docs:
            for agent in doc["receivers"]:
                if 0 <= agent < rx.N_AGENTS:
                    mail[agent].append(doc)
                    fresh[agent].append(doc)
        grid = []
        for agent in range(rx.N_AGENTS):
            find = fair.copy_find(mail[agent], names) if cfg.get("copy") else None
            usable = fair.collapsed(mail[agent], find)
            buckets = fair.index_hits(usable, names)
            row = list(beliefs[agent])
            use_schema = cfg.get("schema", False)
            for slot in order:
                asked += 1
                if buckets.get(slot):
                    held += 1
                schema_v = fair.schema_value(world, row, slot) if use_schema else None
                row[slot] = fair.answer_slot(
                    cfg, buckets.get(slot, []), row[slot], books[agent].for_slot(slot), schema_v
                )
            beliefs[agent] = row
            grid.append(row)
            books[agent].update(fair.collapsed(fresh[agent], find), usable, names)
        answers.append(grid)
    counts = rx.empty_counts()
    rx.consume(counts, world, answers, 0.0)
    packed = rx.pack_metrics(counts)
    packed["coverage"] = held / float(asked)
    return packed


def evaluate(seeds):
    stored = {name: {regime: [] for regime in REGIMES} for name in ("central", "private", "shared", "private_newest")}
    for regime in REGIMES:
        for seed in seeds:
            world = fair.load_world(seed, regime)
            stored["central"][regime].append(metrics_of(world, HIER))
            stored["private"][regime].append(run_private(world, PRIVATE))
            stored["shared"][regime].append(run_share(world))
            stored["private_newest"][regime].append(run_private(world, PRIVATE_NEW))
            print(
                regime,
                seed,
                " ".join(f"{name}:{balanced(stored[name][regime][-1]):.3f}" for name in stored),
                flush=True,
            )
    return stored


def family_mean(stored, name):
    per_seed = None
    for regime in REGIMES:
        rows = stored[name][regime]
        if per_seed is None:
            per_seed = [balanced(row) for row in rows]
        else:
            for i, row in enumerate(rows):
                per_seed[i] += balanced(row)
    per_seed = [value / float(len(REGIMES)) for value in per_seed]
    mu, sd = rx.mean_std(per_seed)
    return {"mean": mu, "std": sd, "seeds": per_seed}


def cell(stored, name, regime, metric):
    values = [row[metric] for row in stored[name][regime]]
    mu, sd = rx.mean_std(values)
    return {"mean": mu, "std": sd, "seeds": values}


def paired(left, right):
    diffs = [a - b for a, b in zip(left, right)]
    mu, sd = rx.mean_std(diffs)
    return {"mean": mu, "std": sd, "p": fair.perm_p(diffs), "n": len(diffs)}


def main():
    started = time.time()
    if "--smoke" in sys.argv:
        world = fair.load_world(399, "stationary")
        central = metrics_of(world, HIER)
        ref = fair.run_cfg(world, HIER)
        print("central", {k: round(central[k], 6) for k in ("sp", "changed", "stable", "derived")})
        print("ref", {k: round(ref[k], 6) for k in ("sp", "changed", "stable", "derived")})
        for label, metrics in (
            ("private", run_private(world, PRIVATE)),
            ("shared", run_share(world)),
            ("newest", run_private(world, PRIVATE_NEW)),
        ):
            print(label, {k: round(metrics[k], 3) for k in ("sp", "changed", "stable", "derived", "coverage")})
        return
    stored = evaluate(TEST)
    names = list(stored)
    summary = {}
    for name in names:
        block = {
            "balanced": family_mean(stored, name)["mean"],
            "balanced_std": family_mean(stored, name)["std"],
        }
        for regime in REGIMES:
            for metric in ("sp", "changed", "stable", "derived", "coverage"):
                if metric == "coverage" and name == "central":
                    continue
                if metric not in stored[name][regime][0]:
                    continue
                block[f"{regime}_{metric}"] = cell(stored, name, regime, metric)["mean"]
        summary[name] = block
    tests = []
    central_bal = family_mean(stored, "central")["seeds"]
    for name in ("private", "shared", "private_newest"):
        tests.append({"metric": "balanced", "other": name, **paired(central_bal, family_mean(stored, name)["seeds"])})
        for regime in REGIMES:
            tests.append(
                {
                    "metric": "sp",
                    "regime": regime,
                    "other": name,
                    **paired(cell(stored, "central", regime, "sp")["seeds"], cell(stored, name, regime, "sp")["seeds"]),
                }
            )
    tests.append(
        {
            "metric": "balanced",
            "other": "shared_minus_private",
            **paired(family_mean(stored, "shared")["seeds"], family_mean(stored, "private")["seeds"]),
        }
    )
    payload = {
        "protocol": {
            "seeds": TEST,
            "regimes": list(REGIMES),
            "rule": "selected clearance, not retuned",
            "private": "each agent revises only from delivered documents and its own initial belief",
            "shared": "broadcast society and organization weights with claim counts; pool by count; no recipient list in the message",
            "belief": "a shared weight does not create a passage the agent does not hold",
            "coverage": "fraction of agent-key-epochs whose mailbox contains a passage of that key",
        },
        "summary": summary,
        "tests": tests,
        "seconds": time.time() - started,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
