"""Contemporaneous witnesses.

A claim is evidence about the world at the time it was written. After exact
copies are treated as one witness, a time slice teaches a source only when
two or more independent witnesses agree and outnumber every other value.
Precision is the posterior probability of matching that contemporaneous
majority. An organization's posterior uses the source's overall posterior
as its prior. The current value is the value with the greatest positive
log-odds among witnesses who speak in the current slice. No second record
is selected, and a source is not removed by a single loss.

Development seeds 600-615 choose the decision. Test seeds are not read here
until that choice beats the existing clearance route on the same score.
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

ROOT = rx.ROOT
OUT_DIR = ROOT / "results"
MU = 0.40
KAPPA = 4.0
DEV = list(range(600, 616))
TEST = list(range(700, 732))
REGIMES = ("stationary", "topic", "open_rumor", "sybil")
MODES = ("map_hier", "map_glob", "latest_hier")


def posterior(hits, trials, center):
    if trials <= 0:
        return center
    return (center * KAPPA + hits) / (KAPPA + trials)


def log_odds(precision):
    precision = min(1.0 - 1e-6, max(1e-6, precision))
    return math.log(precision / (1.0 - precision))


class Memory:
    def __init__(self, hierarchical):
        self.hierarchical = hierarchical
        self.global_hits = defaultdict(float)
        self.global_n = defaultdict(float)
        self.local_hits = defaultdict(lambda: defaultdict(float))
        self.local_n = defaultdict(lambda: defaultdict(float))

    def precision(self, issuer, org):
        center = posterior(self.global_hits[issuer], self.global_n[issuer], MU)
        if not self.hierarchical:
            return center
        return posterior(self.local_hits[org][issuer], self.local_n[org][issuer], center)

    def observe(self, issuer, org, hit):
        self.global_n[issuer] += 1.0
        self.global_hits[issuer] += hit
        self.local_n[org][issuer] += 1.0
        self.local_hits[org][issuer] += hit


def testimonies(docs, names, find):
    """One independent witness per entity: its latest claim in this slice."""
    best = {}
    for doc in docs:
        slot = fair.entity_of_text(doc["text"], names)
        issuer = doc["issuer"]
        if slot is None or not isinstance(issuer, int) or issuer < 0:
            continue
        if find is not None:
            issuer = find(issuer)
        key = (slot, issuer)
        mark = (doc["time"], doc.get("seq", 0))
        held = best.get(key)
        if held is None or mark > held[0]:
            best[key] = (mark, doc["value"], issuer, slot)
    grouped = defaultdict(list)
    for mark, value, issuer, slot in best.values():
        grouped[slot].append((mark, value, issuer))
    return grouped


def teach(memory, grouped, names_org):
    for slot, rows in grouped.items():
        counts = defaultdict(int)
        for _mark, value, _issuer in rows:
            counts[value] += 1
        if not counts:
            continue
        ordered = sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)
        winner, top = ordered[0]
        second = ordered[1][1] if len(ordered) > 1 else 0
        if top < 2 or top <= second:
            continue
        org = names_org(slot)
        for _mark, value, issuer in rows:
            memory.observe(issuer, org, 1.0 if value == winner else 0.0)


def decide_slot(rows, belief, schema_v, precision_of):
    scored = []
    for mark, value, issuer in rows:
        weight = log_odds(precision_of(issuer))
        if weight > 0.0:
            scored.append((weight, mark, value, issuer))
    if not scored:
        if schema_v is not None:
            return schema_v
        return belief
    mass = defaultdict(float)
    latest = {}
    for weight, mark, value, _issuer in scored:
        mass[value] += weight
        if value not in latest or mark > latest[value]:
            latest[value] = mark
    return max(mass, key=lambda value: (mass[value], latest[value], value))


def decide_latest(rows, belief, schema_v, precision_of):
    best = None
    best_mark = None
    for mark, value, issuer in rows:
        if precision_of(issuer) <= 0.5:
            continue
        if best_mark is None or mark > best_mark:
            best = value
            best_mark = mark
    if best is None:
        if schema_v is not None:
            return schema_v
        return belief
    return best


def run_world(world, mode):
    hierarchical = mode != "map_glob"
    latest = mode == "latest_hier"
    memory = Memory(hierarchical)
    names = world.names
    order = list(world.mutable_ids) + list(world.stable_ids) + list(world.derived_ids)
    belief = list(world.init_belief[0])
    mail = []
    answers = []

    def org_of(slot):
        return slot % len(rx.ORGS)

    for epoch_docs in world.visible:
        mail.extend(epoch_docs)
        find = fair.copy_find(mail, names)
        grouped = testimonies(epoch_docs, names, find)
        row = list(belief)
        for slot in order:
            schema_v = fair.schema_value(world, row, slot)
            rows = grouped.get(slot, [])

            def precision_of(issuer, org=org_of(slot)):
                return memory.precision(issuer, org)

            if latest:
                row[slot] = decide_latest(rows, row[slot], schema_v, precision_of)
            else:
                row[slot] = decide_slot(rows, row[slot], schema_v, precision_of)
        belief = row
        answers.append([list(row) for _ in range(rx.N_AGENTS)])
        teach(memory, grouped, org_of)
    counts = rx.empty_counts()
    rx.consume(counts, world, answers, 0.0)
    return rx.pack_metrics(counts)


def balanced(metrics):
    return (metrics["changed"] + metrics["stable"] + metrics["derived"]) / 3.0


def evaluate(seeds, regimes, modes, with_hier=True):
    hier_cfg = next(cfg for cfg in fair.RULES if cfg["name"] == "hier")
    out = {mode: {regime: [] for regime in regimes} for mode in modes}
    if with_hier:
        out["hier"] = {regime: [] for regime in regimes}
    for regime in regimes:
        for seed in seeds:
            world = fair.load_world(seed, regime)
            parts = []
            for mode in modes:
                metrics = run_world(world, mode)
                out[mode][regime].append(metrics)
                parts.append(f"{mode}:{balanced(metrics):.3f}/{metrics['sp']:.3f}")
            if with_hier:
                metrics = fair.run_cfg(world, hier_cfg)
                out["hier"][regime].append(metrics)
                parts.append(f"hier:{balanced(metrics):.3f}/{metrics['sp']:.3f}")
            print(regime, seed, " ".join(parts), flush=True)
    return out


def mean_balanced(rows):
    values = [balanced(row) for row in rows]
    return sum(values) / len(values)


def summarize(raw):
    summary = {}
    for mode, by_regime in raw.items():
        regime_mean = {regime: mean_balanced(rows) for regime, rows in by_regime.items()}
        summary[mode] = {
            "by_regime": regime_mean,
            "mean": sum(regime_mean.values()) / len(regime_mean),
            "detail": {
                regime: {
                    "balanced": mean_balanced(rows),
                    "sp": sum(row["sp"] for row in rows) / len(rows),
                    "changed": sum(row["changed"] for row in rows) / len(rows),
                    "stable": sum(row["stable"] for row in rows) / len(rows),
                    "derived": sum(row["derived"] for row in rows) / len(rows),
                }
                for regime, rows in by_regime.items()
            },
        }
    return summary


def slim_raw(raw):
    keep = ("sp", "changed", "stable", "derived", "obsolete", "delay", "cfa")
    return {
        mode: {
            regime: [{key: row[key] for key in keep} for row in rows]
            for regime, rows in by_regime.items()
        }
        for mode, by_regime in raw.items()
    }


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "probe"
    if task == "probe":
        seeds = [600, 601]
        raw = evaluate(seeds, REGIMES, MODES, with_hier=True)
    elif task == "dev":
        raw = evaluate(DEV, REGIMES, MODES, with_hier=True)
    elif task == "test":
        raw = evaluate(TEST, REGIMES, MODES, with_hier=True)
    else:
        raise SystemExit("task must be probe, dev, or test")
    payload = {"task": task, "summary": summarize(raw), "seeds": slim_raw(raw)}
    path = OUT_DIR / f"witness_{task}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2), flush=True)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
