"""Single-passage record choice, predeclared before seeds 800-815 are read.

The memory adopts one passage. Each held-out record therefore proposes the
latest passage it clears, and the adopted passage is the proposal whose
issuer has the higher weight on its own record. A count of cleared issuers
is not the comparison. Soft routing is the locked temperature-1 mixture of
the same two held-out records. Tau, margin, eta, copy collapse, and the
schema stay at the values already fixed. Development is seeds 800-815.
Test seeds 900-931 stay unread until that comparison is finished.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as rx
import run_fair as fair
import run_heldout as held

DEV = list(range(800, 816))
TEST = list(range(900, 932))
REGIMES = ("stationary", "topic", "open_rumor", "sybil")
TAU = 0.55
TEMPERATURE = 1.0


def latest_cleared(hits, vec):
    best = None
    best_key = None
    best_w = None
    for hit in hits:
        weight = fair.weight_of(vec, hit["issuer"])
        if weight is None or weight + 1e-9 < TAU:
            continue
        key = (hit["time"], hit.get("seq", 0))
        if best_key is None or key > best_key:
            best = hit
            best_key = key
            best_w = weight
    return best, best_w


def proposal_value(hits, belief, glob, org_vec, schema_v):
    soc, sw = latest_cleared(hits, glob)
    org, ow = latest_cleared(hits, org_vec)
    saw = soc is not None or org is not None
    if not saw:
        if schema_v is not None:
            return schema_v
        return belief
    if org is None:
        return soc["value"]
    if soc is None:
        return org["value"]
    if ow > sw + 1e-12:
        return org["value"]
    if sw > ow + 1e-12:
        return soc["value"]
    org_key = (org["time"], org.get("seq", 0))
    soc_key = (soc["time"], soc.get("seq", 0))
    if org_key > soc_key:
        return org["value"]
    return soc["value"]


def mixed_trust(glob, org_vec):
    width = max(len(glob), len(org_vec))
    soc_n = 0
    org_n = 0
    for issuer in range(width):
        if fair.weight_of(glob, issuer) + 1e-9 >= TAU:
            soc_n += 1
        if fair.weight_of(org_vec, issuer) + 1e-9 >= TAU:
            org_n += 1
    share = 1.0 / (1.0 + math.exp(-(soc_n - org_n) / TEMPERATURE))
    mixed = []
    for issuer in range(width):
        society = fair.weight_of(glob, issuer)
        local = fair.weight_of(org_vec, issuer)
        mixed.append(share * society + (1.0 - share) * local)

    def trust(issuer, mixed=mixed):
        return fair.weight_of(mixed, issuer)

    return trust


def run_modes(world):
    cfg = next(item for item in fair.RULES if item["name"] == "hier")
    names = world.names
    order = list(world.mutable_ids) + list(world.stable_ids) + list(world.derived_ids)
    keys = ("loo", "proposal", "soft")
    belief = {key: list(world.init_belief[0]) for key in keys}
    mail = []
    society = held.rag_blank()
    answers = {key: [] for key in keys}
    n_org = len(rx.ORGS)
    eta = cfg.get("eta", 0.5)
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
        pre = society
        blank = fair.pad(held.rag_blank(), hist_max + 1)
        soc_full = held.fit_vector(fresh_rows, pre, eta, max(len(pre) - 1, fresh_max))
        org_full = [held.fit_vector(by_org[org], blank, 1.0, hist_max) for org in range(n_org)]
        row = {key: list(belief[key]) for key in keys}
        for slot in order:
            org = slot % n_org
            if slot in epoch_slots:
                kept_epoch = [item for item in fresh_rows if item[0] != slot]
                epoch_ids = [fresh_max]
                epoch_ids.extend(issuer for _s, _v, issuer in kept_epoch)
                glob = held.fit_vector(kept_epoch, pre, eta, max(len(pre) - 1, max(epoch_ids)))
            else:
                glob = soc_full
            if slot in org_slots[org]:
                kept_org = [item for item in by_org[org] if item[0] != slot]
                org_vec = held.fit_vector(kept_org, blank, 1.0, hist_max)
            else:
                org_vec = org_full[org]
            hits = buckets.get(slot, [])
            schema = {
                key: fair.schema_value(world, row[key], slot) if cfg.get("schema") else None
                for key in keys
            }
            row["loo"][slot] = fair.answer_slot(
                cfg, hits, row["loo"][slot], held.trust_for(glob, org_vec, cfg, False), schema["loo"]
            )
            row["proposal"][slot] = proposal_value(
                hits, row["proposal"][slot], glob, org_vec, schema["proposal"]
            )
            row["soft"][slot] = fair.answer_slot(
                cfg, hits, row["soft"][slot], mixed_trust(glob, org_vec), schema["soft"]
            )
        for key in keys:
            belief[key] = row[key]
            answers[key].append([list(row[key]) for _ in range(rx.N_AGENTS)])
        society = soc_full
    packed = {}
    for key, grid in answers.items():
        counts = rx.empty_counts()
        rx.consume(counts, world, grid, 0.0)
        packed[key] = rx.pack_metrics(counts)
    return packed


def balanced(metrics):
    return (metrics["changed"] + metrics["stable"] + metrics["derived"]) / 3.0


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "probe"
    if task == "probe":
        seeds = [800, 801]
    elif task == "dev":
        seeds = DEV
    elif task == "compare":
        seeds = list(range(700, 732))
    elif task == "test":
        seeds = TEST
    else:
        raise SystemExit("task must be probe, dev, or test")
    stored = {name: {regime: [] for regime in REGIMES} for name in ("loo", "proposal", "soft")}
    summary = {}
    for regime in REGIMES:
        for seed in seeds:
            world = fair.load_world(seed, regime)
            packed = run_modes(world)
            bits = []
            for name in stored:
                stored[name][regime].append(
                    {key: packed[name][key] for key in ("sp", "changed", "stable", "derived", "obsolete", "delay", "cfa")}
                )
                bits.append(f"{name}:{balanced(packed[name]):.3f}/{packed[name]['sp']:.3f}")
            print(regime, seed, " ".join(bits), flush=True)
    for name in stored:
        by = {}
        for regime, rows in stored[name].items():
            by[regime] = sum(balanced(row) for row in rows) / len(rows)
        summary[name] = {"by_regime": by, "mean": sum(by.values()) / len(by)}
    out = rx.ROOT / "results" / f"proposal_{task}.json"
    out.write_text(json.dumps({"task": task, "summary": summary, "seeds": stored}), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
