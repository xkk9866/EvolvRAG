"""A consistent group may speak through a member the threshold has not cleared.

The count switch still adopts the latest passage from a source at or above
0.55. When no such passage exists, the largest perfectly consistent component
of the organization may revise, but only if that component's mean held-out
weight itself clears 0.55. Development seeds 800 and 801.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as rx
import run_fair as fair
import run_heldout as held
import run_consist as cons

REGIMES = ("stationary", "topic", "open_rumor", "sybil")
SEEDS = [800, 801]


def group_cleared(members, trust):
    if len(members) < 2:
        return False
    total = 0.0
    for issuer in members:
        total += fair.weight_of(trust, issuer)
    return total / len(members) + 1e-12 >= 0.55


def run_world(world):
    cfg = next(item for item in fair.RULES if item["name"] == "hier")
    names = world.names
    order = list(world.mutable_ids) + list(world.stable_ids) + list(world.derived_ids)
    modes = ("loo", "union", "window")
    belief = {key: list(world.init_belief[0]) for key in modes}
    answers = {key: [] for key in modes}
    mail = []
    society = held.rag_blank()
    n_org = len(rx.ORGS)
    eta = cfg.get("eta", 0.5)
    history = []
    stats = defaultdict(int)
    for epoch_docs in world.visible:
        mail.extend(epoch_docs)
        find = fair.copy_find(mail, names)
        usable = fair.collapsed(mail, find)
        fresh_c = fair.collapsed(epoch_docs, find)
        hist_rows, hist_max = held.claims_from(usable, names)
        fresh_rows, fresh_max = held.claims_from(fresh_c, names)
        history.append(fresh_rows)
        by_org = [[] for _ in range(n_org)]
        org_slots = [set() for _ in range(n_org)]
        for slot, value, issuer in hist_rows:
            by_org[slot % n_org].append((slot, value, issuer))
            org_slots[slot % n_org].add(slot)
        grouped = [[[row for row in rows if row[0] % n_org == g] for rows in history] for g in range(n_org)]
        epoch_slots = {slot for slot, _v, _i in fresh_rows}
        buckets = fair.index_hits(usable, names)
        pre = society
        blank = fair.pad(held.rag_blank(), hist_max + 1)
        soc_full = held.fit_vector(fresh_rows, pre, eta, max(len(pre) - 1, fresh_max))
        org_full = [held.fit_vector(by_org[g], blank, 1.0, hist_max) for g in range(n_org)]
        row = {key: list(belief[key]) for key in modes}
        for slot in order:
            g = slot % n_org
            if slot in epoch_slots:
                kept = [item for item in fresh_rows if item[0] != slot]
                ids = [fresh_max] + [issuer for _s, _v, issuer in kept]
                glob = held.fit_vector(kept, pre, eta, max(len(pre) - 1, max(ids)))
            else:
                glob = soc_full
            if slot in org_slots[g]:
                org_vec = held.fit_vector([item for item in by_org[g] if item[0] != slot], blank, 1.0, hist_max)
            else:
                org_vec = org_full[g]
            hits = buckets.get(slot, [])
            trust = held.trust_for(glob, org_vec, cfg, False)
            members = cons.consistent_members(grouped[g], slot)
            for key in modes:
                schema = fair.schema_value(world, row[key], slot)
                base = fair.answer_slot(cfg, hits, row[key][slot], trust, schema)
                if key == "loo":
                    row[key][slot] = base
                    continue
                cleared = fair.trusted_pick(hits, row[key][slot], trust, 0.55)[1]
                if key == "fill":
                    if cleared or not group_cleared(members, trust):
                        row[key][slot] = base
                        stats["keep"] += 1
                    else:
                        chosen = cons.latest_member(hits, members)
                        row[key][slot] = base if chosen is None else chosen
                        stats["fill"] += int(chosen is not None and chosen != base)
                elif key == "window":
                    allowed = set(members) if group_cleared(members, trust) else set()
                    mean = 0.0
                    if allowed:
                        mean = sum(fair.weight_of(trust, issuer) for issuer in allowed) / len(allowed)
                    usable = []
                    for hit in hits:
                        issuer = hit["issuer"]
                        weight = fair.weight_of(trust, issuer)
                        individual = weight is not None and weight >= 0.55
                        if individual or issuer in allowed:
                            score = weight if individual else mean
                            usable.append((score, hit))
                    pool = usable
                    if usable:
                        tmax = max(hit["time"] for _score, hit in usable)
                        recent = [(score, hit) for score, hit in usable if hit["time"] >= tmax - 1]
                        if recent:
                            pool = recent
                    if not pool:
                        row[key][slot] = schema if schema is not None else row[key][slot]
                    else:
                        score, hit = max(pool, key=lambda item: (item[0], item[1]["time"], item[1].get("seq", 0)))
                        row[key][slot] = hit["value"]
                    stats["window_delta"] += int(row[key][slot] != base)
                else:
                    allowed = set(members) if group_cleared(members, trust) else set()
                    best = None
                    best_key = None
                    for hit in hits:
                        issuer = hit["issuer"]
                        weight = fair.weight_of(trust, issuer)
                        if not ((weight is not None and weight >= 0.55) or issuer in allowed):
                            continue
                        stamp = (hit["time"], hit.get("seq", 0))
                        if best_key is None or stamp > best_key:
                            best = hit
                            best_key = stamp
                    if best is None:
                        row[key][slot] = schema if schema is not None else row[key][slot]
                    else:
                        row[key][slot] = best["value"]
                    stats["union_delta"] += int(row[key][slot] != base)
        for key in modes:
            belief[key] = row[key]
            answers[key].append([list(row[key]) for _ in range(rx.N_AGENTS)])
        society = soc_full
    out = {}
    for key, grid in answers.items():
        counts = rx.empty_counts()
        rx.consume(counts, world, grid, 0.0)
        out[key] = rx.pack_metrics(counts)
    out["_stats"] = dict(stats)
    return out


def balanced(m):
    return (m["changed"] + m["stable"] + m["derived"]) / 3.0


def main():
    import json
    task = sys.argv[1] if len(sys.argv) > 1 else "probe"
    seeds = {"probe": SEEDS, "dev": list(range(800, 816)), "compare": list(range(700, 732)), "test": list(range(900, 932))}[task]
    out = rx.ROOT / "results" / f"fill_{task}.json"
    stored = {key: {regime: [] for regime in REGIMES} for key in ("loo", "union", "window")}
    done = {regime: 0 for regime in REGIMES}
    if out.exists():
        previous = json.loads(out.read_text(encoding="utf-8"))
        stored = previous.get("seeds", stored)
        for regime in REGIMES:
            done[regime] = len(stored.get("loo", {}).get(regime, []))
    totals = defaultdict(lambda: defaultdict(list))
    for regime in REGIMES:
        extra = defaultdict(int)
        for index, seed in enumerate(seeds):
            if index < done[regime]:
                continue
            packed = run_world(fair.load_world(seed, regime))
            for k, v in packed["_stats"].items():
                extra[k] += v
            bits = []
            for key in ("loo", "union", "window"):
                score = balanced(packed[key])
                totals[key][regime].append(score)
                stored[key][regime].append(
                    {k: packed[key][k] for k in ("sp", "changed", "stable", "derived", "obsolete", "delay", "cfa")}
                )
                bits.append(f"{key}:{score:.3f}/{packed[key]['changed']:.3f}/{packed[key]['stable']:.3f}/{packed[key]['derived']:.3f}")
            print(regime, seed, " ".join(bits), flush=True)
            out.write_text(json.dumps({"task": task, "partial": True, "seeds": stored}), encoding="utf-8")
        print(regime, dict(extra), flush=True)
    summary = {}
    for key, regimes in stored.items():
        by = {}
        for regime, rows in regimes.items():
            if rows:
                by[regime] = sum(balanced(row) for row in rows) / len(rows)
        if by:
            summary[key] = {"by_regime": by, "mean": sum(by.values()) / len(by)}
            print(key, {r: round(v, 3) for r, v in by.items()}, "ALL", round(summary[key]["mean"], 3), flush=True)
    out.write_text(json.dumps({"task": task, "summary": summary, "seeds": stored}), encoding="utf-8")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
