"""Shared-index evaluation of the testimony state-space model.

Worlds are the ones run_fair.load_world builds, so every rule reads the same
document list. The model reads entity, value, issuer, and reported time from
each passage. It never reads recipient lists or source labels.
"""

from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as rx
import run_fair as fair
import tsm

N_CTX = len(rx.ORGS)


def schema_fn(value):
    return "EXPIRED" if value == "EXPIRED" else f"via-{value}"


def ctx_of(slot):
    return slot % N_CTX


def world_claims(docs, names, epoch):
    rows = []
    for doc in docs:
        slot = fair.entity_of_text(doc["text"], names)
        issuer = doc["issuer"]
        if slot is None or not isinstance(issuer, int) or issuer < 0:
            continue
        rows.append((slot, doc["value"], issuer, doc["time"], epoch))
    return rows


def run_world(world, spec, trace=False):
    model = tsm.Model(spec, N_CTX)
    claims = []
    answers = []
    memory = list(world.init_belief[0])
    n = world.n_slots
    log = []
    for epoch, docs in enumerate(world.visible, start=1):
        claims.extend(world_claims(docs, world.names, epoch))
        prob = tsm.Problem(claims, epoch, memory, world.parent if spec.schema else {}, schema_fn, ctx_of, n, spec.schema, spec.clock, spec.echo)
        th = model.fit(prob, first=epoch == 1)
        ans, _post = model.decide(prob, th)
        row = [ans[k] for k in range(n)]
        answers.append([list(row) for _ in range(rx.N_AGENTS)])
        if trace:
            log.append(model.scalars())
    counts = rx.empty_counts()
    rx.consume(counts, world, answers, 0.0)
    metrics = rx.pack_metrics(counts)
    metrics["balanced"] = (metrics["changed"] + metrics["stable"] + metrics["derived"]) / 3.0
    if trace:
        metrics["scalars"] = model.history
        metrics["cred"] = model.credibility(list(range(rx.N_REL + rx.N_NOISY)), N_CTX).tolist()
    return metrics


def job(args):
    seed, regime, spec_kw = args
    torch_threads()
    world = fair.load_world(seed, regime)
    t0 = time.time()
    try:
        out = run_world(world, tsm.Spec(**spec_kw), trace=True)
    except Exception as exc:  # keep the sweep alive; the failure is reported
        import traceback
        traceback.print_exc()
        out = {"balanced": float("nan"), "changed": float("nan"), "stable": float("nan"), "derived": float("nan"),
               "scalars": [{}], "cred": [], "error": repr(exc)}
    out["seconds"] = time.time() - t0
    out["seed"] = seed
    out["regime"] = regime
    return out


def torch_threads():
    import torch
    torch.set_num_threads(1)


def probe(worlds, variants):
    tasks = [(seed, regime, kw) for kw in variants for seed, regime in worlds]
    with Pool(min(len(tasks), int(os.environ.get('NPROC', '48')))) as pool:
        rows = pool.map(job, tasks, chunksize=1)
    hier = next(cfg for cfg in fair.RULES if cfg["name"] == "hier")
    base = {}
    for seed, regime in worlds:
        b = fair.run_cfg(fair.load_world(seed, regime), hier)
        base[(seed, regime)] = (b["changed"] + b["stable"] + b["derived"]) / 3
    i = 0
    for kw in variants:
        line = []
        for seed, regime in worlds:
            r = rows[i]; i += 1
            line.append(f"{regime[:4]}{seed}:{r['balanced']:.3f}/{base[(seed, regime)]:.3f}")
        mean = sum(rows[j]["balanced"] for j in range(i - len(worlds), i)) / len(worlds)
        print(json.dumps(kw), f"mean {mean:.3f}", " ".join(line), flush=True)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "probe":
    worlds = [(int(w.split(":")[0]), w.split(":")[1]) for w in sys.argv[2].split(",")]
    variants = json.loads(sys.argv[3])
    probe(worlds, variants)
    sys.exit(0)

if __name__ == "__main__":
    seeds = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [600]
    regimes = sys.argv[2].split(",") if len(sys.argv) > 2 else ["stationary", "topic", "open_rumor", "sybil"]
    spec_kw = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    tasks = [(s, r, spec_kw) for r in regimes for s in seeds]
    with Pool(min(len(tasks), 24)) as pool:
        rows = pool.map(job, tasks)
    hier = next(cfg for cfg in fair.RULES if cfg["name"] == "hier")
    for regime in regimes:
        sel = [r for r in rows if r["regime"] == regime]
        base = [fair.run_cfg(fair.load_world(r["seed"], regime), hier) for r in sel]
        def avg(key, xs):
            return sum(x[key] for x in xs) / len(xs)
        bal_b = sum((b["changed"] + b["stable"] + b["derived"]) / 3 for b in base) / len(base)
        print(f"{regime:11s} TSM bal {avg('balanced', sel):.3f} ch {avg('changed', sel):.3f} st {avg('stable', sel):.3f} de {avg('derived', sel):.3f} | hier bal {bal_b:.3f} ch {avg('changed', base):.3f} st {avg('stable', base):.3f} de {avg('derived', base):.3f} | sec {avg('seconds', sel):.1f}")
        for r, b in zip(sel, base):
            bb = (b["changed"] + b["stable"] + b["derived"]) / 3
            print(f"   seed {r['seed']} tsm {r['balanced']:.3f} hier {bb:.3f}", {k: round(v, 3) for k, v in r["scalars"][-1].items() if k != "T"})
