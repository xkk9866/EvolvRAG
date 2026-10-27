"""Test-set comparison on seeds 700-731. Every method reads the same world.

Usage:
  python run_main.py baselines <regimes> <seeds> <out.json>
  python run_main.py tsm <regimes> <seeds> <variants.json> <out.json>

Baselines are the rules already in this repository: the published EvolvRAG
count switch and its held-out and group variants, the incumbent gate, and the
truth-discovery family (majority, CRH, Sums, Investment, 2-Estimates, CATD,
TruthFinder, one-coin Dawid-Skene, temporal decay vote). Rules that need copy
collapse and the parent schema receive the same ones EvolvRAG v1 uses.
The testimony state-space model (TSSM) variants are given as Spec keyword sets.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as rx
import run_fair as fair

KEYS = ("sp", "changed", "stable", "derived", "obsolete", "delay", "cfa", "epoch_cfa")


def load(seed, regime, rate=None):
    """run_fair.load_world with an optional change rate."""
    if rate is None:
        return fair.load_world(seed, regime)
    saved = fair.RATE
    try:
        fair.RATE = rate
        return fair.load_world(seed, regime)
    finally:
        fair.RATE = saved


def slim(metrics):
    out = {k: metrics[k] for k in KEYS if k in metrics}
    out["balanced"] = (metrics["changed"] + metrics["stable"] + metrics["derived"]) / 3.0
    return out


def baseline_job(args):
    seed, regime, rate = args
    import run_mainstream as ms
    import run_td_baselines as td
    import run_unified_probe as uni
    import run_heldout as held
    import run_fill as fill
    world = load(seed, regime, rate)
    out = {}
    for name in ("newest", "majority", "crh", "gate", "hier", "global", "org"):
        cfg = next(c for c in fair.RULES if c["name"] == name)
        out[name] = slim(fair.run_cfg(load(seed, regime, rate), cfg))
    packed = ms.run_world(load(seed, regime, rate))
    for name in ("sums", "invest", "two", "catd", "decay"):
        out[name] = slim(packed[name])
    out["truthfinder"] = slim(uni.run_truthfinder(load(seed, regime, rate)))
    out["dawid"] = slim(td.run_dawid(load(seed, regime, rate)))
    out["loo"] = slim(held.run_loo(load(seed, regime, rate))["loo"])
    out["union"] = slim(fill.run_world(load(seed, regime, rate))["union"])
    return {"seed": seed, "regime": regime, "rate": rate, "methods": out}


def tsm_job(args):
    seed, regime, rate, name, kw = args
    import torch
    torch.set_num_threads(1)
    import tsm
    import run_tsm
    world = load(seed, regime, rate)
    spec_kw = dict(kw)
    oracle = spec_kw.pop("oracle", None)
    if oracle:
        spec_kw["oracle"] = make_oracle(regime)
    t0 = time.time()
    try:
        metrics = run_tsm.run_world(world, tsm.Spec(**spec_kw), trace=True)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return {"seed": seed, "regime": regime, "rate": rate, "name": name, "error": repr(exc)}
    out = slim(metrics)
    out["seconds"] = time.time() - t0
    out["scalars"] = metrics["scalars"]
    out["cred"] = metrics["cred"]
    return {"seed": seed, "regime": regime, "rate": rate, "name": name, "metrics": out}


def make_oracle(regime):
    """Generating credibility: an upper bound, not a method."""
    def rho(issuer, ctx):
        if issuer >= rx.N_REL + rx.N_NOISY:
            return 0.01
        reliable = issuer < rx.N_REL
        if regime == "topic" and ctx >= 4:
            reliable = not reliable
        return 0.99 if reliable else 0.01
    return rho


def run_pool(fn, tasks):
    n = int(os.environ.get("NPROC", "48"))
    random.Random(0).shuffle(tasks)
    with Pool(min(n, len(tasks))) as pool:
        return pool.map(fn, tasks, chunksize=1)


def parse_seeds(text):
    if "-" in text and "," not in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(",")]


def parse_regimes(text):
    out = []
    for item in text.split(","):
        if "@" in item:
            regime, rate = item.split("@")
            out.append((regime, float(rate)))
        else:
            out.append((item, None))
    return out


def main():
    mode = sys.argv[1]
    regimes = parse_regimes(sys.argv[2])
    seeds = parse_seeds(sys.argv[3])
    if mode == "baselines":
        out_path = Path(sys.argv[4])
        tasks = [(s, r, rate) for r, rate in regimes for s in seeds]
        rows = run_pool(baseline_job, tasks)
    else:
        variants = json.loads(Path(sys.argv[4]).read_text()) if sys.argv[4].endswith(".json") else json.loads(sys.argv[4])
        out_path = Path(sys.argv[5])
        tasks = [(s, r, rate, name, kw) for name, kw in variants.items() for r, rate in regimes for s in seeds]
        rows = run_pool(tsm_job, tasks)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows), encoding="utf-8")
    print("wrote", out_path, len(rows), flush=True)


if __name__ == "__main__":
    main()
