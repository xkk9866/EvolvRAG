from pathlib import Path
"""Aggregate DynSim results: RTM, its ablations, TSSM and baselines. Paired
sign-flip tests over seeds on the balanced score (mean of changed and stable)."""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

R = str(Path(__file__).resolve().parents[1] / "results")
REG = ["base", "domain", "cliques", "laggy"]
KEYS = ["all", "changed", "stable", "delay", "balanced"]
tab = defaultdict(dict)          # (method, regime) -> seed -> metrics
for f in glob.glob(f"{R}/dynsim/*.json"):
    v, r, s = os.path.basename(f)[:-5].split("_")
    d = json.load(open(f))
    if v == "base":
        for m, x in d.items():
            tab[(m, r)][s] = x
    elif v == "tssm":
        tab[("tssm", r)][s] = d
for f in [x for x in glob.glob(f"{R}/rtm/*.json") if os.path.basename(x).count("_") == 2]:
    v, r, s = os.path.basename(f)[:-5].split("_")
    d = json.load(open(f))
    if v == "cef":
        d = d["cef"]
    tab[(v, r)][s] = {k: d[k] for k in KEYS + ["items"] if k in d}


def signflip(a, b, n=20000, rng=np.random.default_rng(0)):
    d = np.asarray(a) - np.asarray(b)
    obs = abs(d.mean())
    flips = rng.choice([-1, 1], size=(n, len(d)))
    return float(((np.abs((flips * d).mean(1))) >= obs - 1e-12).mean())


out = {}
for (m, r), seeds in sorted(tab.items()):
    out.setdefault(m, {})[r] = {k: float(np.mean([x[k] for x in seeds.values()])) for k in KEYS}
    out[m][r]["n"] = len(seeds)
# tests: rtm vs every method on shared seeds, per regime and pooled
tests = {}
for m in out:
    if m == "rtm":
        continue
    a, b = [], []
    for r in REG:
        shared = sorted(set(tab[("rtm", r)]) & set(tab[(m, r)]))
        a += [tab[("rtm", r)][s]["balanced"] for s in shared]
        b += [tab[(m, r)][s]["balanced"] for s in shared]
    if a:
        tests[m] = {"diff": float(np.mean(a) - np.mean(b)), "p": signflip(a, b), "n": len(a),
                    "wins": int(sum(x > y for x, y in zip(a, b)))}
json.dump({"mean": out, "tests": tests}, open(f"{R}/rtm_summary.json", "w"), indent=1)
for m in sorted(out, key=lambda m: -np.mean([out[m].get(r, {}).get("balanced", 0) for r in REG])):
    row = " ".join(f"{r[:4]}:{out[m][r]['all']:.3f}/{out[m][r]['changed']:.3f}/{out[m][r]['n']}" if r in out[m] else f"{r[:4]}:-" for r in REG)
    print(f"{m:12s} {row}  {tests.get(m, '')}")
