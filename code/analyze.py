"""Tables and paired tests from run_main.py outputs.

A seed's composite is the mean over regimes of (changed + retained + derived)/3.
Paired two-sided sign-flip tests are exact up to 16 seeds and use 100000
Monte Carlo draws beyond that (p is floored at 1e-5).
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

METRICS = ("balanced", "changed", "stable", "derived", "obsolete", "delay", "sp", "cfa")


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0


def signflip(diffs, draws=100000, seed=3157):
    n = len(diffs)
    obs = abs(sum(diffs))
    if obs <= 1e-12:
        return 1.0
    if n <= 16:
        count = 0
        for mask in range(1 << n):
            s = sum(d if (mask >> i) & 1 else -d for i, d in enumerate(diffs))
            if abs(s) + 1e-12 >= obs:
                count += 1
        return count / float(1 << n)
    rng = random.Random(seed)
    count = 1
    for _ in range(draws - 1):
        s = sum(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(s) + 1e-12 >= obs:
            count += 1
    return max(count / float(draws), 1e-5)


def load_rows(paths):
    """Return table[method][regime][seed] = metrics dict."""
    table = defaultdict(lambda: defaultdict(dict))
    for path in paths:
        rows = json.loads(Path(path).read_text())
        for row in rows:
            if "error" in row:
                continue
            regime = row["regime"] if row.get("rate") is None else f"{row['regime']}@{row['rate']}"
            if "methods" in row:
                for name, m in row["methods"].items():
                    table[name][regime][row["seed"]] = m
            else:
                table[row["name"]][regime][row["seed"]] = row["metrics"]
    return table


def composite(table, method, regimes):
    seeds = None
    for r in regimes:
        s = set(table[method][r])
        seeds = s if seeds is None else seeds & s
    return {seed: mean([table[method][r][seed]["balanced"] for r in regimes]) for seed in sorted(seeds or [])}


def paired(a, b):
    common = sorted(set(a) & set(b))
    diffs = [a[s] - b[s] for s in common]
    return {"n": len(diffs), "diff": mean(diffs), "sd": sd(diffs), "p": signflip(diffs)}


def summary(table, methods, regimes):
    out = {}
    for m in methods:
        row = {}
        for r in regimes:
            cells = list(table[m][r].values())
            row[r] = {k: mean([c.get(k) for c in cells]) for k in METRICS if cells and k in cells[0]}
            row[r]["n"] = len(cells)
        comp = composite(table, m, regimes)
        row["composite"] = mean(list(comp.values()))
        row["composite_sd"] = sd(list(comp.values()))
        out[m] = row
    return out


if __name__ == "__main__":
    paths = sys.argv[1].split(",")
    regimes = sys.argv[2].split(",")
    ref = sys.argv[3] if len(sys.argv) > 3 else None
    table = load_rows(paths)
    methods = sorted(table)
    summ = summary(table, methods, regimes)
    for m in sorted(methods, key=lambda x: -summ[x]["composite"]):
        cells = "  ".join(f"{r[:6]} {summ[m][r].get('balanced', float('nan')):.3f}" for r in regimes)
        line = f"{m:16s} comp {summ[m]['composite']:.3f}±{summ[m]['composite_sd']:.3f}  {cells}"
        if ref and ref in table and m != ref:
            t = paired(composite(table, ref, regimes), composite(table, m, regimes))
            line += f"   Δ({ref}-m) {t['diff']:+.3f} p={t['p']:.2g}"
        print(line)
