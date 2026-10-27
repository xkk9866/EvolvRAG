"""CEF baseline after Dong, Berti-Equille and Srivastava (VLDB 2009),
'Truth discovery and copying detection in a dynamic world', on WikiCoach-format
data. Sources are described by exactness (share of a source's adoptions that
name the value true when adopted) and freshness (inverse mean lag between a
true change and the source's adoption). Truth at each epoch is the value with
the largest vote, where a source votes log(n e/(1-e)) for the value it holds,
discounted by freshness when the value is an old restatement. Exactness,
freshness and truth are iterated to a fixed point on all epochs read so far.
Copy detection is the Accu-copy rule of run_fusion_td (accucopy row).

  python cef_baseline.py <out.json> [view]
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_wikicoach as wc  # noqa: E402


def cef(rows, T, n, iters=5):
    by = defaultdict(dict)                      # (k, s) -> {t: v}
    for k, v, s, t in rows:
        by[(k, s)][t] = v
    held = defaultdict(list)                    # (k, t) -> [(s, v, age)]
    for (k, s), seq in by.items():
        since, prev = None, None
        for t in sorted(seq):
            v = seq[t]
            if v != prev:
                since, prev = t, v
            held[(k, t)].append((s, v, t - since))
    srcs = {s for _k, s in by}
    e = {s: 0.8 for s in srcs}
    f = {s: 0.5 for s in srcs}
    truth = {}
    for _ in range(iters):
        for k in range(n):
            for t in range(1, T + 1):
                rows_t = held.get((k, t))
                if not rows_t:
                    continue
                nv = len({v for _s, v, _a in rows_t}) + 1
                score = defaultdict(float)
                for s, v, age in rows_t:
                    ee = min(max(e[s], 0.01), 0.99)
                    w = max(math.log(nv * ee / (1 - ee)), 0.0)
                    score[v] += w * (1.0 if age == 0 else f[s])
                truth[(k, t)] = max(sorted(score), key=score.get)
        # exactness: adoptions that match the truth at adoption time
        hit, tot, lags = defaultdict(float), defaultdict(float), defaultdict(list)
        for (k, s), seq in by.items():
            prev = None
            for t in sorted(seq):
                if seq[t] != prev:
                    tot[s] += 1
                    hit[s] += float(truth.get((k, t)) == seq[t] or any(truth.get((k, u)) == seq[t] for u in range(max(1, t - 2), t)))
                    prev = seq[t]
            # freshness: lag from a truth change to the source holding the new value
            for t in sorted(seq):
                if t > 1 and truth.get((k, t)) is not None and truth.get((k, t - 1)) not in (None, truth[(k, t)]):
                    new = truth[(k, t)]
                    lag = next((u - t for u in sorted(seq) if u >= t and seq[u] == new), None)
                    if lag is not None:
                        lags[s].append(lag)
        e = {s: (hit[s] + 1) / (tot[s] + 2) for s in srcs}
        f = {s: 1.0 / (1.0 + (sum(lags[s]) / len(lags[s]) if lags[s] else 1.0)) for s in srcs}
    return truth


def main():
    view = sys.argv[2] if len(sys.argv) > 2 else "all"
    d, _ctx, _n = wc.load()
    rows = wc.visible(d, view)
    T, n = len(d["boundaries"]), len(d["facts"])
    grid = []
    for t in range(1, T + 1):
        tr = cef([r for r in rows if r[3] <= t], t, n)
        grid.append([tr.get((k, t)) for k in range(n)])
    res = {"cef": wc.score(d, wc.carry(grid))}
    print("cef", {k: round(res["cef"][k], 4) for k in ("all", "changed", "stable", "delay", "balanced")}, flush=True)
    Path(sys.argv[1]).write_text(json.dumps(res), encoding="utf-8")


if __name__ == "__main__":
    main()
