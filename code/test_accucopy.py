"""Sanity check: AccuCopy must beat majority when a bad source is copied.

Eight independent sources with accuracy 0.6 and one bad source with accuracy
0.3 whose claims three copiers repeat 90% of the time. Wrong values are drawn
from 10 alternatives. Copying should be detected and discounted.
"""

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_fusion_td as td


def world(seed, n_items=600):
    rng = random.Random(seed)
    claims, gold = [], {}
    for i in range(n_items):
        slot = f"i{i}|a"
        truth = "t"
        gold[slot] = truth
        wrong = [f"w{j}" for j in range(10)]
        for s in range(8):
            v = truth if rng.random() < 0.6 else rng.choice(wrong)
            claims.append((slot, v, f"ind{s}", "o"))
        bad = truth if rng.random() < 0.3 else rng.choice(wrong)
        claims.append((slot, bad, "bad", "o"))
        for c in range(3):
            v = bad if rng.random() < 0.9 else (truth if rng.random() < 0.5 else rng.choice(wrong))
            claims.append((slot, v, f"copy{c}", "o"))
    return [{"day": "d0", "claims": claims, "gold": gold}, {"day": "d1", "claims": claims, "gold": gold}]


if __name__ == "__main__":
    for seed in range(3):
        days = world(seed)
        b = td.Batch(days, 0, {})
        inv = [None] * b.n_cell
        for f, vals in enumerate(b.values):
            for v, c in vals.items():
                inv[c] = v
        res = {}
        for m in ("majority", "accu", "accucopy", "dawid", "crh"):
            win = b.winners(td.METHODS[m](b))
            res[m] = round(float(np.mean([inv[w] == "t" for w in win])), 3)
        print(seed, res)
