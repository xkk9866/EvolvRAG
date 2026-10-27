"""Tolerance-based value equivalence for Stock and Flight.

Sources report the same quantity at different precision (12,345,678 vs
12.3M shares; 10:42 vs 10:45). Claimed values of one (item, day) are grouped
without looking at gold: values are visited from most to least supported, and
a value joins the first group whose representative is within tolerance
(relative 1% for numbers, 10 minutes for times; gates are exact). Every claim
is rewritten to its group's representative, so every method reads the same
canonical claims. For scoring only, the gold value is mapped to the
representative it falls within, if any.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime

REL = 0.01
MINUTES = 10


def _kind(slot):
    attr = slot.split("|")[1]
    if attr in ("actual_dep", "actual_arr"):
        return "time"
    if attr in ("dep_gate", "arr_gate"):
        return "gate"
    return "num"


def _close(kind, a, b):
    if kind == "gate":
        return a == b
    if kind == "num":
        x, y = float(a), float(b)
        return abs(x - y) <= REL * max(abs(x), abs(y), 1e-9)
    ta = datetime.strptime(a, "%Y-%m-%d %H:%M")
    tb = datetime.strptime(b, "%Y-%m-%d %H:%M")
    return abs((ta - tb).total_seconds()) <= 60 * MINUTES


def _gap(kind, a, b):
    if kind == "gate":
        return 0.0
    if kind == "num":
        return abs(float(a) - float(b))
    ta = datetime.strptime(a, "%Y-%m-%d %H:%M")
    tb = datetime.strptime(b, "%Y-%m-%d %H:%M")
    return abs((ta - tb).total_seconds())


def canonical(days):
    out = []
    for day in days:
        support = defaultdict(Counter)
        for slot, value, source, _org in day["claims"]:
            support[slot][value] += 1
        rep = {}
        reps_of = {}
        for slot, counts in support.items():
            kind = _kind(slot)
            reps = []
            for value, _n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
                target = next((r for r in reps if _close(kind, r, value)), None)
                if target is None:
                    reps.append(value)
                    target = value
                rep[(slot, value)] = target
            reps_of[slot] = reps
        claims = [(slot, rep[(slot, value)], source, org) for slot, value, source, org in day["claims"]]
        gold = {}
        for slot, g in day["gold"].items():
            kind = _kind(slot)
            match = [r for r in reps_of.get(slot, []) if _close(kind, r, g)]
            gold[slot] = min(match, key=lambda r: _gap(kind, r, g)) if match else g
        out.append({"day": day["day"], "claims": claims, "gold": gold})
    return out
