"""Testimony state-space model on the Stock and Flight source files.

Each (item, day) is a fact: prices, times, and gates are reported per day.
Days are linked through the sources: credibility, its context dependence
(attribute for Stock, airline for Flight), and echo propensity are fitted on
every claim up to the scored day. The day being scored is answered from that
fit. Gold values are read only for scoring. The baselines are the rules in
run_fusion.py, which read every item in the files; the model reads the claims
on items that appear in the gold files.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_fusion as fu
import tsm

ROOT = Path(__file__).resolve().parents[1]
NONE = "<none>"


def contained_reporters(day):
    """Sources whose whole report for the day is reproduced inside another
    source's report. A smaller report echoes the larger one; of two equal
    reports the later source id echoes the earlier. Snapshot claims arrive
    together, so containment is the only order available."""
    reports = {}
    for slot, value, source, _org in day["claims"]:
        reports.setdefault(source, set()).add((slot, value))
    names = sorted(reports)
    echo = set()
    for s in names:
        rs = reports[s]
        for t in names:
            if t == s:
                continue
            rt = reports[t]
            if len(rs) > len(rt):
                continue
            if len(rs) == len(rt) and t > s:
                continue
            if rs <= rt:
                echo.add(s)
                break
    return echo


def build(days, upto, ctx_names, echo_days=None):
    """Facts and claims for days 0..upto, restricted to items that have gold."""
    facts = []
    fact_id = {}
    claims = []
    ctx_of = []
    for d in range(upto + 1):
        day = days[d]
        gold_items = set(day["gold"])
        for slot, value, source, org in day["claims"]:
            if slot not in gold_items:
                continue
            key = (slot, d)
            if key not in fact_id:
                fact_id[key] = len(facts)
                facts.append(key)
                ctx_of.append(ctx_names.setdefault(org, len(ctx_names)))
            flag = None if echo_days is None else (1.0 if source in echo_days[d] else 0.0)
            claims.append((fact_id[key], value, source, 1, 1, d, flag))
    return facts, fact_id, claims, ctx_of


def run(name, days, spec, log_every=1):
    torch.set_num_threads(4)
    sources = {}
    ctx_names = {}
    model = None
    items = []
    changed_items = []
    t0 = time.time()
    echo_days = [contained_reporters(day) for day in days] if spec.copy else None
    if echo_days is not None:
        print(name, "containment echoes per day", [len(e) for e in echo_days], flush=True)
    for d in range(1, len(days)):
        facts, fact_id, claims, ctx_of = build(days, d, ctx_names, echo_days)
        enc = [(f, v, sources.setdefault(s, len(sources)), t, a, st, fl) for f, v, s, t, a, st, fl in claims]
        if model is None:
            model = tsm.Model(spec, n_ctx=64)
        prob = tsm.Problem(enc, 1, [NONE] * len(facts), {}, None, lambda k: ctx_of[k], len(facts), False, "arrival", spec.echo)
        th = model.fit(prob, first=d == 1)
        ans, _post = model.decide(prob, th)
        prev = days[d - 1]["gold"]
        hit = 0
        n = 0
        for slot, gold in days[d]["gold"].items():
            k = fact_id.get((slot, d))
            guess = ans.get(k) if k is not None else None
            ok = int(guess == gold)
            items.append(ok)
            hit += ok
            n += 1
            if slot in prev and prev[slot] != gold:
                changed_items.append(ok)
        if d % log_every == 0:
            print(name, days[d]["day"], "acc", round(hit / max(n, 1), 3), "facts", len(facts), "claims", len(claims),
                  {k: round(v, 3) for k, v in model.scalars().items()}, "sec", round(time.time() - t0), flush=True)
    names = sorted(sources, key=lambda s: sources[s])
    cred = model.credibility([sources[s] for s in names], max(len(ctx_names), 1))
    echo = model.copy_propensity([sources[s] for s in names])
    return {"em": float(np.mean(items)), "changed": float(np.mean(changed_items)), "items": items,
            "changed_items": changed_items, "scalars": model.scalars(), "n_sources": len(sources),
            "credibility": {s: cred[i].tolist() for i, s in enumerate(names)},
            "echo": {s: echo[i] for i, s in enumerate(names)}, "contexts": ctx_names}


def baselines(days):
    book = fu.Books()
    items = defaultdict(list)
    changed = defaultdict(list)
    for index, day in enumerate(days):
        if index == 0:
            book.update(day["claims"])
            continue
        scores = book.answer(day["claims"], day["gold"], days[index - 1]["gold"])
        book.update(day["claims"])
        for key in book.belief:
            items[key].extend(scores["_item"][key])
            changed[key].extend(scores["_changed_item"][key])
    return {key: {"em": float(np.mean(items[key])), "changed": float(np.mean(changed[key])),
                  "items": items[key], "changed_items": changed[key]} for key in items}


def main():
    which = sys.argv[1]
    kw = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "results" / f"fusion_tsm_{which}.json"
    days = fu.stock_days() if which == "stock" else fu.flight_days()
    if len(sys.argv) > 4 and sys.argv[4] == "tol":
        import fusion_tol
        days = fusion_tol.canonical(days)
    spec = tsm.Spec(**kw)
    res = {"spec": kw, "tsm": run(which, days, spec), "baselines": baselines(days)}
    for key, row in res["baselines"].items():
        print(which, key, round(row["em"], 4), round(row["changed"], 4), flush=True)
    print(which, "TSM", round(res["tsm"]["em"], 4), round(res["tsm"]["changed"], 4), flush=True)
    out.write_text(json.dumps(res), encoding="utf-8")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
