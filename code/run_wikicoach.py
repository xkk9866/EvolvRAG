"""WikiCoach evaluation: who coaches each club, quarter by quarter, from what
the language editions of Wikipedia say.

  python run_wikicoach.py tssm '<spec json>' <out.json> [view]
  python run_wikicoach.py baselines <out.json> [view]
  python run_wikicoach.py society '<spec json>' <out.json> <agents> <editions per agent> [relay]

At quarter t every method has read the claims of quarters 1..t and answers the
coach in office at the start of quarter t. A quarter is scored for a club when
Wikidata names exactly one coach for it. 'changed' are quarters within one
quarter of a coaching change, 'stable' the others; delay counts quarters from
a change to the first correct answer. Gold is read only for scoring.

view: "all" (every edition) or "k=<n>,seed=<s>": the reader sees n editions per
quarter, drawn at random each quarter (partial observation).
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("TESTIMONY_DATA", ROOT / "data" / "wikicoach" / "claims.json"))
NONE = "<none>"


def load():
    d = json.loads(DATA.read_text(encoding="utf-8"))
    countries = sorted({f["country"] for f in d["facts"]})
    ctx = {c: i for i, c in enumerate(countries)}
    return d, [ctx[f["country"]] for f in d["facts"]], len(countries)


def visible(d, view, agent=0):
    """Claims an observer receives: (fact, value, source, quarter)."""
    rows = [tuple(r[:4]) for r in d["claims"]]
    if view in (None, "all"):
        return rows
    opts = dict(kv.split("=") for kv in view.split(","))
    k, seed = int(opts["k"]), int(opts.get("seed", 0))
    langs = sorted({r[2] for r in rows})
    T = len(d["boundaries"])
    rng = random.Random(seed * 7919 + agent)
    shown = {t: set(rng.sample(langs, min(k, len(langs)))) for t in range(1, T + 1)}
    return [r for r in rows if r[2] in shown[r[3]]]


def score(d, answers):
    """answers[t-1][k] -> value. Returns metrics."""
    facts = d["facts"]
    T = len(d["boundaries"])
    hit = defaultdict(list)
    delays = []
    for k, f in enumerate(facts):
        g = f["gold"]
        for t in range(1, T):        # quarter index t (0-based), the first quarter is warm-up
            if g[t] is None:
                continue
            ok = int(answers[t][k] == g[t])
            hit["all"].append(ok)
            changed = g[t - 1] is not None and g[t - 1] != g[t]
            recent = changed or (t >= 2 and g[t - 2] is not None and g[t - 1] is not None and g[t - 2] != g[t - 1]
                                 and g[t - 1] == g[t])
            hit["changed" if recent else "stable"].append(ok)
            if changed:
                lag = None
                for u in range(t, T):
                    if g[u] != g[t]:
                        break
                    if answers[u][k] == g[t]:
                        lag = u - t
                        break
                delays.append(lag if lag is not None else sum(1 for u in range(t, T) if g[u] == g[t]))
            if g[t - 1] is not None and g[t - 1] != g[t]:
                hit["stale"].append(int(answers[t][k] == g[t - 1]))
    out = {k: float(np.mean(v)) for k, v in hit.items()}
    out["delay"] = float(np.mean(delays)) if delays else 0.0
    out["n"] = len(hit["all"])
    out["balanced"] = 0.5 * (out["changed"] + out["stable"])
    out["items"] = hit["all"]
    return out


# ----------------------------------------------------------------------------- TSSM
def run_tssm(d, ctx, n_ctx, spec_kw, view="all", agent=0, extra=None, log=False):
    import torch
    import tsm
    torch.set_num_threads(int(spec_kw.pop("_threads", 2)))
    keep_post = bool(spec_kw.pop("_post", False))
    posts = []
    spec = tsm.Spec(**{"schema": False, "span": "fact", **spec_kw})
    rows = visible(d, view, agent)
    T = len(d["boundaries"])
    n = len(d["facts"])
    sources = {}
    model = tsm.Model(spec, n_ctx, seed=agent)
    answers = []
    by_t = defaultdict(list)
    for k, v, s, t in rows:
        by_t[t].append((k, v, s, t))
    claims = []
    t0 = time.time()
    for t in range(1, T + 1):
        for k, v, s, tq in by_t[t]:
            claims.append((k, v, sources.setdefault(s, len(sources)), tq, tq))
        if extra is not None:
            for k, v, s, tq in extra(t):
                claims.append((k, v, sources.setdefault(s, len(sources)), tq, t))
        prob = tsm.Problem(claims, t, [NONE] * n, {}, None, lambda k: ctx[k], n, False, spec.clock, spec.echo, spec.span)
        th = model.fit(prob, first=t == 1)
        ans, _post = model.decide(prob, th)
        answers.append([ans.get(k) for k in range(n)])
        if keep_post:
            dist = model.posterior(prob, th)
            posts.append([{v: round(float(p), 4) for v, p in dist.get(k, {}).items() if not str(v).startswith("<")}
                          for k in range(n)])
        if log:
            print("quarter", t, "claims", len(claims), {k: round(x, 3) for k, x in model.scalars().items()},
                  "sec", round(time.time() - t0), flush=True)
    names = sorted(sources, key=sources.get)
    extra_out = {"cred": dict(zip(names, model.credibility([sources[s] for s in names], n_ctx).tolist())),
                 "echo": dict(zip(names, model.copy_propensity([sources[s] for s in names]))),
                 "scalars": model.scalars()}
    if keep_post:
        extra_out["post"] = posts
    return answers, extra_out


# ----------------------------------------------------------------------------- baselines
def as_days(d, rows):
    """Quarters in the day format of run_fusion_td: (slot, value, source, org)."""
    T = len(d["boundaries"])
    days = []
    for t in range(1, T + 1):
        gold = {str(k): f["gold"][t - 1] for k, f in enumerate(d["facts"]) if f["gold"][t - 1] is not None}
        claims = [(str(k), v, s, d["facts"][k]["country"]) for k, v, s, tq in rows if tq == t]
        # every fact is answered, even when gold is missing: keep a placeholder gold
        for k in range(len(d["facts"])):
            gold.setdefault(str(k), "<unscored>")
        days.append({"day": d["boundaries"][t - 1], "claims": claims, "gold": gold})
    return days


def run_baselines(d, view="all"):
    import run_fusion_td as td
    rows = visible(d, view)
    days = as_days(d, rows)
    T = len(days)
    n = len(d["facts"])
    out = {}
    # snapshot estimators, refit on all quarters up to t, answer quarter t
    names = ["majority", "truthfinder", "accu", "dawid", "crh", "invest", "two", "sums", "accucopy"]
    src_id = {}
    ans = {m: [] for m in names}
    for t in range(T):
        b = td.Batch(days, t, src_id)
        inv = [None] * b.n_cell
        for f, vals in enumerate(b.values):
            for v, c in vals.items():
                inv[c] = v
        for m in names:
            win = b.winners(td.METHODS[m](b)) if b.n_cell else []
            row = []
            for k in range(n):
                f = b.fact_id.get((str(k), t))
                row.append(inv[win[f]] if f is not None else None)
            ans[m].append(row)
    for m in names:
        out[m] = carry(ans[m])
    for decay in (0.5, 0.9):
        dyn = td.dynatd_answers(days, decay)
        out[f"dynatd{decay}"] = carry([[dyn[t].get(str(k)) for k in range(n)] for t in range(T)])
    # recency: the value of the edition that changed its claim most recently
    last = {}
    rec = []
    for t in range(1, T + 1):
        for k, v, s, tq in rows:
            if tq != t:
                continue
            prev = last.get((k, s))
            if prev is None or prev[0] != v:
                last[(k, s)] = (v, t)
        row = []
        for k in range(n):
            cands = [(tt, v) for (kk, s), (v, tt) in last.items() if kk == k]
            row.append(max(cands)[1] if cands else None)
        rec.append(row)
    out["newest"] = rec
    return out


def carry(grid):
    """A method that has no claim for a fact this quarter keeps its last answer."""
    out, last = [], {}
    for row in grid:
        new = []
        for k, v in enumerate(row):
            if v is None:
                v = last.get(k)
            else:
                last[k] = v
            new.append(v)
        out.append(new)
    return out


# ----------------------------------------------------------------------------- society
def run_society(d, ctx, n_ctx, spec_kw, agents, k, relay):
    """Each agent sees k editions per quarter; with relay it forwards to the
    others the claims it saw this quarter that support its own answer."""
    view = f"k={k},seed=1"
    inbox = defaultdict(list)       # (agent, quarter) -> forwarded claims
    seen = [visible(d, view, a) for a in range(agents)]
    by_at = [defaultdict(list) for _ in range(agents)]
    for a in range(agents):
        for r in seen[a]:
            by_at[a][r[3]].append(r)
    # agents run in lock step so that a forwarded claim arrives one quarter later
    import torch
    import tsm
    torch.set_num_threads(2)
    spec = tsm.Spec(**{"schema": False, "span": "fact", **spec_kw})
    T = len(d["boundaries"])
    n = len(d["facts"])
    models = [tsm.Model(spec, n_ctx, seed=a) for a in range(agents)]
    sources = [dict() for _ in range(agents)]
    claims = [[] for _ in range(agents)]
    grids = [[] for _ in range(agents)]
    sent = 0
    for t in range(1, T + 1):
        out_msgs = []
        for a in range(agents):
            for kk, v, s, tq in by_at[a][t]:
                claims[a].append((kk, v, sources[a].setdefault(s, len(sources[a])), tq, t))
            for kk, v, s, tq in inbox[(a, t)]:
                claims[a].append((kk, v, sources[a].setdefault(s, len(sources[a])), tq, t))
            prob = tsm.Problem(claims[a], t, [NONE] * n, {}, None, lambda q: ctx[q], n, False, spec.clock, spec.echo, spec.span)
            th = models[a].fit(prob, first=t == 1)
            ans, _ = models[a].decide(prob, th)
            grids[a].append([ans.get(q) for q in range(n)])
            if relay:
                out_msgs.append((a, [r for r in by_at[a][t] if ans.get(r[0]) == r[1]]))
        for a, msgs in out_msgs:
            sent += len(msgs)
            for b in range(agents):
                if b != a and t + 1 <= T:
                    inbox[(b, t + 1)].extend(msgs)
    scores = [score(d, g) for g in grids]
    res = {key: float(np.mean([s[key] for s in scores])) for key in ("all", "changed", "stable", "delay", "balanced")}
    res["messages"] = sent / float(agents * T)
    return res


def main():
    mode = sys.argv[1]
    d, ctx, n_ctx = load()
    if mode == "tssm":
        kw = json.loads(sys.argv[2])
        view = sys.argv[4] if len(sys.argv) > 4 else "all"
        t0 = time.time()
        answers, extra = run_tssm(d, ctx, n_ctx, dict(kw), view, log=True)
        res = score(d, answers)
        res.update(extra)
        res["seconds"] = time.time() - t0
        res["spec"] = kw
        res["view"] = view
        print("TSSM", kw, view, {k: round(res[k], 4) for k in ("all", "changed", "stable", "delay", "balanced")}, flush=True)
        Path(sys.argv[3]).write_text(json.dumps(res), encoding="utf-8")
    elif mode == "baselines":
        view = sys.argv[3] if len(sys.argv) > 3 else "all"
        grids = run_baselines(d, view)
        res = {}
        for m, g in grids.items():
            res[m] = score(d, g)
            print(m, view, {k: round(res[m][k], 4) for k in ("all", "changed", "stable", "delay", "balanced")}, flush=True)
        Path(sys.argv[2]).write_text(json.dumps(res), encoding="utf-8")
    elif mode == "society":
        kw = json.loads(sys.argv[2])
        agents, k = int(sys.argv[4]), int(sys.argv[5])
        relay = len(sys.argv) > 6 and sys.argv[6] == "relay"
        res = run_society(d, ctx, n_ctx, kw, agents, k, relay)
        res.update({"spec": kw, "agents": agents, "k": k, "relay": relay})
        print("society", res, flush=True)
        Path(sys.argv[3]).write_text(json.dumps(res), encoding="utf-8")


if __name__ == "__main__":
    main()
