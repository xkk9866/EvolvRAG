"""RTM on data in the WikiCoach format (WikiCoach, DynSim).

  python run_rtm.py '<spec json>' <out.json> [view]

At epoch t the model has read the claims of epochs 1..t and answers each fact
with its value at t. Scoring is run_wikicoach.score, shared with all baselines.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_wikicoach as wc  # noqa: E402


def run(d, ctx, n_ctx, kw, view="all", agent=0, log=False, keep_post=False, extra=None):
    import torch
    import rtm
    torch.set_num_threads(int(kw.pop("_threads", 2)))
    keep_post = bool(kw.pop("_post", keep_post))
    spec = rtm.Spec(**kw)
    rows = wc.visible(d, view, agent)
    T, n = len(d["boundaries"]), len(d["facts"])
    src = {}
    by_t = defaultdict(list)
    for k, v, s, t in rows:
        by_t[t].append((k, v, src.setdefault(s, len(src)), t))
    model = rtm.Model(spec, n_ctx)
    claims, answers, posts, last = [], [], [], {}
    t0 = time.time()
    for t in range(1, T + 1):
        claims.extend(by_t[t])
        if extra is not None:
            claims.extend((k, v, src.setdefault(s, len(src)), t) for k, v, s in extra(t))
        prob = rtm.Problem(claims, t, lambda k: ctx[k], spec)
        th = model.fit(prob, first=t == 1)
        ans, post = model.decide(prob, th)
        for k, v in ans.items():
            last[k] = v
        answers.append([last.get(k) for k in range(n)])
        if keep_post:
            posts.append([{v: round(p, 4) for v, p in post.get(k, {}).items() if p > 1e-3} for k in range(n)])
        if log:
            print("epoch", t, "claims", prob.n, {a: round(b, 3) for a, b in model.scalars().items()},
                  "sec", round(time.time() - t0), flush=True)
    names = sorted(src, key=src.get)
    table = model.source_table([src[s] for s in names], n_ctx)
    out = {"sources": {s: table[src[s]] for s in names}, "scalars": model.scalars(), "history": model.history}
    if keep_post:
        out["post"] = posts
    return answers, out


def main():
    kw = json.loads(sys.argv[1])
    view = sys.argv[3] if len(sys.argv) > 3 else "all"
    d, ctx, n_ctx = wc.load()
    t0 = time.time()
    answers, extra = run(d, ctx, n_ctx, dict(kw), view, log=True)
    kw.pop("_post", None)
    res = wc.score(d, answers)
    res.update(extra)
    res["answers"] = answers
    res["seconds"] = time.time() - t0
    res["spec"] = kw
    res["view"] = view
    print("RTM", kw, view, {k: round(res[k], 4) for k in ("all", "changed", "stable", "delay", "balanced")}, flush=True)
    Path(sys.argv[2]).write_text(json.dumps(res), encoding="utf-8")


if __name__ == "__main__":
    main()
