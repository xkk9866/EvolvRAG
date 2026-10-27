"""DynSim: a dynamic world in the source model of Dong, Berti-Equille and
Srivastava (2009b), written in the WikiCoach format.

The generator follows that paper's vocabulary, not TSSM's:
  world     every fact has its own volatility; a change installs a new value
  coverage  a source covers a fact with probability cov_s (fixed per fact)
  freshness after a change, a covering source adopts the new value after a
            geometric lag with mean 1/f_s - 1 epochs and reports its stale
            value until then (TSSM has no lag)
  exactness when it adopts, it adopts the true value with probability e_s(c),
            otherwise a popular wrong value (TSSM assumes uniform errors);
            e_s depends on the fact's domain c through a domain specialty
  copying   a copier adopts its parent's current value with a lag of its own,
            and otherwise behaves as an independent source
A source re-states its current value every epoch, as a web page does.

  python dynsim.py <regime> <seed> <out.json>
regimes: base, domain (strong domain specialty), cliques (40% copiers),
         laggy (slow freshness)
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

N_FACTS, N_SRC, N_DOM, T = 1000, 20, 8, 16


def generate(regime, seed):
    rng = random.Random(f"{regime}-{seed}")
    specialty = 2.0 if regime == "domain" else 0.4
    copy_share = 0.4 if regime == "cliques" else 0.12
    fresh_lo, fresh_hi = (0.15, 0.5) if regime == "laggy" else (0.35, 0.95)
    sources = []
    for s in range(N_SRC):
        home = rng.randrange(N_DOM)
        base = rng.uniform(0.5, 0.9)
        exa = [min(0.99, max(0.05, base + (specialty * 0.15 if c == home else -specialty * 0.08)
                             + rng.gauss(0, 0.03))) for c in range(N_DOM)]
        sources.append({"cov": rng.uniform(0.15, 0.5), "fresh": rng.uniform(fresh_lo, fresh_hi), "exa": exa,
                        "parent": None})
    for s in range(3, N_SRC):
        if rng.random() < copy_share:
            sources[s]["parent"] = rng.randrange(0, 3)          # copy one of three hubs
    facts, claims = [], []
    for k in range(N_FACTS):
        dom = k % N_DOM
        vol = rng.betavariate(1.2, 10.0)                       # per-fact volatility
        pool = [f"p{k}_{j}" for j in range(4)]                  # popular wrong values
        gold, z = [], f"f{k}_0"
        for t in range(1, T + 1):
            if t > 1 and rng.random() < vol:
                z = f"f{k}_{t}"
            gold.append(z)
        facts.append({"club": f"F{k}", "label": f"fact {k}", "country": f"D{dom}", "gold": gold})
        cover = [rng.random() < src["cov"] for src in sources]
        state = [None] * N_SRC                                  # current reported value
        pending = [None] * N_SRC                                # (adopt_time, value) of a pending update
        order = [s for s in range(N_SRC) if sources[s]["parent"] is None] + \
                [s for s in range(N_SRC) if sources[s]["parent"] is not None]
        for t in range(1, T + 1):
            truth = gold[t - 1]
            changed = t == 1 or gold[t - 2] != truth
            for s in order:
                if not cover[s]:
                    continue
                src = sources[s]
                if src["parent"] is not None and state[src["parent"]] is not None and rng.random() < 0.85:
                    target = state[src["parent"]]
                    if state[s] != target and (pending[s] is None or pending[s][1] != target):
                        pending[s] = (t + _lag(rng, 0.7), target)
                elif changed or state[s] is None:
                    v = truth if rng.random() < src["exa"][dom] else rng.choice(pool)
                    lag = 0 if t == 1 else _lag(rng, src["fresh"])
                    pending[s] = (t + lag, v)
                if pending[s] is not None and pending[s][0] <= t:
                    state[s] = pending[s][1]
                    pending[s] = None
                if state[s] is not None:
                    claims.append([k, state[s], f"s{s}", t, len(claims), state[s]])
    return {"langs": [f"s{s}" for s in range(N_SRC)], "boundaries": [f"t{t}" for t in range(1, T + 1)],
            "facts": facts, "claims": claims, "regime": regime, "seed": seed,
            "sources": [{k: v for k, v in src.items() if k != "exa"} | {"exa_mean": sum(src["exa"]) / N_DOM}
                        for src in sources]}


def _lag(rng, fresh):
    lag = 0
    while rng.random() > fresh and lag < 10:
        lag += 1
    return lag


if __name__ == "__main__":
    data = generate(sys.argv[1], int(sys.argv[2]))
    Path(sys.argv[3]).write_text(json.dumps(data), encoding="utf-8")
    hit = sum(data["facts"][k]["gold"][t - 1] == v for k, v, _s, t, _i, _x in data["claims"])
    print(sys.argv[1], sys.argv[2], "claims", len(data["claims"]), "agree", round(hit / len(data["claims"]), 3))
