"""Parameter recovery when the model is true (an inference check, not a benchmark).

Facts follow the hazard chain, sources testify with rho_s(c) = sigma(b_s + u_s w_c)
or echo one of three hub sources. We fit TSSM and report how well the learned
credibility map and the answers match the generating ones, for several numbers
of sources, with and without random restarts and the sincerity condition.

  python sim_recovery.py <n_sources> <seed> <spec name or json> <out.json>
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

N_FACTS, N_CTX, T, H, COVER = 2000, 10, 12, 0.10, 0.25


SPECS = {"full": {}, "norestart": {"restarts": 0}, "sincere_norestart": {"sincere": True, "restarts": 0},
         "sincere": {"sincere": True}}


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def generate(n_src, seed):
    rng = random.Random(seed)
    b = [rng.gauss(1.0, 1.5) for _ in range(n_src)]
    u = [rng.gauss(0.0, 1.5) for _ in range(n_src)]
    w = [rng.gauss(0.0, 1.0) for _ in range(N_CTX)]
    hubs = [0, 1, 2]
    echoer = [s >= 3 and rng.random() < 0.3 for s in range(n_src)]
    ctx = [k % N_CTX for k in range(N_FACTS)]
    truth = []
    for k in range(N_FACTS):
        z = ["v0"]
        for t in range(1, T + 1):
            z.append(z[-1] if rng.random() >= H else f"v{t}x{rng.randrange(3)}")
        truth.append(z)
    claims = []
    stamp = 0
    for t in range(1, T + 1):
        said = {}
        order = hubs + [s for s in range(n_src) if s not in hubs]
        for s in order:
            for k in range(N_FACTS):
                if rng.random() >= COVER:
                    continue
                if echoer[s] and rng.random() < 0.8:
                    src = rng.choice(hubs)
                    if (src, k) not in said:
                        continue
                    v = said[(src, k)]
                else:
                    rho = sigmoid(b[s] + u[s] * w[ctx[k]])
                    v = truth[k][t] if rng.random() < rho else f"w{rng.randrange(6)}"
                said[(s, k)] = v
                claims.append((k, v, s, t, t, stamp))
            stamp += 1
    rho = np.array([[sigmoid(b[s] + u[s] * w[c]) for c in range(N_CTX)] for s in range(n_src)])
    return claims, truth, ctx, rho, echoer


def main():
    import torch
    import tsm
    torch.set_num_threads(4)
    n_src, seed = int(sys.argv[1]), int(sys.argv[2])
    kw = SPECS[sys.argv[3]] if sys.argv[3] in SPECS else json.loads(sys.argv[3])
    claims, truth, ctx, rho, echoer = generate(n_src, seed)
    spec = tsm.Spec(**{"schema": False, **kw})
    model = tsm.Model(spec, N_CTX, seed=seed)
    t0 = time.time()
    prob = tsm.Problem(claims, T, ["<none>"] * N_FACTS, {}, None, lambda k: ctx[k], N_FACTS, False,
                       "arrival", spec.echo, spec.span)
    th = model.fit(prob, first=True)
    ans, _ = model.decide(prob, th)
    acc = float(np.mean([ans[k] == truth[k][T] for k in range(N_FACTS)]))
    est = model.credibility(list(range(n_src)), N_CTX)
    honest = [s for s in range(n_src) if not echoer[s]]
    corr = float(np.corrcoef(est[honest].ravel(), rho[honest].ravel())[0, 1])
    mae = float(np.abs(est[honest] - rho[honest]).mean())
    out = {"n_src": n_src, "seed": seed, "spec": kw, "acc": acc, "corr": corr, "mae": mae,
           "h": model.scalars()["h"], "seconds": time.time() - t0, "claims": len(claims),
           "objective": model.history[-1]["objective"]}
    print(json.dumps(out), flush=True)
    Path(sys.argv[4]).write_text(json.dumps(out), encoding="utf-8")


if __name__ == "__main__":
    main()
