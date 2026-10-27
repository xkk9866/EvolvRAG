"""Credibility trajectories in the topic regime: smoothed (hindsight) credit
against credit frozen at arrival. One shared-index run per seed and spec;
after every epoch the fitted credibility map is recorded.

  python case_hindsight.py <seeds> <out.json>
"""

from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as rx
import run_fair as fair
import run_tsm
import tsm


def trace(args):
    seed, name, kw = args
    import torch
    torch.set_num_threads(1)
    world = fair.load_world(seed, "topic")
    spec = tsm.Spec(**kw)
    model = tsm.Model(spec, run_tsm.N_CTX)
    claims, maps, acc = [], [], []
    n = world.n_slots
    for epoch, docs in enumerate(world.visible, start=1):
        claims.extend(run_tsm.world_claims(docs, world.names, epoch))
        prob = tsm.Problem(claims, epoch, list(world.init_belief[0]), world.parent, run_tsm.schema_fn,
                           run_tsm.ctx_of, n, True, spec.clock, spec.echo)
        th = model.fit(prob, first=epoch == 1)
        ans, _ = model.decide(prob, th)
        maps.append(model.credibility(list(range(rx.N_REL + rx.N_NOISY)), run_tsm.N_CTX).tolist())
        truth = world.timeline[epoch - 1]
        hit = [ans[k] == truth[k] for k in world.mutable_ids if world.slot_org(k) >= 4] if hasattr(world, "slot_org") else \
              [ans[k] == truth[k] for k in world.mutable_ids if k % run_tsm.N_CTX >= 4]
        acc.append(sum(hit) / len(hit))
    return {"seed": seed, "name": name, "maps": maps, "acc_swapped": acc}


def main():
    seeds = [int(x) for x in sys.argv[1].split(",")]
    out = Path(sys.argv[2])
    specs = {"hindsight": {}, "online": {"hindsight": False}}
    tasks = [(s, n, kw) for s in seeds for n, kw in specs.items()]
    with Pool(min(len(tasks), 16)) as pool:
        rows = pool.map(trace, tasks, chunksize=1)
    out.write_text(json.dumps(rows), encoding="utf-8")
    print("wrote", out, len(rows))


if __name__ == "__main__":
    main()
