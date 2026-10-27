"""Private memories: each agent revises from the passages delivered to it.

tsm_private  every agent runs the testimony model on its own mailbox, starting
             from its own stored beliefs.
tsm_relay    after deciding, an agent forwards the passages it received this
             epoch that support its own posterior mode, with issuer, reported
             time and the epoch it observed them. Receivers file them under
             that epoch. Nothing else is sent: no beliefs, weights, or lists.
The clearance baselines are run_decentral.run_private and run_relay.run_relay.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as rx
import run_fair as fair

KEYS = ("sp", "changed", "stable", "derived", "obsolete", "delay", "cfa", "epoch_cfa")


def slim(metrics):
    out = {k: metrics[k] for k in KEYS if k in metrics}
    out["balanced"] = (metrics["changed"] + metrics["stable"] + metrics["derived"]) / 3.0
    for k in ("messages", "coverage"):
        if k in metrics:
            out[k] = metrics[k]
    return out


def run_tsm_society(world, spec, relay):
    import tsm
    import run_tsm
    n = world.n_slots
    models = [tsm.Model(spec, run_tsm.N_CTX, seed=a) for a in range(rx.N_AGENTS)]
    claims = [[] for _ in range(rx.N_AGENTS)]
    pending = [[] for _ in range(rx.N_AGENTS)]
    answers = []
    sent_total = 0
    for epoch, docs in enumerate(world.visible, start=1):
        fresh = [[] for _ in range(rx.N_AGENTS)]
        for a in range(rx.N_AGENTS):
            claims[a].extend(pending[a])
            pending[a] = []
        for doc in docs:
            slot = fair.entity_of_text(doc["text"], world.names)
            if slot is None or not isinstance(doc["issuer"], int) or doc["issuer"] < 0:
                continue
            row = (slot, doc["value"], doc["issuer"], doc["time"], epoch)
            for a in doc["receivers"]:
                if 0 <= a < rx.N_AGENTS:
                    claims[a].append(row)
                    fresh[a].append(row)
        grid = []
        outgoing = []
        for a in range(rx.N_AGENTS):
            prob = tsm.Problem(claims[a], epoch, list(world.init_belief[a]), world.parent if spec.schema else {},
                               run_tsm.schema_fn, run_tsm.ctx_of, n, spec.schema, spec.clock, spec.echo)
            th = models[a].fit(prob, first=epoch == 1)
            ans, _post = models[a].decide(prob, th)
            grid.append([ans[k] for k in range(n)])
            if relay == "one":
                # one supporting passage per fact: the latest reported
                best = {}
                for r in fresh[a]:
                    if ans.get(r[0]) == r[1] and (r[0] not in best or r[3] > best[r[0]][3]):
                        best[r[0]] = r
                outgoing.append(list(best.values()))
            elif relay:
                outgoing.append([r for r in fresh[a] if ans.get(r[0]) == r[1]])
        answers.append(grid)
        if relay:
            for a, rows in enumerate(outgoing):
                for other in range(rx.N_AGENTS):
                    if other != a:
                        pending[other].extend(rows)
                sent_total += len(rows)
    counts = rx.empty_counts()
    rx.consume(counts, world, answers, 0.0)
    metrics = rx.pack_metrics(counts)
    metrics["messages"] = sent_total / float(rx.N_AGENTS * len(world.visible))
    return metrics


def job(args):
    seed, regime, name = args
    import torch
    torch.set_num_threads(1)
    import tsm
    world = fair.load_world(seed, regime)
    t0 = time.time()
    if name == "tsm_private":
        m = run_tsm_society(world, tsm.Spec(), relay=False)
    elif name == "tsm_relay":
        m = run_tsm_society(world, tsm.Spec(), relay=True)
    elif name == "tsm_relay1":
        m = run_tsm_society(world, tsm.Spec(), relay="one")
    elif name == "clear_private":
        import run_decentral as dec
        m = dec.run_private(world, dec.PRIVATE)
    elif name == "clear_relay":
        import run_relay
        m = run_relay.run_relay(world)
    else:
        raise ValueError(name)
    out = slim(m)
    out["seconds"] = time.time() - t0
    return {"seed": seed, "regime": regime, "name": name, "metrics": out}


def main():
    regimes = sys.argv[1].split(",")
    a, b = sys.argv[2].split("-")
    seeds = list(range(int(a), int(b) + 1))
    names = sys.argv[3].split(",")
    out_path = Path(sys.argv[4])
    tasks = [(s, r, nm) for nm in names for r in regimes for s in seeds]
    random.Random(0).shuffle(tasks)
    with Pool(min(int(os.environ.get("NPROC", "48")), len(tasks))) as pool:
        rows = pool.map(job, tasks, chunksize=1)
    out_path.write_text(json.dumps(rows), encoding="utf-8")
    print("wrote", out_path, len(rows), flush=True)


if __name__ == "__main__":
    main()
