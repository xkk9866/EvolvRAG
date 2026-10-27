"""Diagnose: at the final epoch, compare objectives of random restarts with the
objective at labelled credibility. Development seeds only."""
import sys, json, math
import numpy as np, torch
sys.path.insert(0, ".")
import run_experiment as rx, run_fair as fair, tsm, run_tsm

def final_problem(world, spec):
    claims = []
    for docs in world.visible:
        claims.extend(run_tsm.world_claims(docs, world.names))
    T = len(world.visible)
    return tsm.Problem(claims, T, list(world.init_belief[0]), world.parent if spec.schema else {}, run_tsm.schema_fn, run_tsm.ctx_of, world.n_slots, spec.schema)

def acc_at(world, prob, ans):
    T = prob.T
    truth = world.timeline[T-1]
    return sum(ans[k] == truth[k] for k in range(world.n_slots)) / world.n_slots

def objective(model, prob, th):
    with torch.no_grad():
        return float(model._forward(th, prob)["loglik"] + model._log_prior(th))

seed, regime = int(sys.argv[1]), sys.argv[2]
world = fair.load_world(seed, regime)
spec = tsm.Spec(steps_first=400)
prob = final_problem(world, spec)
rows = []
for r in range(6):
    model = tsm.Model(spec, 8)
    rng = np.random.RandomState(100 + r)
    model.params.ctx = rng.normal(0, 0.5, size=model.params.ctx.shape)
    model.params._rng = np.random.RandomState(200 + r)
    model.params.beta0 = rng.normal(0, 1)
    th = model.fit(prob, first=True)
    ans, _ = model.decide(prob, th)
    rows.append((objective(model, prob, th), acc_at(world, prob, ans), model.scalars()))
# labelled init: reliable sources high, noisy low (per regime)
model = tsm.Model(spec, 8)
for i in prob.issuers:
    model.params.ensure(i)
    if regime == "topic":
        model.params.bias[i] = 0.0
        model.params.vec[i] = np.array([3.0 if i < 5 else -3.0, 0.0])
    else:
        rel = i < 5
        model.params.bias[i] = 3.0 if rel else -3.0
        model.params.vec[i] = np.zeros(2)
if regime == "topic":
    model.params.ctx = np.array([[1.0, 0.0] if c < 4 else [-1.0, 0.0] for c in range(8)])
th = model.fit(prob, first=True)
ans, _ = model.decide(prob, th)
rows.append((objective(model, prob, th), acc_at(world, prob, ans), model.scalars()))
for i, (obj, acc, sc) in enumerate(rows):
    print(("label " if i == len(rows) - 1 else f"rst{i}  "), round(obj, 1), round(acc, 3), {k: round(v, 3) for k, v in sc.items()})
