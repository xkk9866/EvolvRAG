"""Objective at the generating parameters versus the fitted optimum (dev only)."""
import sys
import numpy as np, torch
sys.path.insert(0, ".")
import run_experiment as rx, run_fair as fair, tsm, run_tsm
torch.set_num_threads(2)
seed, regime, E, mem = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4] == "1"
world = fair.load_world(seed, regime)
spec = tsm.Spec(memory=mem)
claims = []
for e, docs in enumerate(world.visible[:E], start=1):
    claims.extend(run_tsm.world_claims(docs, world.names, e))
prob = tsm.Problem(claims, E, list(world.init_belief[0]), world.parent, run_tsm.schema_fn, run_tsm.ctx_of, world.n_slots, True, "arrival")
def evaluate(tag, m, th):
    with torch.no_grad():
        ll = float(m._forward(th, prob)["loglik"]); lp = float(m._log_prior(th))
    ans, _ = m.decide(prob, th)
    truth = world.timeline[E - 1]
    acc = np.mean([ans[k] == truth[k] for k in range(world.n_slots)])
    print(tag, "ll", round(ll, 1), "prior", round(lp, 1), "obj", round(ll + lp, 1), "acc", round(acc, 3), {k: round(v, 3) for k, v in m.scalars().items()})
# generating point
m = tsm.Model(spec, 8)
P = tsm.Params(8, 2, 7)
for i in prob.issuers:
    P.ensure(i)
    P.bias[i] = 4.0 if i < 5 else -4.0
    P.vec[i] = np.zeros(2); P.cop[i] = -4.0 if i < 10 else 4.0
P.beta0, P.gamma0 = (-4.0, 4.0) if regime == "sybil" else (0.0, -4.0)
P.lh, P.lmu, P.lom = tsm.logit(0.14), tsm.logit(0.89), tsm.logit(0.16)
m.params = P
th = m._tensors(prob, P)
evaluate("truth   ", m, th)
th2, v = m._optimize(prob, P, 100); m._store(prob, th2, P)
evaluate("truth+fit", m, th2)
m3 = tsm.Model(spec, 8)
th3 = m3.fit(prob, first=True)
evaluate("multistart", m3, th3)
