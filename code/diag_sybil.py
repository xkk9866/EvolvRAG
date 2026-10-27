import sys, json
import numpy as np, torch
sys.path.insert(0, ".")
import run_experiment as rx, run_fair as fair, tsm, run_tsm
torch.set_num_threads(1)
seed, E = int(sys.argv[1]), int(sys.argv[2])
world = fair.load_world(seed, "sybil")
spec = tsm.Spec()
claims = []
for e, docs in enumerate(world.visible[:E], start=1):
    claims.extend(run_tsm.world_claims(docs, world.names, e))
prob = tsm.Problem(claims, E, list(world.init_belief[0]), world.parent, run_tsm.schema_fn, run_tsm.ctx_of, world.n_slots, True, "arrival")
print("issuers", len(prob.issuers), "rows", len(prob.rows), "clone rows", sum(r[2] >= 10 for r in prob.rows), "clone q>0", sum(r[2] >= 10 and r[4] > 0 for r in prob.rows), "orig q>0", sum(r[2] < 10 and r[4] > 0 for r in prob.rows))
def show(tag, model, th):
    with torch.no_grad():
        ll = float(model._forward(th, prob)["loglik"]); lp = float(model._log_prior(th))
    ans, _ = model.decide(prob, th)
    truth = world.timeline[E - 1]
    acc = np.mean([ans[k] == truth[k] for k in range(world.n_slots)])
    sc = model.scalars()
    print(tag, "loglik", round(ll, 1), "logprior", round(lp, 1), "acc", round(acc, 3), {k: round(v, 3) for k, v in sc.items()})
    cred = model.credibility(list(range(10)), 8).mean(1)
    print("    cred", np.round(cred, 2).tolist(), "clone cop mean", round(np.mean([tsm.sig(model.params.cop[i]) for i in prob.issuers if i >= 10]), 3), "clone cred mean", round(np.mean([tsm.sig(model.params.bias[i]) for i in prob.issuers if i >= 10]), 3))
m1 = tsm.Model(spec, 8)
th = m1.fit(prob, first=True)
show("chosen ", m1, th)
m2 = tsm.Model(spec, 8)
lab = tsm.Params(8, 2, 5)
for i in prob.issuers:
    lab.ensure(i)
    lab.bias[i] = 3.0 if i < 5 else -3.0; lab.vec[i] = np.zeros(2); lab.cop[i] = -3.0 if i < 10 else 3.0
lab.beta0, lab.gamma0, lab.lh, lab.lmu = -3.0, 3.0, tsm.logit(0.15), tsm.logit(0.8)
th2, v = m2._optimize(prob, lab, 60)
m2.params = lab; m2._store(prob, th2, lab)
show("label  ", m2, th2)
