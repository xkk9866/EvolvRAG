import sys
import numpy as np, torch
sys.path.insert(0, ".")
import run_experiment as rx, run_fair as fair, tsm, run_tsm
torch.set_num_threads(2)
seed, regime, E = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
world = fair.load_world(seed, regime)
spec = tsm.Spec()
claims = []
for e, docs in enumerate(world.visible[:E], start=1):
    claims.extend(run_tsm.world_claims(docs, world.names, e))
prob = tsm.Problem(claims, E, list(world.init_belief[0]), world.parent, run_tsm.schema_fn, run_tsm.ctx_of, world.n_slots, True, "arrival")
def fitfrom(p):
    m = tsm.Model(spec, 8)
    th, v = m._optimize(prob, p, 100)
    m._store(prob, th, p); m.params = p
    with torch.no_grad():
        out = m._forward(th, prob, keep=True)
    return m, th, v, out
static = tsm.Params(8, 2, 3); static.lh = tsm.logit(0.002); static.lmu = tsm.logit(0.995)
lab = tsm.Params(8, 2, 5)
for i in prob.issuers:
    for P in (static, lab):
        P.ensure(i)
    lab.bias[i] = 3.0 if i < 5 else -3.0; lab.vec[i] = np.zeros(2); lab.cop[i] = -3.0
lab.beta0, lab.lh, lab.lmu = -1.0, tsm.logit(0.15), tsm.logit(0.8)
res = {}
for tag, P in (("static", static), ("change", lab)):
    m, th, v, out = fitfrom(P)
    sl = out["single_ll"].numpy(); pl = out["pair_ll"].numpy()
    st = [i for i, k in enumerate(prob.singles) if k in world.stable_ids]
    mu_ = [i for i, k in enumerate(prob.singles) if k in world.mutable_ids]
    with torch.no_grad():
        lp = float(m._log_prior(th))
    ans, _ = m.decide(prob, th)
    truth = world.timeline[E - 1]
    acc = np.mean([ans[k] == truth[k] for k in world.mutable_ids])
    print(tag, "obj", round(v, 1), "prior", round(lp, 1), "stable", round(sl[st].sum(), 1), "mutable", round(sl[mu_].sum(), 1), "pairs", round(pl.sum(), 1), "mut acc", round(acc, 3), {k: round(x, 3) for k, x in m.scalars().items()})
    print("    cred", np.round(m.credibility(list(range(10)), 8).mean(1), 2).tolist())
