import sys, json
import numpy as np, torch
sys.path.insert(0, ".")
import run_experiment as rx, run_fair as fair, tsm, run_tsm
torch.set_num_threads(1)
seed, regime = int(sys.argv[1]), sys.argv[2]
kw = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
world = fair.load_world(seed, regime)
spec = tsm.Spec(**kw)
model = tsm.Model(spec, 8)
claims = []
for e, docs in enumerate(world.visible, start=1):
    claims.extend(run_tsm.world_claims(docs, world.names, e))
    prob = tsm.Problem(claims, e, list(world.init_belief[0]), world.parent, run_tsm.schema_fn, run_tsm.ctx_of, world.n_slots, spec.schema, spec.clock)
    th = model.fit(prob, first=e == 1)
    ans, post = model.decide(prob, th)
    truth = world.timeline[e - 1]
    groups = {"stable": world.stable_ids, "mutable": world.mutable_ids, "derived": world.derived_ids}
    acc = {g: round(np.mean([ans[k] == truth[k] for k in ids]), 3) for g, ids in groups.items()}
    cred = model.credibility(list(range(10)), 8)
    sc = model.scalars()
    print(e, acc, "obj", round(model.history[-1]["objective"], 1), {k: round(v, 3) for k, v in sc.items()})
    print("   cred mean by src", np.round(cred.mean(1), 2).tolist())
    lab = tsm.Params(8, spec.rank, 5)
    for i in prob.issuers:
        lab.ensure(i)
        if regime == "topic" and i < 10:
            lab.bias[i] = 0.0; lab.vec[i] = np.array([3.0 if i < 5 else -3.0, 0.0])
        else:
            lab.bias[i] = 3.0 if i < 5 else -3.0; lab.vec[i] = np.zeros(2)
        lab.cop[i] = -3.0 if i < 10 else 3.0
    if regime == "topic":
        lab.ctx = np.array([[1.0, 0.0] if c < 4 else [-1.0, 0.0] for c in range(8)])
    _th, labv = model._optimize(prob, lab, 200)
    print("   labelled-init objective", round(labv, 1), "chosen", round(model.history[-1]["objective"], 1))
    if e == len(world.visible):
        wrong = [k for k in world.stable_ids if ans[k] != truth[k]][:4]
        for k in wrong:
            rows = [(r[1], r[2], r[3], round(r[4], 2)) for r in prob.rows if r[0] == k]
            print("  WRONG stable", k, "truth", truth[k], "ans", ans[k], "mem", world.init_belief[0][k], rows[:14])
        wrongm = [k for k in world.mutable_ids if ans[k] != truth[k]][:4]
        for k in wrongm:
            rows = [(r[1], r[2], r[3], round(r[4], 2)) for r in prob.rows if r[0] == k]
            print("  WRONG mutable", k, "truth", truth[k], "ans", ans[k], "mem", world.init_belief[0][k], "hist", [world.timeline[i][k] for i in range(8)], rows[:14])
        print("  copy prop", np.round(model.copy_propensity(list(range(10))), 2).tolist())
