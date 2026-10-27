"""Soft mixture, partial pooling, and a contextual router on the same records.

The two records are the ones EvolvRAG already fits: an epoch-mixed society
weight and a cumulative organization weight. Copy collapse and the schema stay
on. The floor stays 0.55. Development seeds 600-615 choose the mixture weight,
the pooling strength, and the router temperature. The test seeds are scored
once after that choice. The hard switch is not retuned.

A supported mixture uses the society weight when the issuer has fewer than
two claims in the organization. A literal mixture always blends the stored
organization vector, which is still the prior 0.40 when that issuer has no
local fit. Partial pooling is n/(n+k) on the organization weight and k/(n+k)
on the society weight. The hierarchical fit recenters the organization CRH
on the society weight. The clearance router maps the two clearance counts
through a sigmoid whose temperature is chosen on development. The labeled
router is a logistic model fit on development only; its development score is
leave-one-seed-out, and the test uses the model fit on every development seed.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as rx
import run_fair as fair
import run_rag as rag

ROOT = rx.ROOT
OUT = ROOT / "results" / "mix.json"
DEV = list(fair.DEV)
TEST = list(fair.TEST)
REGIMES = fair.SELECT_REGIMES
TAU = 0.55
ETA = 0.5
MARGIN = 1
ALPHAS = (0.25, 0.5, 0.75)
KAPPAS = (1.0, 2.0, 4.0, 8.0, 16.0)
TEMPS = (0.5, 1.0, 2.0)
L2 = 1.0
DECISION = {
    "name": "mix-decision",
    "rule": "trusted",
    "cutoff": TAU,
    "schema": True,
    "copy": True,
}


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def clearance(glob, org_vec):
    width = max(len(glob), len(org_vec))
    soc_high = 0
    org_high = 0
    for issuer in range(width):
        if fair.weight_of(glob, issuer) + 1e-9 >= TAU:
            soc_high += 1
        if fair.weight_of(org_vec, issuer) + 1e-9 >= TAU:
            org_high += 1
    return soc_high, org_high


def fit_org_centered(history, society, names):
    center = {issuer: weight for issuer, weight in enumerate(society)}
    fitted = []
    width = len(society)
    for org in range(len(rx.ORGS)):
        claims = []
        for doc in history:
            slot = fair.entity_of_text(doc["text"], names)
            issuer = doc["issuer"]
            if slot is None or slot % len(rx.ORGS) != org:
                continue
            if isinstance(issuer, int) and issuer >= 0:
                claims.append((slot, doc["value"], issuer))
                if issuer + 1 > width:
                    width = issuer + 1
        base = list(society) + [rx.PRIOR] * (width - len(society))
        if claims:
            _truth, _log_w, learned = rx.crh_fit(claims, center=center)
            fresh = list(base)
            for issuer, weight in learned.items():
                if issuer >= len(fresh):
                    fresh.extend([rx.PRIOR] * (issuer + 1 - len(fresh)))
                fresh[issuer] = weight
            fitted.append(fresh)
        else:
            fitted.append(list(base))
    return fitted


class MixBook:
    def __init__(self, kind, param, router):
        self.kind = kind
        self.param = param
        self.router = router
        self.plain = rag.blank_weights()
        self.org = None
        self.org_n = None
        self.choice = None

    def _blend(self, issuer, glob, org_vec, counts, alpha, supported):
        global_w = fair.weight_of(glob, issuer)
        local_w = fair.weight_of(org_vec, issuer)
        n = counts.get(issuer, 0) if isinstance(counts, dict) else 0
        if supported and n < 2:
            return global_w
        return alpha * global_w + (1.0 - alpha) * local_w

    def prepare(self):
        self.choice = None
        if self.kind != "router" or self.org is None or self.router is None:
            return
        choice = []
        for org_i, org_vec in enumerate(self.org):
            soc_high, org_high = clearance(self.plain, org_vec)
            claims = self.org_n[org_i]
            total = sum(claims.values()) if isinstance(claims, dict) else 0
            raw = np.array([soc_high, org_high, soc_high - org_high, math.log1p(total), self.epoch / 8.0], dtype=float)
            feat = (raw - self.router["mean"]) / self.router["std"]
            score = float(self.router["bias"] + np.dot(self.router["coef"], feat))
            choice.append(score > 0.0)
        self.choice = choice

    def for_slot(self, slot):
        if self.org is None:
            return self.plain
        org_i = slot % len(rx.ORGS)
        org_vec = self.org[org_i]
        glob = self.plain
        counts = self.org_n[org_i]
        kind = self.kind
        if kind == "hier":
            soc_high, org_high = clearance(glob, org_vec)
            use_org = org_high >= soc_high + MARGIN

            def trust(issuer, org_vec=org_vec, glob=glob, use_org=use_org):
                if use_org:
                    return fair.weight_of(org_vec, issuer)
                return fair.weight_of(glob, issuer)

            return trust
        if kind == "hcrh":
            return org_vec
        if kind == "router":
            use_org = bool(self.choice[org_i]) if self.choice is not None else False

            def trust(issuer, org_vec=org_vec, glob=glob, use_org=use_org):
                if use_org:
                    return fair.weight_of(org_vec, issuer)
                return fair.weight_of(glob, issuer)

            return trust
        if kind == "pool":
            kappa = float(self.param)

            def trust(issuer, org_vec=org_vec, glob=glob, counts=counts, kappa=kappa):
                n = counts.get(issuer, 0) if isinstance(counts, dict) else 0
                local_w = fair.weight_of(org_vec, issuer)
                global_w = fair.weight_of(glob, issuer)
                return (n * local_w + kappa * global_w) / (n + kappa)

            return trust
        if kind == "soft":
            soc_high, org_high = clearance(glob, org_vec)
            alpha = sigmoid((soc_high - org_high) / float(self.param))

            def trust(issuer, org_vec=org_vec, glob=glob, counts=counts, alpha=alpha):
                return self._blend(issuer, glob, org_vec, counts, alpha, True)

            return trust
        supported = kind == "mix"

        def trust(issuer, org_vec=org_vec, glob=glob, counts=counts, alpha=float(self.param), supported=supported):
            return self._blend(issuer, glob, org_vec, counts, alpha, supported)

        return trust

    def update(self, fresh, history, names):
        self.plain = fair.fit_weights(fresh, self.plain, False, ETA, names)
        if self.kind == "hcrh":
            self.org = fit_org_centered(history, self.plain, names)
        else:
            blank = [rag.blank_weights() for _ in range(len(rx.ORGS))]
            self.org = fair.fit_weights(history, blank, True, 1.0, names)
        self.org_n = []
        for org in range(len(rx.ORGS)):
            _rows, counts = fair.claim_rows(history, names, org=org)
            self.org_n.append(counts)


def run_kind(world, kind, param=None, router=None):
    names = world.names
    order = list(world.mutable_ids) + list(world.stable_ids) + list(world.derived_ids)
    belief = list(world.init_belief[0])
    book = MixBook(kind, param, router)
    mail = []
    answers = []
    for epoch_i, epoch_docs in enumerate(world.visible):
        book.epoch = epoch_i + 1
        fresh = list(epoch_docs)
        mail.extend(fresh)
        find = fair.copy_find(mail, names)
        usable = fair.collapsed(mail, find)
        buckets = fair.index_hits(usable, names)
        book.prepare()
        row = list(belief)
        for slot in order:
            schema_v = fair.schema_value(world, row, slot)
            row[slot] = fair.answer_slot(DECISION, buckets.get(slot, []), row[slot], book.for_slot(slot), schema_v)
        belief = row
        answers.append([list(row) for _ in range(rx.N_AGENTS)])
        book.update(fair.collapsed(fresh, find), usable, names)
    counts = rx.empty_counts()
    rx.consume(counts, world, answers, 0.0)
    return rx.pack_metrics(counts)


def reference_metrics(world, name):
    cfg = next(cfg for cfg in fair.RULES if cfg["name"] == name)
    return fair.run_cfg(world, cfg)


def balanced_of(metrics):
    return (metrics["changed"] + metrics["stable"] + metrics["derived"]) / 3.0


def trace_router_rows(seeds):
    rows = []
    for regime in REGIMES:
        for seed in seeds:
            world = fair.load_world(seed, regime)
            names = world.names
            order = list(world.mutable_ids) + list(world.stable_ids) + list(world.derived_ids)
            belief_g = list(world.init_belief[0])
            belief_o = list(world.init_belief[0])
            book = MixBook("hier", None, None)
            mail = []
            for epoch_i, epoch_docs in enumerate(world.visible):
                book.epoch = epoch_i + 1
                fresh = list(epoch_docs)
                mail.extend(fresh)
                find = fair.copy_find(mail, names)
                usable = fair.collapsed(mail, find)
                buckets = fair.index_hits(usable, names)
                row_g = list(belief_g)
                row_o = list(belief_o)
                glob = book.plain
                org = book.org
                for slot in order:
                    schema_g = fair.schema_value(world, row_g, slot)
                    schema_o = fair.schema_value(world, row_o, slot)
                    if org is None:
                        trust_g = glob
                        trust_o = glob
                    else:
                        org_vec = org[slot % len(rx.ORGS)]
                        trust_g = glob
                        trust_o = org_vec
                    row_g[slot] = fair.answer_slot(DECISION, buckets.get(slot, []), row_g[slot], trust_g, schema_g)
                    row_o[slot] = fair.answer_slot(DECISION, buckets.get(slot, []), row_o[slot], trust_o, schema_o)
                belief_g = row_g
                belief_o = row_o
                if org is not None:
                    truth = world.timeline[epoch_i]
                    for org_i in range(len(rx.ORGS)):
                        slots = [slot for slot in order if slot % len(rx.ORGS) == org_i]
                        if not slots:
                            continue
                        glob_hit = sum(row_g[slot] == truth[slot] for slot in slots) / float(len(slots))
                        org_hit = sum(row_o[slot] == truth[slot] for slot in slots) / float(len(slots))
                        soc_high, org_high = clearance(glob, org[org_i])
                        total = sum(book.org_n[org_i].values())
                        rows.append(
                            {
                                "seed": seed,
                                "regime": regime,
                                "epoch": epoch_i + 1,
                                "org": org_i,
                                "y": int(org_hit > glob_hit),
                                "raw": [soc_high, org_high, soc_high - org_high, math.log1p(total), (epoch_i + 1) / 8.0],
                            }
                        )
                book.update(fair.collapsed(fresh, find), usable, names)
    return rows


def fit_logit(raw, y, l2=L2):
    x = np.asarray(raw, dtype=float)
    target = np.asarray(y, dtype=float)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    feat = (x - mean) / std
    design = np.column_stack([np.ones(len(feat)), feat])
    coef = np.zeros(design.shape[1])
    penalty = np.ones(design.shape[1])
    penalty[0] = 0.0
    for _ in range(30):
        score = design @ coef
        prob = 1.0 / (1.0 + np.exp(-np.clip(score, -30, 30)))
        weight = prob * (1.0 - prob)
        grad = design.T @ (prob - target) + l2 * penalty * coef
        hess = design.T @ (design * weight[:, None]) + l2 * np.diag(penalty)
        step = np.linalg.solve(hess, grad)
        coef = coef - step
        if np.max(np.abs(step)) < 1e-8:
            break
    return {"bias": float(coef[0]), "coef": coef[1:].tolist(), "mean": mean.tolist(), "std": std.tolist(), "l2": l2}


def router_from_rows(rows):
    return fit_logit([row["raw"] for row in rows], [row["y"] for row in rows])


def loo_router(rows):
    by_seed = {}
    for row in rows:
        by_seed.setdefault(row["seed"], []).append(row)
    models = {}
    for held in by_seed:
        train = [row for row in rows if row["seed"] != held]
        models[held] = router_from_rows(train)
    return models


def specs_dev():
    specs = [("mix", alpha) for alpha in ALPHAS]
    specs += [("literal", alpha) for alpha in ALPHAS]
    specs += [("pool", kappa) for kappa in KAPPAS]
    specs += [("soft", temp) for temp in TEMPS]
    specs.append(("hcrh", None))
    return specs


def select_family(scores, family, default):
    ranked = []
    for spec, mean in scores.items():
        kind, param = spec
        if kind != family:
            continue
        distance = 0.0 if param is None else abs(float(param) - float(default))
        ranked.append((mean, distance, spec))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    best = ranked[0][0]
    near = [item for item in ranked if best - item[0] <= 0.005]
    near.sort(key=lambda item: (item[1], -item[0]))
    return {"spec": near[0][2], "dev_mean": near[0][0], "best": best, "band": 0.005}


def mean_std(values):
    return rx.mean_std(values)


def evaluate_specs(seeds, specs, router, label):
    stored = {spec: {regime: [] for regime in REGIMES} for spec in specs}
    stored["hier"] = {regime: [] for regime in REGIMES}
    stored["global"] = {regime: [] for regime in REGIMES}
    stored["org"] = {regime: [] for regime in REGIMES}
    for regime in REGIMES:
        for seed in seeds:
            world = fair.load_world(seed, regime)
            bits = []
            for name in ("hier", "global", "org"):
                metrics = reference_metrics(world, name)
                stored[name][regime].append(metrics)
                bits.append(f"{name}:{balanced_of(metrics):.3f}")
            for spec in specs:
                kind, param = spec
                metrics = run_kind(world, kind, param, router if kind == "router" else None)
                stored[spec][regime].append(metrics)
                bits.append(f"{kind}:{balanced_of(metrics):.3f}")
            print(label, regime, seed, " ".join(bits), flush=True)
            OUT.write_text(json.dumps({"partial": label, "last": [label, regime, seed]}, indent=2), encoding="utf-8")
    return stored


def family_mean(stored, key):
    per_seed = None
    for regime in REGIMES:
        rows = stored[key][regime]
        if per_seed is None:
            per_seed = [balanced_of(row) for row in rows]
        else:
            for i, row in enumerate(rows):
                per_seed[i] += balanced_of(row)
    per_seed = [value / float(len(REGIMES)) for value in per_seed]
    mu, sd = mean_std(per_seed)
    return {"mean": mu, "std": sd, "seeds": per_seed}


def cell_mean(stored, key, regime, metric):
    values = [row[metric] for row in stored[key][regime]]
    mu, sd = mean_std(values)
    return {"mean": mu, "std": sd, "seeds": values}


def paired_seeds(left, right):
    diffs = [a - b for a, b in zip(left, right)]
    mu, sd = mean_std(diffs)
    n = len(diffs)
    if n <= 16:
        p = rx.signflip_p(diffs)
    else:
        p = fair.perm_p(diffs)
    return {"mean": mu, "std": sd, "p": p, "n": n}


def pack_methods(stored, keys):
    out = {}
    for key in keys:
        name = key if isinstance(key, str) else f"{key[0]}:{key[1]}"
        block = {"balanced": family_mean(stored, key)}
        for regime in REGIMES:
            block[regime] = {metric: cell_mean(stored, key, regime, metric) for metric in ("sp", "changed", "stable", "derived")}
        out[name] = block
    return out


def main():
    started = time.time()
    if "--smoke" in sys.argv:
        world = fair.load_world(399, "stationary")
        mine = run_kind(world, "hier")
        ref = reference_metrics(world, "hier")
        print("hier-match", {k: (round(mine[k], 6), round(ref[k], 6)) for k in ("sp", "changed", "stable", "derived")}, flush=True)
        for kind, param in (("mix", 0.5), ("literal", 0.5), ("pool", 4.0), ("soft", 1.0), ("hcrh", None)):
            metrics = run_kind(world, kind, param)
            print(kind, param, {k: round(metrics[k], 3) for k in ("sp", "changed", "stable", "derived")}, "bal", round(balanced_of(metrics), 3), flush=True)
        return
    rows = trace_router_rows(DEV)
    print("router-rows", len(rows), "positive", sum(row["y"] for row in rows), flush=True)
    full_router = router_from_rows(rows)
    loo = loo_router(rows)
    dev_specs = specs_dev()
    dev_router_specs = [("router", None)]
    dev_stored = evaluate_specs(DEV, dev_specs, None, "dev")
    loo_stored = {("router", None): {regime: [] for regime in REGIMES}}
    for regime in REGIMES:
        for seed in DEV:
            world = fair.load_world(seed, regime)
            metrics = run_kind(world, "router", None, loo[seed])
            loo_stored[("router", None)][regime].append(metrics)
            print("dev-loo", regime, seed, round(balanced_of(metrics), 3), flush=True)
    dev_means = {}
    for spec in dev_specs:
        dev_means[spec] = family_mean(dev_stored, spec)["mean"]
    chosen = {
        "mix": select_family(dev_means, "mix", 0.5),
        "literal": select_family(dev_means, "literal", 0.5),
        "pool": select_family(dev_means, "pool", 4.0),
        "soft": select_family(dev_means, "soft", 1.0),
    }
    print("CHOSEN", json.dumps({name: {"spec": list(item["spec"]), "dev": item["dev_mean"]} for name, item in chosen.items()}), flush=True)
    test_specs = [item["spec"] for item in chosen.values()] + [("hcrh", None), ("router", None)]
    test_stored = evaluate_specs(TEST, test_specs, full_router, "test")
    keys = ["hier", "global", "org"] + test_specs
    packed = pack_methods(test_stored, keys)
    dev_packed = pack_methods(dev_stored, ["hier", "global", "org"] + dev_specs)
    dev_packed.update(pack_methods(loo_stored, [("router", None)]))
    hier_seeds = packed["hier"]["balanced"]["seeds"]
    tests = []
    for name, block in packed.items():
        if name == "hier":
            continue
        stat = paired_seeds(hier_seeds, block["balanced"]["seeds"])
        tests.append({"metric": "balanced", "other": name, **stat})
        for regime in REGIMES:
            stat = paired_seeds(packed["hier"][regime]["sp"]["seeds"], block[regime]["sp"]["seeds"])
            tests.append({"metric": "sp", "regime": regime, "other": name, **stat})
    payload = {
        "protocol": {
            "dev_seeds": DEV,
            "test_seeds": TEST,
            "regimes": list(REGIMES),
            "objective": "mean over stationary, topic, open_rumor, sybil of (changed+stable+derived)/3",
            "band": 0.005,
            "records": "same society-wide and organization records as clearance routing",
            "supported_mix": "society weight when the issuer has fewer than two organization claims",
            "literal_mix": "always blend the stored organization vector, prior 0.40 if unfitted",
            "pool": "(n*organization + k*society) / (n+k)",
            "soft": "sigmoid((society clearances - organization clearances) / T) as the society share",
            "hcrh": "organization CRH recentered on the society weight, same pseudo-count 4",
            "router": "logistic on clearance counts, claim count, and epoch; L2 1; fit on development; test is not used to fit",
            "router_dev": "leave one development seed out",
            "chosen_on_dev": {name: {"kind": item["spec"][0], "param": item["spec"][1], "dev_mean": item["dev_mean"]} for name, item in chosen.items()},
            "hard_switch_not_retuned": True,
        },
        "router_fit": full_router,
        "router_train_positive": sum(row["y"] for row in rows) / float(len(rows)),
        "dev": {name: {"balanced": block["balanced"]["mean"], **{regime: block[regime]["sp"]["mean"] for regime in REGIMES}} for name, block in dev_packed.items()},
        "test": {
            name: {
                "balanced": block["balanced"]["mean"],
                "balanced_std": block["balanced"]["std"],
                **{regime: block[regime]["sp"]["mean"] for regime in REGIMES},
                **{f"{regime}_changed": block[regime]["changed"]["mean"] for regime in REGIMES},
                **{f"{regime}_stable": block[regime]["stable"]["mean"] for regime in REGIMES},
                **{f"{regime}_derived": block[regime]["derived"]["mean"] for regime in REGIMES},
            }
            for name, block in packed.items()
        },
        "tests": tests,
        "seconds": time.time() - started,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
