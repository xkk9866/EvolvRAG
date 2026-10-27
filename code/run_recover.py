"""Recoverable clearance: an organization loss is not a permanent ban.

The society record stays an epoch mixture. The organization record keeps
every past claim, but a claim is an additive hit or loss against that
epoch's own consensus, not a log-score that one disagreement holds under
0.55 until tens of millions of later claims. When a slot is decided, that
slot's own claims are left out of the organization weight, so a source
with a clean record on other entities can publish a minority value.

Development seeds 600-615 lock the rule before seeds 700-731 are read.
The four-regime mean is the original objective. Leave-one-slot-out is
kept when it is inside 0.005 of clearance routing on that objective and
ahead of it on the lead worlds. Lead and slip are not in the four-regime
mean. Lead: one reliable issuer states the new value and four reliable
issuers keep stating the old one. Slip: each reliable issuer is wrong
once.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as rx
import run_fair as fair

ROOT = rx.ROOT
OUT = ROOT / "results" / "recover.json"
DEV = list(range(600, 616))
TEST = list(range(700, 732))
REGIMES = ("stationary", "topic", "open_rumor", "sybil")
TAU = 0.55
ETA = 0.5
MARGIN = 1
ALPHA = 4.0
PRIOR = rx.PRIOR
FLOOR = rx.TRUST_FLOOR
CAP = rx.TRUST_CAP
HIER = dict(fair.RULES[4])


def clip(weight):
    return min(CAP, max(FLOOR, weight))


def additive_weight(hits, n):
    if n < 2:
        return PRIOR
    raw = hits / n
    return clip((PRIOR * ALPHA + raw * n) / (ALPHA + n))


def crh_weight(losses, n):
    """The paper's shrunk CRH unit, from cumulative loss counts."""
    if n < 2:
        return PRIOR
    perfect = -math.log(rx.CRH_EPS / (n + rx.CRH_EPS))
    raw = -math.log((losses + rx.CRH_EPS) / (n + rx.CRH_EPS))
    raw_unit = 0.0 if perfect <= 0 else raw / perfect
    return clip((PRIOR * ALPHA + raw_unit * n) / (ALPHA + n))


def claims_of(docs, names):
    seen = {}
    for doc in docs:
        slot = fair.entity_of_text(doc["text"], names)
        issuer = doc["issuer"]
        if slot is None or not isinstance(issuer, int) or issuer < 0:
            continue
        seen[(slot, issuer)] = doc["value"]
    return [(slot, value, issuer) for (slot, issuer), value in seen.items()]


def fit_additive(claims, rounds=6):
    """Iterated additive accuracy, shrunk toward the prior. Same loop shape as CRH."""
    weight = {}
    per_issuer = {}
    per_slot = {}

    def current(issuer):
        return weight.get(issuer, PRIOR)

    for _ in range(rounds):
        per_issuer, per_slot = label_epoch(claims, current)
        weight = {}
        for issuer, (hits, n) in per_issuer.items():
            if n >= 2:
                weight[issuer] = additive_weight(hits, n)
    return weight, per_issuer, per_slot


def label_epoch(claims, weight_of):
    scores = defaultdict(lambda: defaultdict(float))
    for slot, value, issuer in claims:
        scores[slot][value] += weight_of(issuer)
    truth = {}
    for slot, bucket in scores.items():
        truth[slot] = max(bucket, key=lambda value: (bucket[value], str(value)))
    per_issuer = {}
    per_slot = {}
    for slot, value, issuer in claims:
        hit = 1 if truth[slot] == value else 0
        hits, n = per_issuer.get(issuer, (0, 0))
        per_issuer[issuer] = (hits + hit, n + 1)
        per_slot[(issuer, slot)] = (hit, 1)
    return per_issuer, per_slot


def crh_counts(claims):
    """CRH consensus, then hit counts on the same deduped claims."""
    dedup = {}
    for slot, value, issuer in claims:
        dedup[(slot, issuer)] = value
    rows = [(slot, value, issuer) for (slot, issuer), value in dedup.items()]
    if not rows:
        return {}, {}, {}
    truth, _log_w, learned = rx.crh_fit(rows)
    per = {}
    slots = {}
    for slot, value, issuer in rows:
        hit = 1 if truth.get(slot) == value else 0
        hits, n = per.get(issuer, (0, 0))
        per[issuer] = (hits + hit, n + 1)
        slots.setdefault(issuer, {})[slot] = (hit, 1)
    return learned, per, slots


class RecoverState:
    def __init__(self):
        self.soc = {}
        self.org_last = [dict() for _ in range(len(rx.ORGS))]
        self.org_per = [dict() for _ in range(len(rx.ORGS))]
        self.org_slot = [dict() for _ in range(len(rx.ORGS))]

    def soc_weight(self, issuer):
        return self.soc.get(issuer, PRIOR)

    def org_weight(self, org, issuer, slot, leave_slot):
        hits, n = self.org_per[org].get(issuer, (0, 0))
        if leave_slot:
            sh, sn = self.org_slot[org].get(issuer, {}).get(slot, (0, 0))
            hits -= sh
            n -= sn
        return additive_weight(hits, n)

    def trust_for(self, slot, kind):
        org = slot % len(rx.ORGS)
        leave = kind == "loo"
        issuers = set(self.soc)
        issuers.update(self.org_per[org])
        soc_high = 0
        org_high = 0
        for issuer in issuers:
            if self.soc_weight(issuer) + 1e-9 >= TAU:
                soc_high += 1
            if self.org_weight(org, issuer, slot, leave) + 1e-9 >= TAU:
                org_high += 1
        use_org = org_high >= soc_high + MARGIN

        def trust(issuer, org=org, use_org=use_org, leave=leave):
            if not isinstance(issuer, int) or issuer < 0:
                return None
            if use_org:
                return self.org_weight(org, issuer, slot, leave)
            return self.soc_weight(issuer)

        return trust

    def update(self, docs, names):
        claims = claims_of(docs, names)
        if not claims:
            return
        _learned, per_issuer, _slots = crh_counts(claims)
        for issuer, (hits, n) in per_issuer.items():
            if n < 2:
                continue
            fresh = additive_weight(hits, n)
            old = self.soc.get(issuer, PRIOR)
            self.soc[issuer] = (1.0 - ETA) * old + ETA * fresh
        for slot, value, issuer in claims:
            self.org_last[slot % len(rx.ORGS)][(slot, issuer)] = value
        for org in range(len(rx.ORGS)):
            rows = [(slot, value, issuer) for (slot, issuer), value in self.org_last[org].items()]
            if not rows:
                continue
            _learned, per, slots = crh_counts(rows)
            self.org_per[org] = per
            self.org_slot[org] = slots


def run_kind(world, kind):
    names = world.names
    order = list(world.mutable_ids) + list(world.stable_ids) + list(world.derived_ids)
    belief = list(world.init_belief[0])
    state = RecoverState()
    mail = []
    answers = []
    cfg = dict(HIER)
    for epoch_docs in world.visible:
        fresh = list(epoch_docs)
        mail.extend(fresh)
        find = fair.copy_find(mail, names)
        usable = fair.collapsed(mail, find)
        fresh_c = fair.collapsed(fresh, find)
        buckets = fair.index_hits(usable, names)
        row = list(belief)
        for slot in order:
            schema_v = fair.schema_value(world, row, slot) if cfg.get("schema") else None
            row[slot] = fair.answer_slot(cfg, buckets.get(slot, []), row[slot], state.trust_for(slot, kind), schema_v)
        belief = row
        answers.append([list(row) for _ in range(rx.N_AGENTS)])
        state.update(fresh_c, names)
    counts = rx.empty_counts()
    rx.consume(counts, world, answers, 0.0)
    return rx.pack_metrics(counts)


def balanced(metrics):
    return (metrics["changed"] + metrics["stable"] + metrics["derived"]) / 3.0


def mean_std(values):
    return rx.mean_std(values)


def lead_world(seed):
    """One reliable issuer leads. Four reliable issuers repeat the old value."""
    world = fair.load_world(seed, "stationary")
    for slot in world.mutable_ids:
        for epoch_i in range(len(world.visible) - 1):
            prev = world.truth0[slot] if epoch_i == 0 else world.timeline[epoch_i - 1][slot]
            cur = world.timeline[epoch_i][slot]
            if cur == prev or cur == "EXPIRED" or prev == "EXPIRED":
                continue
            for follow in (epoch_i, epoch_i + 1):
                bucket = world.visible[follow]
                kept = [doc for doc in bucket if doc["slot"] != slot]
                truth_now = world.timeline[follow][slot]
                for rank, issuer in enumerate((1, 2, 3, 4)):
                    kept.append(
                        {
                            "slot": slot,
                            "value": prev,
                            "time": follow + 1,
                            "issuer": issuer,
                            "text": rx.surface(world, slot, prev),
                            "receivers": list(range(rx.N_AGENTS)),
                            "seq": follow * 100000 + 40000 + rank,
                        }
                    )
                kept.append(
                    {
                        "slot": slot,
                        "value": truth_now,
                        "time": follow + 1,
                        "issuer": 0,
                        "text": rx.surface(world, slot, truth_now),
                        "receivers": list(range(rx.N_AGENTS)),
                        "seq": follow * 100000 + 90000,
                    }
                )
                world.visible[follow] = kept
            break
    return world


def slip_world(seed):
    """Each reliable issuer is wrong on one stable claim, then continues."""
    world = fair.load_world(seed, "stationary")
    flipped = set()
    if len(world.visible) < 2:
        return world
    for doc in world.visible[1]:
        issuer = doc["issuer"]
        if issuer in flipped or not isinstance(issuer, int) or issuer < 0 or issuer >= rx.N_REL:
            continue
        if doc["slot"] not in world.stable_ids:
            continue
        wrong = "qSLIP"
        doc["value"] = wrong
        doc["text"] = rx.surface(world, doc["slot"], wrong)
        flipped.add(issuer)
        if len(flipped) == rx.N_REL:
            break
    return world


def weight_trace(world):
    """Issuer 0's worst organization weight after each epoch."""
    names = world.names
    mail = []
    org_claims = [[] for _ in range(len(rx.ORGS))]
    rows = []
    for epoch_i, epoch_docs in enumerate(world.visible):
        mail.extend(epoch_docs)
        find = fair.copy_find(mail, names)
        fresh = fair.collapsed(epoch_docs, find)
        claims = claims_of(fresh, names)
        for slot, value, issuer in claims:
            org_claims[slot % len(rx.ORGS)].append((slot, value, issuer))
        crh_vals = []
        add_vals = []
        for org in range(len(rx.ORGS)):
            if not org_claims[org]:
                continue
            _truth, _log_w, learned = rx.crh_fit(org_claims[org])
            if 0 in learned:
                crh_vals.append(learned[0])
            _weight, per, _slots = fit_additive(org_claims[org])
            if 0 in per and per[0][1] >= 2:
                add_vals.append(additive_weight(*per[0]))
        rows.append(
            {
                "epoch": epoch_i + 1,
                "crh_min": None if not crh_vals else min(crh_vals),
                "additive_min": None if not add_vals else min(add_vals),
            }
        )
    return rows


def evaluate(seeds, regime, kinds, loader):
    stored = {kind: [] for kind in kinds}
    balanced_rows = {kind: [] for kind in kinds}
    for seed in seeds:
        world = loader(seed, regime)
        bits = []
        for kind in kinds:
            if kind == "hier":
                metrics = fair.run_cfg(world, HIER)
            else:
                metrics = run_kind(world, kind)
            stored[kind].append(metrics)
            balanced_rows[kind].append(balanced(metrics))
            bits.append(f"{kind}:{metrics['sp']:.3f}/{balanced(metrics):.3f}")
        print(regime, seed, " ".join(bits), flush=True)
    summary = {}
    for kind in kinds:
        block = {}
        for metric in ("sp", "changed", "stable", "derived"):
            values = [row[metric] for row in stored[kind]]
            mu, sd = mean_std(values)
            block[metric] = {"mean": mu, "std": sd, "seeds": values}
        mu, sd = mean_std(balanced_rows[kind])
        block["balanced"] = {"mean": mu, "std": sd, "seeds": balanced_rows[kind]}
        summary[kind] = block
    return summary


def loader_regime(seed, regime):
    return fair.load_world(seed, regime)


def loader_lead(seed, _regime):
    return lead_world(seed)


def loader_slip(seed, _regime):
    return slip_world(seed)


def choose(four, lead):
    """Lock the operator from development scores only."""
    hier = four["hier"]
    loo = four["loo"]
    if loo + 0.005 < hier:
        return "hier"
    if lead["loo"] > lead["hier"] + 0.005:
        return "loo"
    if four["add"] > hier + 0.005 and four["add"] >= loo:
        return "add"
    return "hier"


def paired(left, right):
    diffs = [a - b for a, b in zip(left, right)]
    mu, sd = mean_std(diffs)
    return {"mean": mu, "std": sd, "p": fair.perm_p(diffs), "n": len(diffs)}


def run_relay(world):
    """Each agent forwards the eligible passage it adopted from new mail."""
    names = world.names
    order = list(world.mutable_ids) + list(world.stable_ids) + list(world.derived_ids)
    beliefs = [list(row) for row in world.init_belief]
    books = [RecoverState() for _ in range(rx.N_AGENTS)]
    mail = [[] for _ in range(rx.N_AGENTS)]
    pending = [[] for _ in range(rx.N_AGENTS)]
    answers = []
    cfg = dict(HIER)
    for epoch_docs in world.visible:
        fresh = [[] for _ in range(rx.N_AGENTS)]
        for agent in range(rx.N_AGENTS):
            mail[agent].extend(pending[agent])
            pending[agent] = []
        for doc in epoch_docs:
            for agent in doc["receivers"]:
                if 0 <= agent < rx.N_AGENTS:
                    mail[agent].append(doc)
                    fresh[agent].append(doc)
        grid = []
        adopted = []
        for agent in range(rx.N_AGENTS):
            find = fair.copy_find(mail[agent], names) if cfg.get("copy") else None
            usable = fair.collapsed(mail[agent], find)
            fresh_c = fair.collapsed(fresh[agent], find)
            buckets = fair.index_hits(usable, names)
            fresh_buckets = fair.index_hits(fresh_c, names)
            row = list(beliefs[agent])
            sent = []
            for slot in order:
                schema_v = fair.schema_value(world, row, slot) if cfg.get("schema") else None
                trust = books[agent].trust_for(slot, "loo")
                row[slot] = fair.answer_slot(cfg, buckets.get(slot, []), row[slot], trust, schema_v)
                picked, saw = fair.trusted_pick(fresh_buckets.get(slot, []), None, trust, TAU)
                if saw and picked == row[slot]:
                    for hit in fresh_buckets.get(slot, []):
                        if hit["value"] == picked and fair.weight_of(trust, hit["issuer"]) is not None:
                            if fair.weight_of(trust, hit["issuer"]) + 1e-9 >= TAU:
                                sent.append(hit)
                                break
            beliefs[agent] = row
            grid.append(row)
            books[agent].update(fresh_c, names)
            adopted.append(sent)
        answers.append(grid)
        for agent, docs in enumerate(adopted):
            for doc in docs:
                relay = {
                    "slot": doc["slot"],
                    "value": doc["value"],
                    "time": doc["time"],
                    "issuer": doc["issuer"],
                    "text": doc["text"],
                    "seq": doc.get("seq", 0),
                    "receivers": [],
                }
                for other in range(rx.N_AGENTS):
                    if other != agent:
                        pending[other].append(relay)
    counts = rx.empty_counts()
    rx.consume(counts, world, answers, 0.0)
    return rx.pack_metrics(counts)


def run_private(world):
    cfg = dict(HIER, corpus="local")
    return fair.run_cfg(world, cfg)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "dev"
    payload = {}
    if OUT.exists() and mode != "dev":
        payload = json.loads(OUT.read_text(encoding="utf-8"))
    if mode == "dev":
        formula = {
            "crh_one_loss_n100": crh_weight(1, 100),
            "additive_one_loss_n4": additive_weight(3, 4),
            "additive_one_loss_n100": additive_weight(99, 100),
            "additive_clean_n2": additive_weight(2, 2),
        }
        print("formula", json.dumps(formula), flush=True)
        four = {}
        for regime in REGIMES:
            four[regime] = evaluate(DEV, regime, ("hier", "add", "loo"), loader_regime)
        lead = evaluate(DEV, "lead", ("hier", "loo"), loader_lead)
        slip = evaluate(DEV, "slip", ("hier", "loo"), loader_slip)
        four_mean = {}
        for kind in ("hier", "add", "loo"):
            values = []
            for regime in REGIMES:
                values.extend(four[regime][kind]["balanced"]["seeds"])
            mu, sd = mean_std(values)
            four_mean[kind] = mu
            print("dev-four", kind, round(mu, 4), round(sd, 4), flush=True)
        lead_mean = {kind: lead[kind]["balanced"]["mean"] for kind in ("hier", "loo")}
        print("dev-lead", json.dumps(lead_mean), flush=True)
        chosen = choose(four_mean, {"hier": lead_mean["hier"], "loo": lead_mean["loo"]})
        print("chosen", chosen, flush=True)
        trace = weight_trace(lead_world(600))
        payload = {
            "formula": formula,
            "dev_four": four,
            "dev_four_mean": four_mean,
            "dev_lead": lead,
            "dev_slip": slip,
            "trace_seed_600": trace,
            "chosen": chosen,
            "rule": "loo if four-regime mean is within 0.005 of hier and lead balanced mean exceeds hier by more than 0.005; else hier unless add beats hier by more than 0.005",
        }
        OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("wrote", OUT, flush=True)
        return
    chosen = payload.get("chosen")
    if chosen not in ("hier", "add", "loo"):
        raise RuntimeError("development lock is missing")
    if mode == "test":
        four = {}
        for regime in REGIMES:
            four[regime] = evaluate(TEST, regime, ("hier", "add", "loo"), loader_regime)
        lead = evaluate(TEST, "lead", ("hier", "loo"), loader_lead)
        slip = evaluate(TEST, "slip", ("hier", "loo"), loader_slip)
        tests = []
        for regime in REGIMES:
            for metric in ("sp", "changed", "stable", "derived", "balanced"):
                for other in ("hier", "add"):
                    if other == "loo":
                        continue
                    tests.append(
                        {
                            "regime": regime,
                            "metric": metric,
                            "other": other,
                            **paired(
                                four[regime]["loo"][metric]["seeds"],
                                four[regime][other][metric]["seeds"],
                            ),
                        }
                    )
        for name, block in (("lead", lead), ("slip", slip)):
            for metric in ("sp", "changed", "balanced"):
                tests.append(
                    {
                        "regime": name,
                        "metric": metric,
                        "other": "hier",
                        **paired(block["loo"][metric]["seeds"], block["hier"][metric]["seeds"]),
                    }
                )
        payload["test_four"] = four
        payload["test_lead"] = lead
        payload["test_slip"] = slip
        payload["tests"] = tests
        OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("wrote", OUT, flush=True)
        return
    if mode == "relay":
        stored = {name: {regime: [] for regime in REGIMES} for name in ("central", "private", "relay")}
        seeds = TEST
        for regime in REGIMES:
            for seed in seeds:
                world = fair.load_world(seed, regime)
                stored["central"][regime].append(balanced(run_kind(world, "loo")))
                stored["private"][regime].append(balanced(fair.run_cfg(world, dict(HIER, corpus="local"))))
                stored["relay"][regime].append(balanced(run_relay(world)))
                print(
                    "relay",
                    regime,
                    seed,
                    " ".join(f"{name}:{stored[name][regime][-1]:.3f}" for name in stored),
                    flush=True,
                )
        summary = {}
        tests = []
        for name in stored:
            values = []
            per = {}
            for regime in REGIMES:
                mu, sd = mean_std(stored[name][regime])
                per[regime] = {"mean": mu, "std": sd}
                values.extend(stored[name][regime])
            mu, sd = mean_std(values)
            summary[name] = {"balanced": mu, "std": sd, "per_regime": per, "seeds": stored[name]}
        for regime in REGIMES:
            tests.append(
                {
                    "regime": regime,
                    "other": "private",
                    **paired(stored["relay"][regime], stored["private"][regime]),
                }
            )
            tests.append(
                {
                    "regime": regime,
                    "other": "central",
                    **paired(stored["relay"][regime], stored["central"][regime]),
                }
            )
        flat_relay = []
        flat_private = []
        flat_central = []
        for regime in REGIMES:
            flat_relay.extend(stored["relay"][regime])
            flat_private.extend(stored["private"][regime])
            flat_central.extend(stored["central"][regime])
        tests.append({"regime": "four", "other": "private", **paired(flat_relay, flat_private)})
        tests.append({"regime": "four", "other": "central", **paired(flat_relay, flat_central)})
        payload["relay"] = {"summary": summary, "tests": tests}
        OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("wrote", OUT, flush=True)
        return
    raise RuntimeError(mode)


if __name__ == "__main__":
    main()
