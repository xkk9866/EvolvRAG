"""Contested-evidence society for EvolvRAG.

Documents carry a slot, a value, a time, and an issuer id. They do not carry a
source-quality label. The primary endpoint, fixed before the test seeds are
read, is the stability-plasticity score: the mean of changed-fact accuracy and
retention. Thresholds are chosen on development seeds only. Test seeds are
reported separately. Message counts are revision traffic. A liveness ping is
not added, and recency baselines do not gossip.
"""

from __future__ import annotations

import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "results" / "metrics.json"
DATA_PATH = ROOT / "data" / "society_events.json"

N_AGENTS = 8
N_STABLE = 40
N_MUTABLE = 80
N_DERIVED = 40
EPOCHS = 8
# Development seeds select thresholds. Test seeds are the reported comparison.
SEEDS_DEV = [0, 1, 2, 3, 4]
SEEDS_TEST = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
RATES = [0.08, 0.16, 0.28]
P_OBS_LIST = [0.25, 0.45, 0.70, 1.00]
DEFAULT_RATE = 0.16
DEFAULT_POBS = 0.45

P_EXPIRE = 0.05
P_SECOND = 0.42
P_RUMOR_EVENT = 0.30
P_RUMOR_IDLE = 0.04
P_COLLUDE = 0.08
P_CONFIRM = 0.36
P_RUMOR_STABLE = 0.10
P_COLLUDE_STABLE = 0.04
P_OVERRIDE = 0.16
P_DERIVED_DOC = 0.22
P_DERIVED_RUMOR = 0.14
P_SKEW = 0.18
P_INIT_ERR = 0.11
# Numerical floor inside the CRH logarithm. Not a decision threshold.
CRH_EPS = 1e-6
# Pseudo-claims in the shrink toward the prior. Selected on development seeds.
CRH_ALPHA = 4.0

TS_WINDOW = 3
RESET_EVERY = 3
FANOUT = 2
MAX_QUERIES = 4
MAX_PUSHES = 4

STICK = 0.58
MARGIN = 0.08
SCHEMA_TRUST = 0.74
SCHEMA_CLEAR = 0.12
ONE_ISSUER_TRUST = 0.66
PRIOR = 0.40
TRUST_UP = 0.06
TRUST_DOWN = 0.05
TRUST_FLOOR = 0.18
TRUST_CAP = 0.88
CONFIRM_BONUS = 0.05
QUERY_DISCOUNT = 0.55
HIGH_MARGIN = 0.42

N_REL = 5
N_NOISY = 5
SCHEMA = -1
PRIOR_ISSUER = -2

ORGS = [
    "Harbor Clinic",
    "North Depot",
    "Cedar School",
    "Lumen Grid",
    "West Archive",
    "Pioneer Bank",
    "Maple Transit",
    "Orchid Lab",
]
ROLES = [
    "duty lead",
    "desk contact",
    "budget owner",
    "night route",
    "audit chair",
    "field liaison",
]


def mean_std(xs):
    n = len(xs)
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return mu, math.sqrt(var)


def clip_trust(x):
    return max(TRUST_FLOOR, min(TRUST_CAP, x))


class World:
    def __init__(self):
        self.n_prim = N_STABLE + N_MUTABLE
        self.n_slots = self.n_prim + N_DERIVED
        self.stable_ids = list(range(N_STABLE))
        self.mutable_ids = list(range(N_STABLE, self.n_prim))
        self.derived_ids = list(range(self.n_prim, self.n_slots))
        self.parent = {}
        for j, slot in enumerate(self.derived_ids):
            self.parent[slot] = self.mutable_ids[j]
        self.children = defaultdict(list)
        for child, parent in self.parent.items():
            self.children[parent].append(child)
        self.names = []
        self.roles = []
        for i in range(self.n_slots):
            org = ORGS[i % len(ORGS)]
            role = ROLES[(i // len(ORGS)) % len(ROLES)]
            if i in self.parent:
                role = "dependent " + role
            self.names.append(f"{org} #{i:03d}")
            self.roles.append(role)
        self.truth0 = [f"v0" for _ in range(self.n_slots)]
        for child, parent in self.parent.items():
            self.truth0[child] = f"via-{self.truth0[parent]}"


def surface(world, slot, value):
    return f"{world.names[slot]} lists {world.roles[slot]} as {value}."


def query_text(world, slot):
    return f"What is the {world.roles[slot]} of {world.names[slot]}?"


def pick_observers(rng, p_obs):
    return [i for i in range(N_AGENTS) if rng.random() < p_obs]


def skew_time(rng, epoch):
    if rng.random() < P_SKEW:
        return max(1, epoch + rng.choice([-1, 1]))
    return epoch


def rel_issuer(rng):
    return rng.randrange(N_REL)


def noisy_issuer(rng):
    return N_REL + rng.randrange(N_NOISY)


def wrong_value(rng, truth, epoch):
    pool = [f"v{k}" for k in range(0, EPOCHS + 1) if f"v{k}" != truth and f"v{k}" != "v0"]
    pool.append(f"q{epoch}")
    if truth != "EXPIRED" and rng.random() < 0.18:
        return "EXPIRED"
    return rng.choice(pool)


def make_doc(world, rng, slot, value, epoch, issuer, receivers):
    return {
        "slot": slot,
        "value": value,
        "time": skew_time(rng, epoch),
        "issuer": issuer,
        "text": surface(world, slot, value),
        "receivers": list(receivers),
    }


class TruthBook:
    def __init__(self, world):
        self.world = world
        self.value = list(world.truth0)
        self.expired = [False] * world.n_slots
        self.history = [set() for _ in range(world.n_slots)]
        self.open_version = {}
        self.versions = defaultdict(list)
        self.timeline = []
        self.history_at = []

    def assign(self, slot, new, epoch):
        old = self.value[slot]
        if old == new:
            return
        if old != "EXPIRED":
            self.history[slot].add(old)
        if slot in self.open_version:
            prev, start = self.open_version.pop(slot)
            if start <= epoch - 1:
                self.versions[slot].append((prev, start, epoch - 1))
        self.value[slot] = new
        if new == "EXPIRED":
            self.expired[slot] = True
        else:
            self.open_version[slot] = (new, epoch)

    def snapshot(self):
        self.timeline.append(list(self.value))
        self.history_at.append([set(h) for h in self.history])

    def close(self):
        for slot, (prev, start) in self.open_version.items():
            self.versions[slot].append((prev, start, EPOCHS))


def generate_trajectory(seed, rate, p_obs, regime="stationary"):
    rng = random.Random(seed * 10007 + int(rate * 1000) + int(p_obs * 1000))
    # Stationary draws the same issuer stream as the original generator.
    # Topic, drift, and collusion are extra worlds and are not the primary cell.

    def true_issuer(slot, epoch):
        org = slot % len(ORGS)
        if regime == "topic":
            if org < 4:
                return rng.randrange(N_REL)
            return N_REL + rng.randrange(N_NOISY)
        if regime == "drift" and epoch > EPOCHS // 2:
            return N_REL + rng.randrange(N_NOISY)
        return rng.randrange(N_REL)

    def false_issuer(slot, epoch):
        org = slot % len(ORGS)
        if regime == "topic":
            if org < 4:
                return N_REL + rng.randrange(N_NOISY)
            return rng.randrange(N_REL)
        if regime == "drift" and epoch > EPOCHS // 2:
            return rng.randrange(N_REL)
        return N_REL + rng.randrange(N_NOISY)

    p_collude = 0.35 if regime == "collude" else P_COLLUDE
    p_collude_stable = 0.20 if regime == "collude" else P_COLLUDE_STABLE
    # Overlap gives truth a thinner address list than a lie, so recipient
    # count is no longer a near-perfect truth filter. Stationary keeps p_obs.
    truth_p = 0.25 if regime == "overlap" else p_obs
    phase = {"epoch": 1}

    def truth_receivers(rng_):
        # unicast_truth addresses every true update to one agent. Stationary
        # still draws an independent inclusion for each agent.
        if regime == "unicast_truth":
            return [rng_.randrange(N_AGENTS)]
        return pick_observers(rng_, truth_p)

    def _wide_rumor(rng_, empty_fallback):
        obs = pick_observers(rng_, p_obs if regime != "overlap" else 0.55)
        return obs if obs else empty_fallback

    def one_receiver(rng_):
        # Stationary rumors stay unicast. open_rumor uses the truth draw.
        # overlap uses a wider draw than truth. drift_prop switches at epoch 5.
        wide = regime == "open_rumor" or regime == "overlap" or (regime == "drift_prop" and phase["epoch"] >= 5)
        if wide:
            return _wide_rumor(rng_, [rng_.randrange(N_AGENTS)])
        return [rng_.randrange(N_AGENTS)]

    def directed(agent, rng_):
        wide = regime == "open_rumor" or regime == "overlap" or (regime == "drift_prop" and phase["epoch"] >= 5)
        if wide:
            return _wide_rumor(rng_, [agent])
        return [agent]
    world = World()
    book = TruthBook(world)
    init = [list(world.truth0) for _ in range(N_AGENTS)]
    for agent in range(N_AGENTS):
        for slot in range(world.n_slots):
            if rng.random() >= P_INIT_ERR:
                continue
            if slot in world.parent:
                init[agent][slot] = f"via-v{rng.choice([1, 2, 3, 4, 5, 6])}"
            else:
                init[agent][slot] = f"v{rng.choice([1, 2, 3, 4, 5, 6])}"
    epochs = []
    for epoch in range(1, EPOCHS + 1):
        phase["epoch"] = epoch
        docs = []
        for slot in world.mutable_ids:
            if book.expired[slot]:
                continue
            acted = False
            if rng.random() < P_EXPIRE:
                book.assign(slot, "EXPIRED", epoch)
                for child in world.children[slot]:
                    book.assign(child, "EXPIRED", epoch)
                obs = truth_receivers(rng)
                docs.append(make_doc(world, rng, slot, "EXPIRED", epoch, true_issuer(slot, epoch), obs))
                acted = True
            elif rng.random() < rate:
                new_v = f"v{epoch}"
                book.assign(slot, new_v, epoch)
                obs = truth_receivers(rng)
                issuer = true_issuer(slot, epoch)
                docs.append(make_doc(world, rng, slot, new_v, epoch, issuer, obs))
                if rng.random() < P_SECOND:
                    obs2 = truth_receivers(rng)
                    issuer2 = true_issuer(slot, epoch)
                    docs.append(make_doc(world, rng, slot, new_v, epoch, issuer2, obs2))
                for child in world.children[slot]:
                    if book.expired[child]:
                        continue
                    final = f"via-{new_v}"
                    override = rng.random() < P_OVERRIDE
                    if override:
                        final = f"ov{epoch}-{child}"
                    book.assign(child, final, epoch)
                    if override or rng.random() < P_DERIVED_DOC:
                        docs.append(
                            make_doc(
                                world,
                                rng,
                                child,
                                final,
                                epoch,
                                true_issuer(slot, epoch),
                                truth_receivers(rng),
                            )
                        )
                    if rng.random() < P_DERIVED_RUMOR:
                        rumor_v = wrong_value(rng, final, epoch)
                        if rumor_v == final:
                            rumor_v = f"q{epoch}"
                        docs.append(
                            make_doc(
                                world,
                                rng,
                                child,
                                rumor_v,
                                epoch,
                                false_issuer(slot, epoch),
                                one_receiver(rng),
                            )
                        )
                acted = True
            rumor_p = P_RUMOR_EVENT if acted else P_RUMOR_IDLE
            for agent in range(N_AGENTS):
                if rng.random() < rumor_p:
                    rumor_v = wrong_value(rng, book.value[slot], epoch)
                    docs.append(
                        make_doc(
                            world,
                            rng,
                            slot,
                            rumor_v,
                            epoch,
                            false_issuer(slot, epoch),
                            directed(agent, rng),
                        )
                    )
            if acted and rng.random() < p_collude:
                rumor_v = wrong_value(rng, book.value[slot], epoch)
                iss_a = false_issuer(slot, epoch)
                iss_b = false_issuer(slot, epoch)
                pair = rng.sample(range(N_AGENTS), 2)
                docs.append(make_doc(world, rng, slot, rumor_v, epoch, iss_a, directed(pair[0], rng)))
                docs.append(make_doc(world, rng, slot, rumor_v, epoch, iss_b, directed(pair[1], rng)))
        for slot in world.stable_ids:
            if rng.random() < P_CONFIRM:
                docs.append(
                    make_doc(
                        world,
                        rng,
                        slot,
                        book.value[slot],
                        epoch,
                        true_issuer(slot, epoch),
                        truth_receivers(rng),
                    )
                )
            for agent in range(N_AGENTS):
                if rng.random() < P_RUMOR_STABLE:
                    rumor_v = wrong_value(rng, book.value[slot], epoch)
                    docs.append(
                        make_doc(
                            world,
                            rng,
                            slot,
                            rumor_v,
                            epoch,
                            false_issuer(slot, epoch),
                            directed(agent, rng),
                        )
                    )
            if rng.random() < p_collude_stable:
                rumor_v = wrong_value(rng, book.value[slot], epoch)
                pair = rng.sample(range(N_AGENTS), 2)
                docs.append(make_doc(world, rng, slot, rumor_v, epoch, false_issuer(slot, epoch), directed(pair[0], rng)))
                docs.append(make_doc(world, rng, slot, rumor_v, epoch, false_issuer(slot, epoch), directed(pair[1], rng)))
        book.snapshot()
        epochs.append(docs)
    book.close()
    world.timeline = book.timeline
    world.history_at = book.history_at
    world.versions = book.versions
    world.init_belief = init
    world.visible = epochs
    return world


def group_docs(docs):
    grouped = defaultdict(list)
    for doc in docs:
        grouped[doc["slot"]].append(doc)
    return grouped


def issuer_weight(trust, issuer, use_trust):
    if issuer == SCHEMA:
        return SCHEMA_TRUST
    if not use_trust or not isinstance(trust, list) or issuer < 0 or issuer >= len(trust):
        return PRIOR
    return trust[issuer]


def unique_issuer_scores(docs, trust, use_trust, clusters=None):
    scores = defaultdict(float)
    issuers = defaultdict(set)
    best_issuer = {}
    best_w = defaultdict(float)
    if not clusters:
        for doc in docs:
            issuer = doc["issuer"]
            if issuer in issuers[doc["value"]]:
                continue
            weight = issuer_weight(trust, issuer, use_trust)
            scores[doc["value"]] += weight
            issuers[doc["value"]].add(issuer)
            if weight >= best_w[doc["value"]]:
                best_w[doc["value"]] = weight
                best_issuer[doc["value"]] = issuer
        return scores, issuers, best_issuer
    best = {}
    for doc in docs:
        issuer = doc["issuer"]
        cid = clusters.get(issuer, issuer) if isinstance(issuer, int) and issuer >= 0 else issuer
        weight = issuer_weight(trust, issuer, use_trust)
        key = (doc["value"], cid)
        held = best.get(key)
        if held is not None and weight <= held[0]:
            continue
        best[key] = (weight, issuer)
    for (value, cid), (weight, issuer) in best.items():
        scores[value] += weight
        issuers[value].add(cid)
        if weight >= best_w[value]:
            best_w[value] = weight
            best_issuer[value] = issuer
    return scores, issuers, best_issuer


def decide(belief, docs, trust, schema_value, flags, clusters=None):
    scores, issuers, best_issuer = unique_issuer_scores(docs, trust, flags["trust"], clusters)
    use_gate = flags.get("margin", True) and not flags.get("open", False)
    if belief is not None and use_gate:
        scores[belief] += STICK
    if flags["dep"] and schema_value is not None:
        if SCHEMA not in issuers[schema_value]:
            scores[schema_value] += SCHEMA_TRUST
            issuers[schema_value].add(SCHEMA)
            best_issuer.setdefault(schema_value, SCHEMA)
    if not flags["exp"]:
        scores.pop("EXPIRED", None)
    if not scores:
        return belief, False, 0.0, None
    if not flags["margin"] and not flags.get("open", False):
        newest = max(docs, key=lambda d: (d["time"], d.get("seq", 0))) if docs else None
        if newest is None:
            return belief, False, 0.0, None
        return newest["value"], newest["value"] != belief, 0.0, newest["issuer"]
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if not use_gate:
        best_v, best_s = ranked[0]
        second_s = ranked[1][1] if len(ranked) > 1 else 0.0
        top = best_issuer.get(best_v, PRIOR_ISSUER)
        return best_v, best_v != belief, best_s - second_s, top
    best_v, best_s = ranked[0]
    second_s = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best_s - second_s
    n_iss = len(issuers.get(best_v, ()))
    top = best_issuer.get(best_v, PRIOR_ISSUER)
    top_w = issuer_weight(trust, top, flags["trust"])
    schema_ok = (
        flags["dep"]
        and best_v == schema_value
        and top == SCHEMA
        and second_s <= best_s - SCHEMA_CLEAR
    )
    backed = n_iss >= 2 or top_w >= ONE_ISSUER_TRUST or schema_ok
    if best_v == belief:
        return belief, False, margin, top
    strong_pair = n_iss >= 2 and best_s >= STICK
    strong_single = top_w >= ONE_ISSUER_TRUST and best_s >= STICK and top != SCHEMA
    if margin >= MARGIN and (strong_pair or strong_single or schema_ok):
        return best_v, True, margin, top
    return belief, False, margin, top


def crh_fit(claims, n_iter=6, shrink=True, center=None):
    """Categorical CRH: 0-1 loss, max-normalized negative-log weights.

    Weights returned for the baseline are the paper's -log scores.
    `unit` is that score divided by the perfect-source score, then shrunk
    toward the prior. EvolvRAG's margin uses `unit` only for issuers with
    at least two claims in the current mailbox.
    """
    dedup = {}
    for slot, value, issuer in claims:
        if issuer is None or issuer < 0:
            continue
        dedup[(slot, issuer)] = value
    rows = [(slot, value, issuer) for (slot, issuer), value in dedup.items()]
    if not rows:
        return {}, {}, {}
    slots = sorted({slot for slot, _value, _issuer in rows})
    issuers = sorted({issuer for _slot, _value, issuer in rows})
    counts = defaultdict(int)
    for _slot, _value, issuer in rows:
        counts[issuer] += 1
    log_w = {issuer: 1.0 for issuer in issuers}
    unit = {issuer: PRIOR for issuer in issuers}
    truth = {}
    for _ in range(n_iter):
        for slot in slots:
            scores = defaultdict(float)
            for s, value, issuer in rows:
                if s == slot:
                    scores[value] += log_w[issuer]
            truth[slot] = max(scores, key=lambda value: (scores[value], value))
        loss = {}
        for issuer in issuers:
            loss[issuer] = sum(
                1.0 for slot, value, src in rows if src == issuer and truth.get(slot) != value
            )
        peak = max(loss.values()) if loss else 0.0
        for issuer in issuers:
            if peak <= 0:
                log_w[issuer] = 1.0
                raw_unit = 1.0
            else:
                log_w[issuer] = -math.log((loss[issuer] + CRH_EPS) / (peak + CRH_EPS))
                perfect = -math.log(CRH_EPS / (counts[issuer] + CRH_EPS))
                raw = -math.log((loss[issuer] + CRH_EPS) / (counts[issuer] + CRH_EPS))
                raw_unit = 0.0 if perfect <= 0 else raw / perfect
            base = PRIOR if center is None else center.get(issuer, PRIOR)
            if counts[issuer] < 2:
                unit[issuer] = base
            elif not shrink:
                unit[issuer] = min(TRUST_CAP, max(TRUST_FLOOR, raw_unit))
            else:
                shrunk = (base * CRH_ALPHA + raw_unit * counts[issuer]) / (CRH_ALPHA + counts[issuer])
                unit[issuer] = min(TRUST_CAP, max(TRUST_FLOOR, shrunk))
    learned = {issuer: unit[issuer] for issuer in issuers if counts[issuer] >= 2}
    return truth, log_w, learned


def copy_clusters(docs, provisional):
    """Collapse issuers who copy a value the provisional vote rejected.

    Agreement on the provisional truth does not create a link. One shared
    false value does. A pair that splits, with one side on the provisional
    truth, is left independent.
    """
    by_iss = defaultdict(dict)
    for doc in docs:
        issuer = doc["issuer"]
        if issuer is None or not isinstance(issuer, int) or issuer < 0:
            continue
        by_iss[issuer][doc["slot"]] = doc["value"]
    issuers = sorted(by_iss)
    parent = {issuer: issuer for issuer in issuers}

    def find(issuer):
        while parent[issuer] != issuer:
            parent[issuer] = parent[parent[issuer]]
            issuer = parent[issuer]
        return issuer

    def union(left, right):
        ra, rb = find(left), find(right)
        if ra != rb:
            parent[rb] = ra

    for i, left in enumerate(issuers):
        for right in issuers[i + 1 :]:
            same_false = 0
            blocked = False
            for slot in set(by_iss[left]) & set(by_iss[right]):
                va, vb = by_iss[left][slot], by_iss[right][slot]
                truth = provisional.get(slot)
                if va != vb:
                    if truth is not None and truth in (va, vb):
                        blocked = True
                elif truth != va:
                    same_false += 1
            if same_false >= 1 and not blocked:
                union(left, right)
    return {issuer: find(issuer) for issuer in issuers}


def crh_org_weights(claims, shrink=True):
    """Global CRH weight, then a per-organization weight shrunk toward it."""
    _truth, _log_w, learned = crh_fit(claims, shrink=shrink)
    global_w = {issuer: PRIOR for issuer in range(N_REL + N_NOISY)}
    global_w.update(learned)
    by_org = defaultdict(list)
    for slot, value, issuer in claims:
        if issuer is None or not isinstance(issuer, int) or issuer < 0:
            continue
        by_org[slot % len(ORGS)].append((slot, value, issuer))
    org_w = {}
    for org, rows in by_org.items():
        _t, _l, local = crh_fit(rows, shrink=shrink, center=global_w)
        for issuer, weight in local.items():
            org_w[(org, issuer)] = weight
    return global_w, org_w


def majority_truth(claims, held):
    dedup = {}
    for slot, value, issuer in claims:
        if issuer is None or issuer < 0:
            continue
        dedup[(slot, issuer)] = value
    buckets = defaultdict(lambda: defaultdict(set))
    for (slot, issuer), value in dedup.items():
        buckets[slot][value].add(issuer)
    truth = {}
    for slot, vals in buckets.items():
        best_n = max(len(members) for members in vals.values())
        cands = [value for value, members in vals.items() if len(members) == best_n]
        current = held(slot)
        truth[slot] = current if current in cands else min(cands)
    return truth


class Memory:
    def __init__(self, world, method, seed, budget=None):
        self.world = world
        self.method = method
        self.budget = budget
        self.rng = random.Random(20_000 + seed * 17 + sum(ord(c) for c in method))
        off = dict(social=False, margin=False, trust=False, dep=False, exp=False, query=False, relay=False)
        self.flags = {
            "Static": dict(off, mode="static"),
            "Timestamp": dict(off, mode="timestamp", exp=True),
            "Reset": dict(off, mode="reset", exp=True),
            "LTM": dict(off, mode="ltm", exp=True),
            "Majority": dict(off, mode="majority"),
            "CRH": dict(off, mode="crh"),
            "PoolCRH": dict(off, mode="poolcrh"),
            "PoolMajority": dict(off, mode="poolmaj"),
            "PoolCount": dict(off, mode="poolcount"),
            "PoolUniform": dict(off, mode="central", margin=True, trust=False, dep=False, exp=True),
            "PoolOpen": dict(off, mode="central", margin=True, trust=True, dep=False, exp=True, open=True),
            "PoolWeight": dict(off, mode="central", margin=True, trust=True, dep=False, exp=True),
            "NoShrink": dict(off, mode="central", margin=True, trust=True, dep=True, exp=True, shrink=False),
            "Indep": dict(off, mode="central", margin=True, trust=True, dep=True, exp=True, indep=True),
            "OrgTrust": dict(off, mode="central", margin=True, trust=True, dep=True, exp=True, org=True),
            "Routed": dict(off, mode="central", margin=True, trust=True, dep=True, exp=True, indep=True, org=True),
            "Central": dict(off, mode="central", margin=True, trust=True, dep=True, exp=True),
            "EvolvRAG": dict(off, mode="evolv", margin=True, trust=True, dep=True, exp=True, relay=True),
            "NoTrust": dict(off, mode="evolv", margin=True, trust=False, dep=True, exp=True, relay=True),
            "NoRelay": dict(off, mode="evolv", margin=True, trust=True, dep=True, exp=True, relay=False),
            "NoDep": dict(off, mode="evolv", margin=True, trust=True, dep=False, exp=True, relay=True),
            "NoExpire": dict(off, mode="evolv", margin=True, trust=True, dep=True, exp=False, relay=True),
        }[method]
        self.belief = [list(row) for row in world.init_belief]
        self.belief_time = [[0 for _ in range(world.n_slots)] for _ in range(N_AGENTS)]
        self.last_issuer = [[PRIOR_ISSUER for _ in range(world.n_slots)] for _ in range(N_AGENTS)]
        self.trust = [[PRIOR for _ in range(N_REL + N_NOISY)] for _ in range(N_AGENTS)]
        self.docs = [[[] for _ in range(world.n_slots)] for _ in range(N_AGENTS)]
        self.seq = 0
        self.comm = 0
        self.snapshot = [set() for _ in range(N_AGENTS)]
        self.touched = [set() for _ in range(N_AGENTS)]
        for agent in range(N_AGENTS):
            for slot in range(world.n_slots):
                self._store(agent, slot, world.init_belief[agent][slot], 0, PRIOR_ISSUER)

    def _store(self, agent, slot, value, time, issuer):
        self.seq += 1
        doc = {"slot": slot, "value": value, "time": time, "issuer": issuer, "seq": self.seq, "text": surface(self.world, slot, value)}
        self.docs[agent][slot].append(doc)
        return doc

    def _inbox(self, docs):
        boxes = [[] for _ in range(N_AGENTS)]
        for doc in docs:
            for agent in doc["receivers"]:
                boxes[agent].append(
                    {
                        "slot": doc["slot"],
                        "value": doc["value"],
                        "time": doc["time"],
                        "issuer": doc["issuer"],
                        "text": doc["text"],
                    }
                )
        return boxes

    def _schema_value(self, agent, slot):
        parent = self.world.parent.get(slot)
        if parent is None:
            return None
        parent_v = self.belief[agent][parent]
        if parent_v in (None, "EXPIRED"):
            return "EXPIRED" if parent_v == "EXPIRED" else None
        return f"via-{parent_v}"

    def _local_commit(self, agent, slot, extra, flags):
        schema_v = self._schema_value(agent, slot) if flags["dep"] else None
        value, committed, margin, issuer = decide(
            self.belief[agent][slot], extra, self.trust[agent], schema_v, flags
        )
        if committed and value != self.belief[agent][slot]:
            self.belief[agent][slot] = value
            times = [d["time"] for d in extra if d["value"] == value]
            self.belief_time[agent][slot] = max(times) if times else self.belief_time[agent][slot]
            self.last_issuer[agent][slot] = issuer if issuer is not None else PRIOR_ISSUER
            self.touched[agent].add(slot)
        return committed, margin

    def _update_trust(self, agent, docs, flags):
        if not flags["trust"]:
            return
        by_slot = group_docs(docs)
        for slot, packet in by_slot.items():
            buckets = defaultdict(set)
            for doc in packet:
                if doc["issuer"] < 0:
                    continue
                buckets[doc["value"]].add(doc["issuer"])
            if not buckets:
                continue
            held = self.belief[agent][slot]
            if held in buckets:
                for issuer in buckets[held]:
                    self.trust[agent][issuer] = clip_trust(self.trust[agent][issuer] + CONFIRM_BONUS)
            ranked = sorted(buckets.items(), key=lambda kv: len(kv[1]), reverse=True)
            if len(ranked[0][1]) < 2:
                continue
            if len(ranked) > 1 and len(ranked[1][1]) >= len(ranked[0][1]):
                continue
            agreed = ranked[0][0]
            for issuer in ranked[0][1]:
                self.trust[agent][issuer] = clip_trust(self.trust[agent][issuer] + TRUST_UP)
            for value, members in ranked[1:]:
                if value == agreed:
                    continue
                for issuer in members:
                    self.trust[agent][issuer] = clip_trust(self.trust[agent][issuer] - TRUST_DOWN)

    def _revise_agent(self, agent, packet, flags):
        by_slot = group_docs(packet)
        order = list(self.world.mutable_ids) + list(self.world.stable_ids) + list(self.world.derived_ids)
        contested = []
        for slot in order:
            extra = list(by_slot.get(slot, []))
            committed, margin = self._local_commit(agent, slot, extra, flags)
            if not committed and extra:
                contested.append((margin, slot, extra))
        return contested

    def _query_round(self, flags):
        if not flags["query"]:
            return
        for agent in range(N_AGENTS):
            packet = self._last_boxes[agent]
            by_slot = group_docs(packet)
            near = []
            for slot, extra in by_slot.items():
                scores, issuers, _best = unique_issuer_scores(extra, self.trust[agent], flags["trust"])
                belief = self.belief[agent][slot]
                challengers = [(value, score) for value, score in scores.items() if value != belief]
                if not challengers:
                    continue
                value, score = max(challengers, key=lambda item: item[1])
                if len(issuers[value]) != 1:
                    continue
                if score < PRIOR + 0.08 or score >= STICK:
                    continue
                near.append((STICK - score, slot, extra, value))
            near.sort(key=lambda item: item[0])
            for _gap, slot, extra, wanted in near[:MAX_QUERIES]:
                asked = False
                for shift in range(1, N_AGENTS):
                    peer = (agent + shift) % N_AGENTS
                    issuer = self.last_issuer[peer][slot]
                    if self.belief[peer][slot] != wanted:
                        continue
                    if issuer < 0 and issuer != SCHEMA:
                        continue
                    if any(d["issuer"] == issuer and d["value"] == wanted for d in extra):
                        continue
                    self.comm += 2
                    asked = True
                    extra.append(
                        {
                            "slot": slot,
                            "value": wanted,
                            "time": self.belief_time[peer][slot],
                            "issuer": issuer,
                            "text": surface(self.world, slot, wanted),
                            "seq": 0,
                        }
                    )
                    self._local_commit(agent, slot, extra, flags)
                    break
                if not asked:
                    continue

    def _relay(self, flags):
        if not flags["relay"]:
            return
        pushes = []
        for agent in range(N_AGENTS):
            candidates = [slot for slot in self.touched[agent] if self.last_issuer[agent][slot] >= 0 or self.last_issuer[agent][slot] == SCHEMA]
            self.rng.shuffle(candidates)
            sent = 0
            for slot in candidates:
                if sent >= MAX_PUSHES:
                    break
                peers = [i for i in range(N_AGENTS) if i != agent and self.belief[i][slot] != self.belief[agent][slot]]
                self.rng.shuffle(peers)
                for peer in peers[:FANOUT]:
                    pushes.append((agent, peer, slot))
                    sent += 1
        for agent, peer, slot in pushes:
            issuer = self.last_issuer[agent][slot]
            if issuer < 0 and issuer != SCHEMA:
                continue
            mark = (agent, peer, slot, self.belief[agent][slot])
            if mark in self._sent:
                continue
            self._sent.add(mark)
            extra = [
                {
                    "slot": slot,
                    "value": self.belief[agent][slot],
                    "time": max(1, self.belief_time[agent][slot]),
                    "issuer": issuer,
                    "text": surface(self.world, slot, self.belief[agent][slot]),
                    "seq": 0,
                }
            ]
            local = [d for d in self._last_boxes[peer] if d["slot"] == slot]
            if any(d["issuer"] == issuer and d["value"] == extra[0]["value"] for d in local):
                continue
            self.comm += 1
            self._local_commit(peer, slot, local + extra, flags)
            for child in self.world.children.get(slot, []):
                child_local = [d for d in self._last_boxes[peer] if d["slot"] == child]
                self._local_commit(peer, child, child_local, flags)

    def _naive_gossip(self, boxes):
        for agent in range(N_AGENTS):
            if not boxes[agent]:
                continue
            doc = max(boxes[agent], key=lambda d: d["time"])
            peer = (agent + 1 + self.rng.randrange(N_AGENTS - 1)) % N_AGENTS
            self.comm += 1
            self._store(peer, doc["slot"], doc["value"], doc["time"], doc["issuer"])

    def _crh_trust(self, agent, docs, shrink=True):
        claims = [
            (doc["slot"], doc["value"], doc["issuer"])
            for doc in docs
            if doc["issuer"] is not None and doc["issuer"] >= 0
        ]
        _truth, _log_w, learned = crh_fit(claims, shrink=shrink)
        for issuer, weight in learned.items():
            self.trust[agent][issuer] = weight

    def _discover(self, agent, packet, mode):
        claims = [
            (doc["slot"], doc["value"], doc["issuer"])
            for doc in packet
            if doc["issuer"] is not None and doc["issuer"] >= 0
        ]
        if not claims:
            return
        by_slot = group_docs(packet)
        if mode == "majority":
            truth = majority_truth(claims, lambda slot: self.belief[agent][slot])
        else:
            truth, _log_w, _learned = crh_fit(claims)
        for slot, value in truth.items():
            if value == self.belief[agent][slot]:
                continue
            extra = by_slot.get(slot, [])
            times = [doc["time"] for doc in extra if doc["value"] == value]
            issuers = [doc["issuer"] for doc in extra if doc["value"] == value and doc["issuer"] >= 0]
            self.belief[agent][slot] = value
            if times:
                self.belief_time[agent][slot] = max(times)
            if issuers:
                self.last_issuer[agent][slot] = issuers[0]

    def _init_board(self):
        self.central_belief = []
        for slot in range(self.world.n_slots):
            votes = defaultdict(int)
            for agent in range(N_AGENTS):
                votes[self.belief[agent][slot]] += 1
            self.central_belief.append(max(votes, key=lambda value: (votes[value], str(value))))
        self.central_trust = [PRIOR for _ in range(N_REL + N_NOISY)]
        self.central_time = [0] * self.world.n_slots
        self.central_issuer = [PRIOR_ISSUER] * self.world.n_slots

    def _pool_crh_step(self, pool):
        claims = [
            (doc["slot"], doc["value"], doc["issuer"])
            for doc in pool
            if doc["issuer"] is not None and doc["issuer"] >= 0
        ]
        truth, _log_w, _learned = crh_fit(claims)
        changed = []
        for slot, value in truth.items():
            if value == self.central_belief[slot]:
                continue
            self.central_belief[slot] = value
            times = [doc["time"] for doc in pool if doc["slot"] == slot and doc["value"] == value]
            if times:
                self.central_time[slot] = max(times)
            changed.append(slot)
        for agent in range(N_AGENTS):
            for slot in changed:
                self.belief[agent][slot] = self.central_belief[slot]
        self.comm += N_AGENTS * len(changed)

    def _apply_board(self, pending_slots):
        for slot in pending_slots:
            for agent in range(N_AGENTS):
                self.belief[agent][slot] = self.central_belief[slot]
                self.belief_time[agent][slot] = self.central_time[slot]
        self.comm += N_AGENTS * len(pending_slots)

    def _pool_vote_step(self, pool, mode):
        claims = [
            (doc["slot"], doc["value"], doc["issuer"])
            for doc in pool
            if doc["issuer"] is not None and doc["issuer"] >= 0
        ]
        by_slot = group_docs(pool)
        changed = []
        if mode == "poolmaj":
            truth = majority_truth(claims, lambda slot: self.central_belief[slot])
            for slot, value in truth.items():
                if value == self.central_belief[slot]:
                    continue
                self.central_belief[slot] = value
                times = [doc["time"] for doc in pool if doc["slot"] == slot and doc["value"] == value]
                if times:
                    self.central_time[slot] = max(times)
                changed.append(slot)
            self._apply_board(changed)
            return
        for slot, extra in by_slot.items():
            buckets = defaultdict(set)
            for doc in extra:
                if doc["issuer"] is None or doc["issuer"] < 0:
                    continue
                buckets[doc["value"]].add(doc["issuer"])
            if not buckets:
                continue
            scores = {value: len(members) for value, members in buckets.items()}
            belief = self.central_belief[slot]
            scores[belief] = scores.get(belief, 0) + 1
            ranked = sorted(scores.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
            best_v, best_s = ranked[0]
            second_s = ranked[1][1] if len(ranked) > 1 else 0
            raw = len(buckets.get(best_v, ()))
            if best_v == belief or raw < 2 or best_s < second_s + 1:
                continue
            self.central_belief[slot] = best_v
            times = [doc["time"] for doc in extra if doc["value"] == best_v]
            if times:
                self.central_time[slot] = max(times)
            changed.append(slot)
        self._apply_board(changed)

    def _pool(self, docs):
        pool = []
        seen = set()
        for doc in docs:
            key = (doc["slot"], doc["value"], doc["time"], doc["issuer"])
            if key in seen:
                continue
            seen.add(key)
            pool.append(
                {
                    "slot": doc["slot"],
                    "value": doc["value"],
                    "time": doc["time"],
                    "issuer": doc["issuer"],
                    "text": doc["text"],
                }
            )
        return pool

    def deliver(self, docs, epoch):
        flags = self.flags
        if flags["mode"] == "static":
            return
        boxes = self._inbox(docs)
        if flags["mode"] in ("central", "poolcrh", "poolmaj", "poolcount"):
            pool = self._pool(docs)
            if epoch == 1:
                self._init_board()
            if flags["mode"] == "poolcrh":
                self._pool_crh_step(pool)
            elif flags["mode"] in ("poolmaj", "poolcount"):
                self._pool_vote_step(pool, flags["mode"])
            else:
                self._central_step(pool, flags)
            return
        if flags["mode"] in ("majority", "crh"):
            for agent, packet in enumerate(boxes):
                self._discover(agent, packet, flags["mode"])
            return
        if flags["mode"] in ("timestamp", "reset", "ltm"):
            for agent, packet in enumerate(boxes):
                for doc in packet:
                    self._store(agent, doc["slot"], doc["value"], doc["time"], doc["issuer"])
            if flags["social"]:
                self._naive_gossip(boxes)
            if flags["mode"] == "reset" and epoch % RESET_EVERY == 0:
                for agent in range(N_AGENTS):
                    keep = set()
                    for slot in range(self.world.n_slots):
                        for doc in self.docs[agent][slot]:
                            if doc["time"] > epoch - RESET_EVERY:
                                keep.add(doc["seq"])
                    self.snapshot[agent] = keep
            return
        self._last_boxes = boxes
        self.touched = [set() for _ in range(N_AGENTS)]
        self._sent = set()
        for agent, packet in enumerate(boxes):
            if flags["trust"]:
                self._crh_trust(agent, packet)
            self._revise_agent(agent, packet, flags)
        self._relay(flags)

    def _central_step(self, pool, flags):
        by_slot = group_docs(pool)
        scratch = list(self.trust[0])
        self.trust[0] = list(self.central_trust)
        org_w = {}
        pool_claims = [
            (doc["slot"], doc["value"], doc["issuer"])
            for doc in pool
            if doc["issuer"] is not None and doc["issuer"] >= 0
        ]
        if flags.get("org") and flags["trust"]:
            global_w, org_w = crh_org_weights(pool_claims, shrink=flags.get("shrink", True))
            updated = [global_w.get(i, PRIOR) for i in range(N_REL + N_NOISY)]
        elif flags["trust"]:
            self._crh_trust(0, pool, shrink=flags.get("shrink", True))
            updated = list(self.trust[0])
        else:
            updated = list(self.trust[0])
        self.central_trust = updated
        self.trust[0] = scratch
        clusters = None
        if flags.get("indep"):
            provisional, _log_w, _learned = crh_fit(pool_claims)
            clusters = copy_clusters(pool, provisional)
        saved_belief = list(self.central_belief)
        saved_time = list(self.central_time)
        saved_issuer = list(self.central_issuer)
        groups = [
            list(self.world.mutable_ids) + list(self.world.stable_ids),
            list(self.world.derived_ids),
        ]
        pending = []
        for group in groups:
            group_pending = []
            for slot in group:
                extra = list(by_slot.get(slot, []))
                schema_v = None
                if flags["dep"]:
                    parent = self.world.parent.get(slot)
                    if parent is not None:
                        parent_v = self.central_belief[parent]
                        schema_v = "EXPIRED" if parent_v == "EXPIRED" else f"via-{parent_v}"
                trust_now = updated
                if org_w:
                    org = slot % len(ORGS)
                    trust_now = [org_w.get((org, i), updated[i]) for i in range(len(updated))]
                value, committed, margin, issuer = decide(
                    self.central_belief[slot], extra, trust_now, schema_v, flags, clusters
                )
                if committed and value != self.central_belief[slot]:
                    times = [d["time"] for d in extra if d["value"] == value]
                    stamp = max(times) if times else self.central_time[slot]
                    group_pending.append((margin, slot, value, issuer, stamp))
            for _margin, slot, value, issuer, stamp in group_pending:
                self.central_belief[slot] = value
                self.central_time[slot] = stamp
                self.central_issuer[slot] = issuer if issuer is not None else PRIOR_ISSUER
            pending.extend(group_pending)
        pending.sort(key=lambda item: (item[0], item[1]), reverse=True)
        if self.budget is not None:
            self.central_belief = saved_belief
            self.central_time = saved_time
            self.central_issuer = saved_issuer
            moved = {slot for _margin, slot, _value, _issuer, _stamp in pending}
            chosen = pending[: self.budget]
            chosen_slots = {slot for _margin, slot, _value, _issuer, _stamp in chosen}
            kept = []
            for item in chosen:
                _margin, slot, _value, _issuer, _stamp = item
                parent = self.world.parent.get(slot)
                if parent is not None and parent in moved and parent not in chosen_slots:
                    continue
                kept.append(item)
            pending = kept
        pushed = []
        for _margin, slot, value, issuer, stamp in pending:
            self.central_belief[slot] = value
            self.central_time[slot] = stamp
            self.central_issuer[slot] = issuer if issuer is not None else PRIOR_ISSUER
            pushed.append(slot)
        for agent in range(N_AGENTS):
            if self.budget is None:
                self.belief[agent] = list(self.central_belief)
                self.belief_time[agent] = list(self.central_time)
            else:
                for slot in pushed:
                    self.belief[agent][slot] = self.central_belief[slot]
                    self.belief_time[agent][slot] = self.central_time[slot]
        self.comm += N_AGENTS * len(pushed)

    def answer(self, agent, slot, epoch):
        mode = self.flags["mode"]
        if mode in ("evolv", "central", "static", "majority", "crh", "poolcrh", "poolmaj", "poolcount"):
            return self.belief[agent][slot]
        best = None
        best_key = None
        for doc in self.docs[agent][slot]:
            if mode == "timestamp" and doc["time"] < epoch - (TS_WINDOW - 1):
                continue
            if mode == "reset" and self.snapshot[agent] and doc["seq"] not in self.snapshot[agent]:
                continue
            if mode == "reset" and not self.snapshot[agent] and doc["time"] < epoch - (RESET_EVERY - 1):
                continue
            key = (doc["time"], doc["seq"])
            if best_key is None or key > best_key:
                best_key = key
                best = doc
        if best is None:
            return "NONE"
        return best["value"]

    def evaluate(self, epoch):
        return [[self.answer(agent, slot, epoch) for slot in range(self.world.n_slots)] for agent in range(N_AGENTS)]


def empty_counts():
    return dict(
        all_hit=0,
        all_n=0,
        changed_hit=0,
        changed_n=0,
        stable_hit=0,
        stable_n=0,
        derived_hit=0,
        derived_n=0,
        obsolete=0,
        obsolete_n=0,
        unnec=0,
        unnec_n=0,
        delay_sum=0.0,
        delay_n=0,
        comm=0.0,
        epoch_hit=[0] * EPOCHS,
        epoch_n=[0] * EPOCHS,
    )


def consume(counts, world, answers_by_epoch, comm):
    for slot, versions in world.versions.items():
        for value, start, end in versions:
            if end < start:
                continue
            for agent in range(N_AGENTS):
                delay = None
                for epoch in range(start, end + 1):
                    if answers_by_epoch[epoch - 1][agent][slot] == value:
                        delay = epoch - start
                        break
                if delay is None:
                    delay = end - start + 1
                counts["delay_sum"] += delay
                counts["delay_n"] += 1
    for epoch_i, grid in enumerate(answers_by_epoch):
        for agent in range(N_AGENTS):
            for slot in range(world.n_slots):
                pred = grid[agent][slot]
                truth = world.timeline[epoch_i][slot]
                counts["all_n"] += 1
                counts["epoch_n"][epoch_i] += 1
                if pred == truth:
                    counts["all_hit"] += 1
                    counts["epoch_hit"][epoch_i] += 1
                if slot in world.stable_ids:
                    counts["stable_n"] += 1
                    counts["unnec_n"] += 1
                    if pred == truth:
                        counts["stable_hit"] += 1
                    elif pred not in ("NONE", truth):
                        counts["unnec"] += 1
                changed_now = bool(world.history_at[epoch_i][slot]) and truth != "EXPIRED"
                if slot in world.mutable_ids and changed_now:
                    counts["changed_n"] += 1
                    if pred == truth:
                        counts["changed_hit"] += 1
                if slot in world.derived_ids and changed_now:
                    counts["derived_n"] += 1
                    if pred == truth:
                        counts["derived_hit"] += 1
                if world.history_at[epoch_i][slot]:
                    counts["obsolete_n"] += 1
                    if pred in world.history_at[epoch_i][slot] and pred != truth:
                        counts["obsolete"] += 1
    counts["comm"] = comm / float(N_AGENTS * EPOCHS)


def run_method(world, method, seed, budget=None):
    memory = Memory(world, method, seed, budget=budget)
    answers = []
    for epoch_i, docs in enumerate(world.visible):
        memory.deliver(docs, epoch_i + 1)
        answers.append(memory.evaluate(epoch_i + 1))
    counts = empty_counts()
    consume(counts, world, answers, memory.comm)
    return counts, answers


def ratio(hit, n):
    if n == 0:
        return None
    return hit / n


def pack_metrics(counts):
    packed = {
        "cfa": ratio(counts["all_hit"], counts["all_n"]),
        "changed": ratio(counts["changed_hit"], counts["changed_n"]),
        "stable": ratio(counts["stable_hit"], counts["stable_n"]),
        "derived": ratio(counts["derived_hit"], counts["derived_n"]),
        "obsolete": ratio(counts["obsolete"], counts["obsolete_n"]),
        "unnec": ratio(counts["unnec"], counts["unnec_n"]),
        "delay": ratio(counts["delay_sum"], counts["delay_n"]),
        "comm": counts["comm"],
        "epoch_cfa": [ratio(counts["epoch_hit"][i], counts["epoch_n"][i]) for i in range(EPOCHS)],
    }
    if packed["changed"] is not None and packed["stable"] is not None:
        packed["sp"] = 0.5 * (packed["changed"] + packed["stable"])
    else:
        packed["sp"] = None
    if any(v is None for v in packed.values() if not isinstance(v, list)) or any(
        v is None for v in packed["epoch_cfa"]
    ):
        raise RuntimeError("metric denominator was empty")
    return packed


METHODS = [
    "Static",
    "Timestamp",
    "Reset",
    "LTM",
    "Majority",
    "CRH",
    "PoolCRH",
    "Central",
    "EvolvRAG",
    "NoTrust",
    "NoRelay",
    "NoDep",
    "NoExpire",
]
ABLATIONS = ["NoTrust", "NoRelay", "NoDep", "NoExpire"]
BUDGETS = [1, 2, 4, 8, 16]
PRIMARY = "sp"


def build_trace(world, answers):
    best = None
    best_score = -1
    for epoch_i, docs in enumerate(world.visible):
        by_slot = defaultdict(list)
        for doc in docs:
            by_slot[doc["slot"]].append(doc)
        for slot, packet in by_slot.items():
            if slot not in world.mutable_ids:
                continue
            truth = world.timeline[epoch_i][slot]
            prev = world.truth0[slot] if epoch_i == 0 else world.timeline[epoch_i - 1][slot]
            if truth == prev or truth == "EXPIRED":
                continue
            values = {d["value"] for d in packet}
            if truth not in values or len(values) < 2:
                continue
            saw_truth = set()
            for doc in packet:
                if doc["value"] == truth:
                    saw_truth.update(doc["receivers"])
            answers_now = [answers[epoch_i][a][slot] for a in range(N_AGENTS)]
            adopted = [
                a
                for a in range(N_AGENTS)
                if answers_now[a] == truth and world.init_belief[a][slot] != truth
            ]
            held = [a for a in range(N_AGENTS) if answers_now[a] == prev]
            if len(adopted) < 1 or len(held) < 2 or len(saw_truth) < 2:
                continue
            child = world.children[slot][0] if world.children[slot] else None
            child_ok = False
            child_pack = None
            if child is not None:
                child_truth = world.timeline[epoch_i][child]
                child_answers = [answers[epoch_i][a][child] for a in range(N_AGENTS)]
                child_ok = child_truth.startswith("via-") and any(a == child_truth for a in child_answers)
                child_pack = {
                    "slot": child,
                    "name": world.names[child],
                    "role": world.roles[child],
                    "query": query_text(world, child),
                    "value": child_truth,
                    "answers": child_answers,
                }
            true_docs = [d for d in packet if d["value"] == truth]
            rival = next(d for d in packet if d["value"] != truth)
            score = len(adopted) * 3 + len(held) + (4 if child_ok else 0) + len(saw_truth)
            if score <= best_score:
                continue
            best_score = score
            best = {
                "epoch": epoch_i + 1,
                "slot": slot,
                "name": world.names[slot],
                "role": world.roles[slot],
                "query": query_text(world, slot),
                "old_value": prev,
                "new_value": truth,
                "true_document": true_docs[0]["text"],
                "true_issuer": true_docs[0]["issuer"],
                "true_time": true_docs[0]["time"],
                "rival_document": rival["text"],
                "rival_value": rival["value"],
                "rival_issuer": rival["issuer"],
                "rival_time": rival["time"],
                "observers": sorted(saw_truth),
                "missed": [i for i in range(N_AGENTS) if i not in saw_truth],
                "adopted": adopted,
                "held_previous": held,
                "init": [world.init_belief[a][slot] for a in range(N_AGENTS)],
                "evolv_answers": answers_now,
                "child": child_pack,
            }
    return best


def saturated(name, value):
    # Revision traffic may be exactly zero for a method that does not send.
    if name in ("comm", "sp"):
        return False
    if name == "delay":
        shown = round(value, 2)
        return shown <= 0 or shown == 1
    shown = round(value, 3)
    return shown <= 0 or shown >= 1


def signflip_p(diffs):
    n = len(diffs)
    if n == 0 or n > 16:
        raise RuntimeError("sign-flip test expects 1 to 16 paired seeds")
    obs = abs(sum(diffs))
    if obs <= 1e-12:
        return 1.0
    count = 0
    total = 1 << n
    for mask in range(total):
        signed = 0.0
        for i, diff in enumerate(diffs):
            signed += diff if (mask >> i) & 1 else -diff
        if abs(signed) + 1e-9 >= obs:
            count += 1
    return count / total


def select_thresholds(seeds):
    global STICK, MARGIN, ONE_ISSUER_TRUST, CRH_ALPHA
    worlds = [generate_trajectory(seed, DEFAULT_RATE, DEFAULT_POBS) for seed in seeds]
    grid = []
    for stick in (0.42, 0.58, 0.74):
        for margin in (0.04, 0.08, 0.16):
            for one in (0.50, 0.66, 0.82):
                for alpha in (2.0, 4.0, 8.0):
                    STICK, MARGIN, ONE_ISSUER_TRUST, CRH_ALPHA = stick, margin, one, alpha
                    scores = []
                    accs = []
                    for seed, world in zip(seeds, worlds):
                        counts, _answers = run_method(world, "EvolvRAG", seed)
                        metrics = pack_metrics(counts)
                        scores.append(metrics["sp"])
                        accs.append(metrics["cfa"])
                    mu, sd = mean_std(scores)
                    acc_mu, _acc_sd = mean_std(accs)
                    grid.append(
                        {
                            "stick": stick,
                            "margin": margin,
                            "one": one,
                            "alpha": alpha,
                            "sp": mu,
                            "sp_std": sd,
                            "cfa": acc_mu,
                        }
                    )
    best = max(grid, key=lambda row: row["sp"])
    near = [row for row in grid if best["sp"] - row["sp"] <= 0.01]

    def distance(row):
        return (
            abs(row["stick"] - 0.58)
            + abs(row["margin"] - 0.08)
            + abs(row["one"] - 0.66)
            + abs(row["alpha"] - 4.0) / 10.0
        )

    chosen = min(near, key=distance)
    STICK = chosen["stick"]
    MARGIN = chosen["margin"]
    ONE_ISSUER_TRUST = chosen["one"]
    CRH_ALPHA = chosen["alpha"]
    return grid, chosen


def summarize(bucket):
    summary = []
    for (method, rate, p_obs), metric_map in bucket.items():
        cell = {"method": method, "rate": rate, "p_obs": p_obs}
        for name, values in metric_map.items():
            if name == "epoch_cfa":
                cols = list(zip(*values))
                cell["epoch_cfa_mean"] = [mean_std(list(col))[0] for col in cols]
                cell["epoch_cfa_std"] = [mean_std(list(col))[1] for col in cols]
            else:
                mu, sd = mean_std(values)
                cell[name] = mu
                cell[name + "_std"] = sd
                cell[name + "_seeds"] = values
        summary.append(cell)
    return summary


def paired_tests(summary):
    def seeds_of(method, name):
        for cell in summary:
            if (
                cell["method"] == method
                and abs(cell["rate"] - DEFAULT_RATE) < 1e-9
                and abs(cell["p_obs"] - DEFAULT_POBS) < 1e-9
            ):
                return cell[name + "_seeds"]
        return None

    report = []
    for metric in ("sp", "cfa", "changed", "stable"):
        base = seeds_of("EvolvRAG", metric)
        if base is None:
            continue
        for other in ("CRH", "PoolCRH", "Central", "LTM", "Majority"):
            alt = seeds_of(other, metric)
            if alt is None or len(alt) != len(base):
                continue
            diffs = [a - b for a, b in zip(base, alt)]
            mu, sd = mean_std(diffs)
            report.append(
                {
                    "metric": metric,
                    "versus": other,
                    "mean_diff": mu,
                    "std_diff": sd,
                    "p_signflip": signflip_p(diffs),
                    "n": len(diffs),
                }
            )
    return report


def main():
    global STICK, MARGIN, ONE_ISSUER_TRUST, CRH_ALPHA
    smoke = bool(os.environ.get("SMOKE"))
    if smoke:
        seeds = [0]
        rates = [DEFAULT_RATE]
        p_list = [DEFAULT_POBS]
        grid, chosen = [], {
            "stick": STICK,
            "margin": MARGIN,
            "one": ONE_ISSUER_TRUST,
            "alpha": CRH_ALPHA,
            "sp": None,
        }
    else:
        grid, chosen = select_thresholds(SEEDS_DEV)
        print(
            "LOCKED",
            chosen["stick"],
            chosen["margin"],
            chosen["one"],
            chosen["alpha"],
            f"dev_sp={chosen['sp']:.4f}",
        )
        seeds = SEEDS_TEST
        rates = RATES
        p_list = P_OBS_LIST
    bucket = defaultdict(lambda: defaultdict(list))
    saved_trace = None
    default_worlds = {}
    for seed in seeds:
        for rate in rates:
            for p_obs in p_list:
                world = generate_trajectory(seed, rate, p_obs)
                if abs(rate - DEFAULT_RATE) < 1e-9 and abs(p_obs - DEFAULT_POBS) < 1e-9:
                    default_worlds[seed] = world
                for method in METHODS:
                    if method in ABLATIONS and not (
                        abs(rate - DEFAULT_RATE) < 1e-9 and abs(p_obs - DEFAULT_POBS) < 1e-9
                    ):
                        continue
                    counts, answers = run_method(world, method, seed)
                    metrics = pack_metrics(counts)
                    key = (method, rate, p_obs)
                    for name, value in metrics.items():
                        bucket[key][name].append(value)
                    if (
                        saved_trace is None
                        and seed == seeds[0]
                        and abs(rate - DEFAULT_RATE) < 1e-9
                        and abs(p_obs - DEFAULT_POBS) < 1e-9
                        and method == "EvolvRAG"
                    ):
                        saved_trace = build_trace(world, answers)
    budget_cells = []
    if not smoke:
        for budget in BUDGETS:
            store = defaultdict(list)
            for seed, world in default_worlds.items():
                counts, _answers = run_method(world, "Central", seed, budget=budget)
                metrics = pack_metrics(counts)
                for name, value in metrics.items():
                    if name != "epoch_cfa":
                        store[name].append(value)
            cell = {"method": "CentralBudget", "budget": budget, "rate": DEFAULT_RATE, "p_obs": DEFAULT_POBS}
            for name, values in store.items():
                mu, sd = mean_std(values)
                cell[name] = mu
                cell[name + "_std"] = sd
            budget_cells.append(cell)
            print("BUDGET", budget, f"sp={cell['sp']:.4f}", f"cfa={cell['cfa']:.4f}", f"comm={cell['comm']:.4f}")
    summary = summarize(bucket)
    tests = [] if smoke else paired_tests(summary)
    for row in tests:
        if row["metric"] == "sp":
            print(
                "TEST",
                row["metric"],
                "vs",
                row["versus"],
                f"diff={row['mean_diff']:.4f}",
                f"p={row['p_signflip']:.4f}",
            )
    payload = {
        "config": {
            "agents": N_AGENTS,
            "stable_slots": N_STABLE,
            "mutable_slots": N_MUTABLE,
            "derived_slots": N_DERIVED,
            "epochs": EPOCHS,
            "dev_seeds": SEEDS_DEV,
            "seeds": seeds,
            "rates": rates,
            "p_obs": p_list,
            "p_expire": P_EXPIRE,
            "p_init_err": P_INIT_ERR,
            "p_second_issuer": P_SECOND,
            "p_collude": P_COLLUDE,
            "p_override": P_OVERRIDE,
            "margin": MARGIN,
            "stick": STICK,
            "one_issuer_trust": ONE_ISSUER_TRUST,
            "crh_alpha": CRH_ALPHA,
            "schema_trust": SCHEMA_TRUST,
            "fanout": FANOUT,
            "heartbeat": 0,
            "gossip_baselines": False,
            "query": False,
            "primary": PRIMARY,
            "timestamp_window": TS_WINDOW,
            "reset_every": RESET_EVERY,
            "default_rate": DEFAULT_RATE,
            "default_p_obs": DEFAULT_POBS,
            "labels_visible": False,
            "locked": chosen,
        },
        "sensitivity": grid,
        "budget": budget_cells,
        "tests": tests,
        "cells": summary,
        "trace": saved_trace,
    }
    if not smoke:
        RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if saved_trace is not None:
            DATA_PATH.write_text(json.dumps(saved_trace, indent=2), encoding="utf-8")
    print_tables(summary)
    bad = []
    watch = ["cfa", "changed", "stable", "derived", "obsolete", "unnec", "delay", "sp"]
    for cell in summary:
        if abs(cell["rate"] - DEFAULT_RATE) > 1e-9 or abs(cell["p_obs"] - DEFAULT_POBS) > 1e-9:
            if cell["method"] not in ("EvolvRAG", "Central", "CRH", "LTM", "Static"):
                continue
        for name in watch:
            if saturated(name, cell[name]):
                bad.append((cell["method"], cell["rate"], cell["p_obs"], name, round(cell[name], 4)))
    print("SATURATED", len(bad))
    for item in bad:
        print(" ", item)
    print("TRACE", json.dumps(saved_trace, ensure_ascii=False)[:1500] if saved_trace else None)


def print_tables(summary):
    def find(method, rate, p_obs):
        for cell in summary:
            if cell["method"] == method and abs(cell["rate"] - rate) < 1e-9 and abs(cell["p_obs"] - p_obs) < 1e-9:
                return cell
        return None

    headers = ["sp", "cfa", "changed", "stable", "derived", "obsolete", "unnec", "delay", "comm"]
    print("MAIN")
    for method in ["Static", "Timestamp", "Reset", "LTM", "Majority", "CRH", "PoolCRH", "Central", "EvolvRAG"]:
        cell = find(method, DEFAULT_RATE, DEFAULT_POBS)
        if cell is None:
            continue
        print(method, " ".join(f"{cell[h]:.4f}" for h in headers))
    print("ABLATION")
    for method in ["EvolvRAG", "NoTrust", "NoRelay", "NoDep", "NoExpire"]:
        cell = find(method, DEFAULT_RATE, DEFAULT_POBS)
        if cell is None:
            continue
        print(method, " ".join(f"{cell[h]:.4f}" for h in headers))
    print("BY_RATE")
    for rate in RATES:
        for method in ["Static", "LTM", "CRH", "Central", "EvolvRAG"]:
            cell = find(method, rate, DEFAULT_POBS)
            if cell is None:
                continue
            print(rate, method, f"{cell['sp']:.4f}", f"{cell['cfa']:.4f}", f"{cell['obsolete']:.4f}", f"{cell['comm']:.4f}")
    print("BY_OBS")
    for p_obs in P_OBS_LIST:
        for method in ["LTM", "CRH", "Central", "EvolvRAG"]:
            cell = find(method, DEFAULT_RATE, p_obs)
            if cell is None:
                continue
            print(
                p_obs,
                method,
                f"{cell['sp']:.4f}",
                f"{cell['cfa']:.4f}",
                f"{cell['comm']:.4f}",
                f"{cell['changed']:.4f}",
                f"{cell['stable']:.4f}",
            )


if __name__ == "__main__":
    main()
