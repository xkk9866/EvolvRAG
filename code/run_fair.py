"""Issuer recency on the full document list. Recipient lists are not an input.

A question names an entity. The conflict set is every passage whose text
contains that entity's name. Before weights are used, an issuer is collapsed
into another when its set of (entity, value, time) claims is a subset of the
other's. The answer is the latest passage from an issuer whose reliability
clears a floor. The envelope uses the larger of two label-free records: the
issuer's epoch-mixed society-wide weight, and its cumulative weight inside
the organization that owns the entity. A specialist is not silenced by a
mixed global average, and a generally reliable issuer is not silenced by a
thin organization. The decision uses the records from before the epoch.

The named system is chosen on seeds 600-615 before seeds 700-731 are read.
For each organization, the organization record is used when it clears more
issuers than the society-wide record. Otherwise the society-wide record is
used. Seeds 400-531 are not reused.
The score is the mean, over stationary, topic, open-rumor, and sybil worlds,
of (changed accuracy + retention + derived accuracy) / 3. Within 0.005 the
simpler candidate wins. An index that keeps a document only when it has two
or more recipients is scored and is not eligible.
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as rx
import run_rag as rag

ROOT = rx.ROOT
OUT = ROOT / "results" / "hier.json"
DEV = list(range(600, 616))
TEST = list(range(700, 732))
STRESS_SEEDS = list(range(700, 708))
SCALE_SEEDS = list(range(700, 704))
RATE = rx.DEFAULT_RATE
POBS = rx.DEFAULT_POBS
SELECT_REGIMES = ("stationary", "topic", "open_rumor", "sybil")
POOL_PRIOR_CLAIMS = 4
FILLER = ("bulletin", "memo", "posted", "internal", "staff", "notice", "archive", "note", "copy", "filed")

RULES = [
    {"name": "newest", "rule": "newest", "corpus": "full", "cutoff": 0.0, "eta": 1.0, "topic": False, "schema": False, "copy": False, "envelope": False, "hier": False, "cumulative": False},
    {"name": "majority", "rule": "majority", "corpus": "full", "cutoff": 0.0, "eta": 1.0, "topic": False, "schema": False, "copy": False, "envelope": False, "hier": False, "cumulative": False},
    {"name": "crh", "rule": "crh", "corpus": "full", "cutoff": 0.0, "eta": 1.0, "topic": False, "schema": False, "copy": False, "envelope": False, "hier": False, "cumulative": False},
    {"name": "gate", "rule": "gate", "corpus": "full", "cutoff": 0.0, "eta": 0.25, "topic": True, "schema": True, "copy": False, "envelope": False, "hier": False, "cumulative": False},
    {"name": "hier", "rule": "trusted", "corpus": "full", "cutoff": 0.55, "eta": 0.5, "margin": 1, "topic": False, "schema": True, "copy": True, "envelope": False, "hier": True, "cumulative": False},
    {"name": "hier-m0", "rule": "trusted", "corpus": "full", "cutoff": 0.55, "eta": 0.5, "margin": 0, "topic": False, "schema": True, "copy": True, "envelope": False, "hier": True, "cumulative": False},
    {"name": "hier-m2", "rule": "trusted", "corpus": "full", "cutoff": 0.55, "eta": 0.5, "margin": 2, "topic": False, "schema": True, "copy": True, "envelope": False, "hier": True, "cumulative": False},
    {"name": "hier-50", "rule": "trusted", "corpus": "full", "cutoff": 0.50, "eta": 0.5, "margin": 1, "topic": False, "schema": True, "copy": True, "envelope": False, "hier": True, "cumulative": False},
    {"name": "global", "rule": "trusted", "corpus": "full", "cutoff": 0.55, "eta": 0.5, "topic": False, "schema": True, "copy": True, "envelope": False, "hier": False, "cumulative": False},
    {"name": "org", "rule": "trusted", "corpus": "full", "cutoff": 0.55, "eta": 1.0, "topic": True, "schema": True, "copy": True, "envelope": False, "hier": False, "cumulative": True},
    {"name": "hier-nocopy", "rule": "trusted", "corpus": "full", "cutoff": 0.55, "eta": 0.5, "margin": 1, "topic": False, "schema": True, "copy": False, "envelope": False, "hier": True, "cumulative": False},
    {"name": "hier-noschema", "rule": "trusted", "corpus": "full", "cutoff": 0.55, "eta": 0.5, "margin": 1, "topic": False, "schema": False, "copy": True, "envelope": False, "hier": True, "cumulative": False},
    {"name": "shortcut-newest", "rule": "newest", "corpus": "shortcut", "cutoff": 0.0, "eta": 1.0, "topic": False, "schema": False, "copy": False, "envelope": False, "hier": False, "cumulative": False, "eligible": False},
]


def eligible(cfg):
    return cfg.get("eligible", True)


def simplicity(cfg):
    if cfg["rule"] == "newest":
        rank = 0
    elif cfg["rule"] == "majority":
        rank = 1
    elif cfg["rule"] == "crh":
        rank = 2
    elif cfg["rule"] == "trusted" and not cfg.get("envelope") and not cfg.get("hier") and not cfg.get("topic"):
        rank = 3
    elif cfg["rule"] == "trusted" and cfg.get("topic"):
        rank = 4
    elif cfg["rule"] == "trusted" and (cfg.get("envelope") or cfg.get("hier")):
        rank = 5
    else:
        rank = 6
    return (
        rank,
        0 if cfg["corpus"] == "local" else 1,
        1 if cfg.get("copy") else 0,
        1 if cfg.get("schema") else 0,
        0 if cfg.get("eta", 1) >= 1 else 1,
        float(cfg.get("cutoff") or 0.0),
    )


def entity_of_text(text, names):
    mark = text.find(" #")
    if mark < 0:
        return None
    digits = text[mark + 2 : mark + 5]
    if len(digits) != 3 or not digits.isdigit():
        return None
    slot = int(digits)
    if slot >= len(names) or names[slot].lower() not in text.lower():
        return None
    return slot


def stamp(world):
    for epoch_i, docs in enumerate(world.visible):
        for i, doc in enumerate(docs):
            doc["seq"] = epoch_i * 100000 + i


def paraphrase_doc(doc):
    rng = random.Random((doc["time"] + 3) * 10007 + doc["slot"] * 17 + int(doc["issuer"] or 0) * 3)
    extra = " ".join(rng.choice(FILLER) for _ in range(6))
    out = dict(doc)
    out["text"] = doc["text"] + " " + extra
    return out


def sybil_copies(world):
    rng = random.Random(world.names and 9001)
    next_id = rx.N_REL + rx.N_NOISY
    for epoch_docs in world.visible:
        extra = []
        for doc in list(epoch_docs):
            issuer = doc["issuer"]
            if not isinstance(issuer, int) or issuer < rx.N_REL or issuer >= rx.N_REL + rx.N_NOISY:
                continue
            for _ in range(3):
                clone = dict(doc)
                clone["issuer"] = next_id
                next_id += 1
                clone["text"] = doc["text"]
                extra.append(clone)
        epoch_docs.extend(extra)
    return world


def load_world(seed, regime, paraphrase=False):
    if regime == "sybil":
        world = rx.generate_trajectory(seed, RATE, POBS, regime="stationary")
        sybil_copies(world)
    else:
        world = rx.generate_trajectory(seed, RATE, POBS, regime=regime)
    if paraphrase:
        for epoch_docs in world.visible:
            for i, doc in enumerate(epoch_docs):
                epoch_docs[i] = paraphrase_doc(doc)
    stamp(world)
    return world


def copy_find(docs, names):
    """Point a copied issuer at the issuer whose claim triples contain its own.

    A sybil clone of one passage has a singleton triple set. It is linked to
    the earlier issuer who published that same entity, value, and time. Two
    issuers who merely agree on one fact, and then disagree on others, are
    left apart.
    """
    claims = defaultdict(set)
    for doc in docs:
        issuer = doc["issuer"]
        slot = entity_of_text(doc["text"], names)
        if isinstance(issuer, int) and issuer >= 0 and slot is not None:
            claims[issuer].add((slot, doc["value"], doc["time"]))
    owners = defaultdict(set)
    for issuer, triples in claims.items():
        for triple in triples:
            owners[triple].add(issuer)
    parent = {issuer: issuer for issuer in claims}

    def find(issuer):
        parent.setdefault(issuer, issuer)
        while parent[issuer] != issuer:
            parent[issuer] = parent[parent[issuer]]
            issuer = parent[issuer]
        return issuer

    for issuer, triples in claims.items():
        if not triples:
            continue
        candidates = None
        for triple in triples:
            group = owners[triple]
            candidates = set(group) if candidates is None else candidates & group
            if len(candidates) <= 1:
                break
        if not candidates:
            continue
        for other in candidates:
            if other == issuer:
                continue
            other_set = claims[other]
            subset = triples < other_set or (triples == other_set and issuer > other)
            if not subset:
                continue
            left, right = find(issuer), find(other)
            if left != right:
                parent[left] = right
    return find


def collapsed(docs, find):
    if find is None:
        return docs
    out = []
    for doc in docs:
        item = dict(doc)
        issuer = doc["issuer"]
        if isinstance(issuer, int) and issuer >= 0:
            item["issuer"] = find(issuer)
        out.append(item)
    return out


def weight_of(trust, issuer):
    if callable(trust):
        return trust(issuer)
    if not isinstance(issuer, int) or issuer < 0:
        return None
    if issuer >= len(trust):
        return rx.PRIOR
    return trust[issuer]


def pad(previous, length):
    if len(previous) >= length:
        return list(previous)
    return list(previous) + [rx.PRIOR] * (length - len(previous))


def trusted_pick(hits, belief, trust, cutoff):
    best = None
    best_key = None
    for hit in hits:
        weight = weight_of(trust, hit["issuer"])
        if weight is None or weight + 1e-9 < cutoff:
            continue
        key = (hit["time"], hit.get("seq", 0))
        if best_key is None or key > best_key:
            best = hit
            best_key = key
    if best is None:
        return belief, False
    return best["value"], True


def schema_value(world, row, slot):
    parent = world.parent.get(slot)
    if parent is None:
        return None
    parent_v = row[parent]
    return "EXPIRED" if parent_v == "EXPIRED" else f"via-{parent_v}"


def answer_slot(cfg, hits, belief, trust, schema_v):
    rule = cfg["rule"]
    if not hits and not (cfg.get("schema") and schema_v is not None):
        return belief
    if rule == "newest":
        if not hits:
            return belief
        return max(hits, key=lambda hit: (hit["time"], hit.get("seq", 0)))["value"]
    if rule == "majority":
        return rag.majority_value(hits, belief)
    if rule == "crh":
        return rag.crh_value(hits, belief)
    if rule == "gate":
        return rag.gate_value(hits, belief, trust, schema_v, cfg.get("schema", False))
    chosen, saw = trusted_pick(hits, belief, trust, cfg.get("cutoff", 0.36))
    if cfg.get("schema") and schema_v is not None and not saw:
        return schema_v
    return chosen


def claim_rows(docs, names, org=None):
    rows = []
    counts = defaultdict(int)
    seen = {}
    for doc in docs:
        slot = entity_of_text(doc["text"], names)
        issuer = doc["issuer"]
        if slot is None or not isinstance(issuer, int) or issuer < 0:
            continue
        if org is not None and slot % len(rx.ORGS) != org:
            continue
        seen[(slot, issuer)] = doc["value"]
    for (slot, issuer), value in seen.items():
        rows.append((slot, value, issuer))
        counts[issuer] += 1
    return rows, counts


class PoolState:
    """Global reliability, with an organization-level record when it has support.

    The decision weight is the claim-weighted pool of the two. An organization
    with fewer than two claims from an issuer contributes only the global
    record, so a thin context cannot freeze the belief at the initial value.
    """

    def __init__(self):
        width = rx.N_REL + rx.N_NOISY
        self.glob = rag.blank_weights()
        self.org_w = [[rx.PRIOR] * width for _ in range(len(rx.ORGS))]
        self.org_n = [[0] * width for _ in range(len(rx.ORGS))]

    def _ensure(self, issuer):
        if issuer < len(self.glob):
            return
        extra = issuer - len(self.glob) + 1
        self.glob.extend([rx.PRIOR] * extra)
        for org in range(len(rx.ORGS)):
            self.org_w[org].extend([rx.PRIOR] * extra)
            self.org_n[org].extend([0] * extra)

    def trust_fn(self, slot):
        org = slot % len(rx.ORGS)

        def trust(issuer, org=org):
            if not isinstance(issuer, int) or issuer < 0:
                return None
            self._ensure(issuer)
            glob = self.glob[issuer]
            n = self.org_n[org][issuer]
            if n < 2:
                return glob
            local = self.org_w[org][issuer]
            return (n * local + POOL_PRIOR_CLAIMS * glob) / (n + POOL_PRIOR_CLAIMS)

        return trust

    def update(self, docs, names, eta):
        self.glob = fit_weights(docs, self.glob, False, eta, names)
        for org in range(len(rx.ORGS)):
            rows, counts = claim_rows(docs, names, org=org)
            if not rows:
                continue
            base = pad(self.org_w[org], max(counts) + 1 if counts else len(self.org_w[org]))
            learned = rag.weights_from_claims(rows, base)
            for issuer, n in counts.items():
                if n < 2 or issuer < 0:
                    continue
                self._ensure(issuer)
                unit = learned[issuer]
                old_n = self.org_n[org][issuer]
                old_w = self.org_w[org][issuer]
                self.org_w[org][issuer] = unit if old_n == 0 else (old_n * old_w + n * unit) / (old_n + n)
                self.org_n[org][issuer] = old_n + n


def fit_weights(docs, previous, topic, eta, names):
    max_id = max([rx.N_REL + rx.N_NOISY - 1] + [doc["issuer"] for doc in docs if isinstance(doc["issuer"], int) and doc["issuer"] >= 0])
    if topic:
        mixed = []
        for org in range(len(rx.ORGS)):
            base = pad(previous[org], max_id + 1)
            claims = []
            for doc in docs:
                slot = entity_of_text(doc["text"], names)
                issuer = doc["issuer"]
                if slot is None or slot % len(rx.ORGS) != org:
                    continue
                if isinstance(issuer, int) and issuer >= 0:
                    claims.append((slot, doc["value"], issuer))
            fresh = rag.weights_from_claims(claims, base) if claims else base
            mixed.append(rag.mix(base, fresh, eta))
        return mixed
    base = pad(previous, max_id + 1)
    claims = []
    for doc in docs:
        slot = entity_of_text(doc["text"], names)
        issuer = doc["issuer"]
        if slot is None or not isinstance(issuer, int) or issuer < 0:
            continue
        claims.append((slot, doc["value"], issuer))
    if not claims:
        return base
    fresh = rag.weights_from_claims(claims, base)
    return rag.mix(base, fresh, eta)


def index_hits(docs, names):
    buckets = defaultdict(list)
    for doc in docs:
        slot = entity_of_text(doc["text"], names)
        if slot is not None:
            buckets[slot].append(doc)
    return buckets


class TrustBook:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pool = PoolState() if cfg.get("pool") else None
        self.org = None
        self.org_n = None
        if cfg.get("topic") and self.pool is None and not cfg.get("envelope") and not cfg.get("hier"):
            self.plain = [rag.blank_weights() for _ in range(len(rx.ORGS))]
        else:
            self.plain = rag.blank_weights()

    def for_slot(self, slot):
        if self.cfg.get("hier") and self.org is not None:
            org_i = slot % len(rx.ORGS)
            org_vec = self.org[org_i]
            counts = self.org_n[org_i]
            glob = self.plain
            tau = float(self.cfg.get("cutoff", 0.55))
            margin = int(self.cfg.get("margin", 1))
            width = max(len(glob), len(org_vec))
            soc_high = 0
            org_high = 0
            for issuer in range(width):
                if weight_of(glob, issuer) + 1e-9 >= tau:
                    soc_high += 1
                if weight_of(org_vec, issuer) + 1e-9 >= tau:
                    org_high += 1
            use_org = org_high >= soc_high + margin

            def trust(issuer, org_vec=org_vec, glob=glob, use_org=use_org):
                local_w = weight_of(org_vec, issuer)
                global_w = weight_of(glob, issuer)
                if use_org and local_w is not None:
                    return local_w
                return global_w

            return trust
        if self.cfg.get("envelope") and self.org is not None:
            org_vec = self.org[slot % len(rx.ORGS)]
            glob = self.plain

            def trust(issuer, org_vec=org_vec, glob=glob):
                global_w = weight_of(glob, issuer)
                local_w = weight_of(org_vec, issuer)
                if global_w is None:
                    return local_w
                if local_w is None:
                    return global_w
                return max(global_w, local_w)

            return trust
        if self.pool is not None:
            return self.pool.trust_fn(slot)
        if self.cfg.get("topic"):
            return self.plain[slot % len(rx.ORGS)]
        return self.plain

    def update(self, fresh, history, names):
        if self.cfg["rule"] not in ("trusted", "gate"):
            return
        eta = self.cfg.get("eta", 0.5)
        if self.cfg.get("envelope") or self.cfg.get("hier"):
            self.plain = fit_weights(fresh, self.plain, False, eta, names)
            blank = [rag.blank_weights() for _ in range(len(rx.ORGS))]
            self.org = fit_weights(history, blank, True, 1.0, names)
            self.org_n = []
            for org in range(len(rx.ORGS)):
                _rows, counts = claim_rows(history, names, org=org)
                self.org_n.append(counts)
            return
        docs = history if self.cfg.get("cumulative") else fresh
        if self.cfg.get("cumulative"):
            eta = 1.0
            if self.pool is not None:
                self.pool = PoolState()
                self.pool.update(docs, names, eta)
                return
            if self.cfg.get("topic"):
                blank = [rag.blank_weights() for _ in range(len(rx.ORGS))]
            else:
                blank = rag.blank_weights()
            self.plain = fit_weights(docs, blank, self.cfg.get("topic", False), eta, names)
            return
        if self.pool is not None:
            self.pool.update(docs, names, eta)
            return
        self.plain = fit_weights(docs, self.plain, self.cfg.get("topic", False), eta, names)


def run_cfg(world, cfg):
    names = world.names
    order = list(world.mutable_ids) + list(world.stable_ids) + list(world.derived_ids)
    corpus = cfg["corpus"]
    answers = []
    if corpus == "local":
        beliefs = [list(row) for row in world.init_belief]
        books = [TrustBook(cfg) for _ in range(rx.N_AGENTS)]
        mail = [[] for _ in range(rx.N_AGENTS)]
        for epoch_docs in world.visible:
            fresh = [[] for _ in range(rx.N_AGENTS)]
            for doc in epoch_docs:
                for agent in doc["receivers"]:
                    if 0 <= agent < rx.N_AGENTS:
                        mail[agent].append(doc)
                        fresh[agent].append(doc)
            grid = []
            for agent in range(rx.N_AGENTS):
                find = copy_find(mail[agent], names) if cfg.get("copy") else None
                usable = collapsed(mail[agent], find)
                buckets = index_hits(usable, names)
                row = list(beliefs[agent])
                for slot in order:
                    schema_v = schema_value(world, row, slot) if cfg.get("schema") else None
                    row[slot] = answer_slot(cfg, buckets.get(slot, []), row[slot], books[agent].for_slot(slot), schema_v)
                beliefs[agent] = row
                grid.append(row)
                books[agent].update(collapsed(fresh[agent], find), usable, names)
            answers.append(grid)
    else:
        belief = list(world.init_belief[0])
        book = TrustBook(cfg)
        mail = []
        for epoch_docs in world.visible:
            visible = epoch_docs
            if corpus == "shortcut":
                visible = [doc for doc in epoch_docs if len(doc["receivers"]) >= 2]
            fresh = list(visible)
            mail.extend(visible)
            find = copy_find(mail, names) if cfg.get("copy") else None
            usable = collapsed(mail, find)
            buckets = index_hits(usable, names)
            row = list(belief)
            for slot in order:
                schema_v = schema_value(world, row, slot) if cfg.get("schema") else None
                row[slot] = answer_slot(cfg, buckets.get(slot, []), row[slot], book.for_slot(slot), schema_v)
            belief = row
            answers.append([list(row) for _ in range(rx.N_AGENTS)])
            book.update(collapsed(fresh, find), usable, names)
    counts = rx.empty_counts()
    rx.consume(counts, world, answers, 0.0)
    return rx.pack_metrics(counts)


def cell_of(rows):
    cell = {}
    for key in ("sp", "changed", "stable", "derived", "obsolete", "delay", "cfa"):
        values = [row[key] for row in rows]
        mu, sd = rx.mean_std(values)
        cell[key] = mu
        cell[key + "_std"] = sd
        cell[key + "_seeds"] = values
    return cell


def balanced(cell):
    return (cell["changed"] + cell["stable"] + cell["derived"]) / 3.0


def evaluate(seeds, regime, rules, paraphrase=False, label=""):
    bucket = {i: [] for i in range(len(rules))}
    for seed in seeds:
        world = load_world(seed, regime, paraphrase=paraphrase)
        scores = []
        for i, cfg in enumerate(rules):
            metrics = run_cfg(world, cfg)
            bucket[i].append(metrics)
            scores.append(f"{i}:{metrics['sp']:.3f}/{metrics['changed']:.3f}")
        print(label or regime, seed, " ".join(scores), flush=True)
    return {i: cell_of(bucket[i]) for i in bucket}


def select(dev):
    ranked = []
    for i, cfg in enumerate(RULES):
        if not eligible(cfg):
            continue
        scores = {regime: balanced(dev[regime][i]) for regime in SELECT_REGIMES}
        ranked.append((sum(scores.values()) / len(scores), simplicity(cfg), i, cfg, scores))
    best = max(row[0] for row in ranked)
    near = [row for row in ranked if best - row[0] <= 0.005]
    near.sort(key=lambda row: (row[1], -row[0]))
    chosen = near[0]
    return {
        "index": chosen[2],
        "cfg": chosen[3],
        "dev_mean_balanced": chosen[0],
        "dev_balanced": chosen[4],
        "band": 0.005,
        "objective": "mean over stationary, topic, open_rumor, sybil of (changed+stable+derived)/3",
        "grid": [
            {
                "cfg": cfg,
                "eligible": eligible(cfg),
                "balanced": {regime: balanced(dev[regime][i]) for regime in SELECT_REGIMES},
                "sp": {regime: dev[regime][i]["sp"] for regime in SELECT_REGIMES},
                "mean_balanced": None
                if not eligible(cfg)
                else sum(balanced(dev[regime][i]) for regime in SELECT_REGIMES) / len(SELECT_REGIMES),
            }
            for i, cfg in enumerate(RULES)
        ],
    }


def perm_p(diffs):
    n = len(diffs)
    if n <= 16:
        return rx.signflip_p(diffs)
    obs = abs(sum(diffs))
    if obs <= 1e-12:
        return 1.0
    rng = random.Random(3157)
    count = 1
    for _ in range(99999):
        signed = 0.0
        for diff in diffs:
            signed += diff if rng.randrange(2) else -diff
        if abs(signed) + 1e-9 >= obs:
            count += 1
    return count / 100000.0


def paired(cells, left, right, metric):
    diffs = [a - b for a, b in zip(cells[left][metric + "_seeds"], cells[right][metric + "_seeds"])]
    mu, sd = rx.mean_std(diffs)
    return {"left": left, "right": right, "metric": metric, "mean_diff": mu, "std_diff": sd, "p": perm_p(diffs), "n": len(diffs)}


def slim(cell):
    keep = {"sp", "changed", "stable", "derived"}
    return {key: value for key, value in cell.items() if not key.endswith("_seeds") or key[: -len("_seeds")] in keep}


def address_audit(seeds):
    rows = []
    for seed in seeds:
        world = rx.generate_trajectory(seed, RATE, POBS, regime="stationary")
        truth_ge = rumor_ge = truth_n = rumor_n = 0
        for epoch_i, docs in enumerate(world.visible):
            for doc in docs:
                match = doc["value"] == world.timeline[epoch_i][doc["slot"]]
                wide = len(doc["receivers"]) >= 2
                if match:
                    truth_n += 1
                    truth_ge += int(wide)
                else:
                    rumor_n += 1
                    rumor_ge += int(wide)
        rows.append(
            {
                "truth_wide": truth_ge / float(truth_n),
                "rumor_wide": rumor_ge / float(rumor_n or 1),
                "truth_n": truth_n,
                "rumor_n": rumor_n,
            }
        )
    return {
        "truth_wide": sum(row["truth_wide"] for row in rows) / len(rows),
        "rumor_wide": sum(row["rumor_wide"] for row in rows) / len(rows),
        "n": len(rows),
    }


def tie_audit(seed):
    clean = load_world(seed, "stationary", paraphrase=False)
    messy = load_world(seed, "stationary", paraphrase=True)
    out = {}
    for label, world in (("clean", clean), ("paraphrase", messy)):
        docs = [doc for epoch in world.visible for doc in epoch]
        index = rag.Index(docs)
        spreads = []
        slot = world.mutable_ids[0]
        query = rx.query_text(world, slot)
        hits = index.search(query, k=500, margin=0.0)
        by_slot = [hit for hit in hits if entity_of_text(hit["text"], world.names) == slot]
        if len(by_slot) >= 2:
            scores = [hit["bm25"] for hit in by_slot]
            spreads.append(max(scores) - min(scores))
        out[label] = {"n": len(by_slot), "spread": spreads[0] if spreads else None}
    return out


def pair_stress(cfg, regimes, seeds, paraphrase=False):
    newest = {"rule": "newest", "corpus": "full", "cutoff": 0.0, "eta": 1.0, "topic": False, "schema": False, "copy": False}
    out = {}
    for regime in regimes:
        cells = evaluate(seeds, regime, [cfg, newest], paraphrase=paraphrase, label=f"stress-{regime}")
        out[regime] = {
            "selected_sp": cells[0]["sp"],
            "newest_sp": cells[1]["sp"],
            "selected_changed": cells[0]["changed"],
            "newest_changed": cells[1]["changed"],
            "selected_stable": cells[0]["stable"],
            "newest_stable": cells[1]["stable"],
            "selected_derived": cells[0]["derived"],
            "newest_derived": cells[1]["derived"],
            "paired_sp": paired(cells, 0, 1, "sp"),
        }
    return out


def sensitivity(cfg):
    neighbors = []
    for cutoff in (0.28, 0.32, 0.36, 0.40, 0.45):
        variant = dict(cfg, cutoff=cutoff, rule="trusted")
        cells = {}
        for regime in ("stationary", "open_rumor"):
            cells[regime] = evaluate(STRESS_SEEDS, regime, [variant], label=f"cut{cutoff}-{regime}")[0]
        neighbors.append(
            {
                "cutoff": cutoff,
                "sp_stationary": cells["stationary"]["sp"],
                "changed_stationary": cells["stationary"]["changed"],
                "stable_stationary": cells["stationary"]["stable"],
                "derived_stationary": cells["stationary"]["derived"],
                "sp_open": cells["open_rumor"]["sp"],
                "balanced_mean": 0.5 * (balanced(cells["stationary"]) + balanced(cells["open_rumor"])),
            }
        )
    return neighbors


def scale(cfg):
    saved = rx.N_AGENTS
    out = []
    try:
        for agents in (8, 16, 32):
            rx.N_AGENTS = agents
            row = {"agents": agents}
            block = pair_stress(cfg, ("stationary", "open_rumor"), SCALE_SEEDS)
            row.update(block)
            out.append(row)
            print("scale", row, flush=True)
    finally:
        rx.N_AGENTS = saved
    return out


def pobs_sweep(cfg):
    newest = {"rule": "newest", "corpus": "full", "cutoff": 0.0, "eta": 1.0, "topic": False, "schema": False, "copy": False}
    out = []
    for p_obs in (0.25, 0.45, 0.70):
        bucket = {0: defaultdict(list), 1: defaultdict(list)}
        for seed in SCALE_SEEDS:
            for regime in ("stationary", "open_rumor"):
                world = rx.generate_trajectory(seed, RATE, p_obs, regime=regime)
                stamp(world)
                bucket[0][regime].append(run_cfg(world, cfg))
                bucket[1][regime].append(run_cfg(world, newest))
        row = {"p_obs": p_obs}
        for label, items in (("selected", bucket[0]), ("newest", bucket[1])):
            for regime, rows in items.items():
                mu, sd = rx.mean_std([item["sp"] for item in rows])
                row[f"{label}_{regime}"] = mu
                row[f"{label}_{regime}_std"] = sd
        out.append(row)
        print("pobs", row, flush=True)
    return out


def dump(payload):
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("WROTE", OUT, flush=True)


def smoke():
    wanted = ("hier", "hier-m0", "hier-m2", "global", "org")
    cfgs = [cfg for cfg in RULES if cfg["name"] in wanted]
    for regime in ("stationary", "topic", "open_rumor", "sybil"):
        world = load_world(399, regime)
        print("REGIME", regime, flush=True)
        for cfg in cfgs:
            metrics = run_cfg(world, cfg)
            bal = (metrics["changed"] + metrics["stable"] + metrics["derived"]) / 3
            print(cfg["name"], {k: round(metrics[k], 3) for k in ("sp", "changed", "stable", "derived")}, "bal", round(bal, 3), flush=True)


def main():
    started = time.time()
    import run_causal

    run_causal.check_identity()
    dev = {}
    for regime in SELECT_REGIMES:
        print("DEV", regime, flush=True)
        dev[regime] = evaluate(DEV, regime, RULES)
    chosen = select(dev)
    print("SELECTED", chosen["index"], json.dumps(chosen["cfg"]), "mean", round(chosen["dev_mean_balanced"], 4), flush=True)
    payload = {"protocol": {"dev_seeds": DEV, "test_seeds": TEST, "selection": chosen, "rules": RULES}, "seconds": time.time() - started}
    dump(payload)
    test = {}
    for regime in ("stationary", "topic", "open_rumor", "sybil", "overlap"):
        print("TEST", regime, flush=True)
        test[regime] = evaluate(TEST, regime, RULES)
    payload["test"] = {regime: {str(i): slim(cell) for i, cell in cells.items()} for regime, cells in test.items()}
    payload["address_audit_dev"] = address_audit(DEV)
    payload["tie_audit_seed_700"] = tie_audit(700)
    label = chosen["index"]
    tests = []
    interesting = {i for i, cfg in enumerate(RULES) if cfg["name"] in ("newest", "majority", "crh", "gate", "global", "org", "hier-nocopy", "hier-noschema", "hier-m0", "hier-m2", "hier-50", "shortcut-newest") or i == label}
    for regime, cells in test.items():
        for other in interesting:
            if other == label:
                continue
            for metric in ("sp", "changed", "stable", "derived"):
                tests.append({"regime": regime, **paired(cells, label, other, metric)})
    payload["tests"] = tests
    payload["seconds"] = time.time() - started
    dump(payload)
    cfg = chosen["cfg"]
    payload["sensitivity_cutoff_not_used_for_selection"] = sensitivity(cfg)
    dump(payload)
    payload["stress"] = {
        "paraphrase": pair_stress(cfg, ("stationary",), TEST, paraphrase=True),
        "unicast_truth": pair_stress(cfg, ("unicast_truth",), STRESS_SEEDS),
        "drift_prop": pair_stress(cfg, ("drift_prop",), STRESS_SEEDS),
        "collude": pair_stress(cfg, ("collude",), STRESS_SEEDS),
    }
    dump(payload)
    payload["scale"] = scale(cfg)
    payload["p_obs"] = pobs_sweep(cfg)
    payload["seconds"] = time.time() - started
    dump(payload)


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        smoke()
    else:
        main()
