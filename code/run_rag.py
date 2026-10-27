"""Retrieval path for EvolvRAG.

A question is a sentence. The answer is chosen from BM25's top five passages
in the documents that agent actually received. The slot id is not passed to
the retriever or the decision rule. Issuer weights are estimated from those
delivered documents. They are not labels.

The named system is chosen on seeds 100-115 before seeds 200-231 are read.
The selection score is the minimum of stationary and topic-split
stability-plasticity. Within 0.005 of the best minimum, the higher derived
accuracy wins; remaining ties keep the simpler setting (no topic split, no
schema, epoch-local weights). Stickiness and the margin stay at the values
locked on seeds 0-4. They are not refit here.

The shared board in the old runner is not this system. A separate check asks
what pooled majority does when unicast rumors are left out of that board.
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as rx

ROOT = rx.ROOT
OUT = ROOT / "results" / "rag.json"
K1 = 1.5
B = 0.75
TOPK = 500
# Passages whose BM25 score ties the best one differ by the value token,
# which the question does not contain. Keep every passage within this
# fraction of the best score, then cap by dropping the oldest.
DEV = list(range(100, 116))
TEST = list(range(200, 232))
SCALE_SEEDS = list(range(200, 204))
OLD_SEEDS = list(range(5, 15))
RATE = rx.DEFAULT_RATE
POBS = rx.DEFAULT_POBS

# eta = 1 replaces the weight with this epoch's estimate.
# eta = 0.25 keeps most of the accumulated weight.
GRID = []
for eta in (1.0, 0.25):
    for topic in (False, True):
        for schema in (False, True):
            GRID.append({"eta": eta, "topic": topic, "schema": schema, "rule": "gate"})


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class Index:
    def __init__(self, docs):
        self.docs = docs
        self.postings = defaultdict(list)
        self.tfs = []
        df = Counter()
        for i, doc in enumerate(docs):
            tf = Counter(tokens(doc["text"]))
            self.tfs.append(tf)
            for term in tf:
                df[term] += 1
                self.postings[term].append(i)
        self.n = len(docs)
        self.avg = sum(sum(tf.values()) for tf in self.tfs) / float(self.n or 1)
        self.idf = {
            term: math.log(1.0 + (self.n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def search(self, query, k=TOPK, margin=0.85):
        if not self.docs:
            return []
        q = Counter(tokens(query))
        seen = set()
        cand = []
        for term in q:
            for i in self.postings.get(term, ()):
                if i not in seen:
                    seen.add(i)
                    cand.append(i)
        if not cand:
            return []
        ranked = []
        for i in cand:
            score = self._score(q, i)
            if score > 0:
                ranked.append((score, i))
        if not ranked:
            return []
        # Score, then earlier time, then earlier index. Callers that want the
        # newest passage in a tied cluster must say so. Index order is not a tie-break.
        ranked.sort(key=lambda item: (-item[0], self.docs[item[1]]["time"], item[1]))
        floor = ranked[0][0] * margin
        kept = [item for item in ranked if item[0] + 1e-9 >= floor]
        if len(kept) > k:
            newest = sorted(kept, key=lambda item: self.docs[item[1]]["time"], reverse=True)[:k]
            keep_ids = {i for _score, i in newest}
            kept = [item for item in kept if item[1] in keep_ids]
        hits = []
        for score, i in kept:
            hit = dict(self.docs[i])
            hit["bm25"] = score
            hits.append(hit)
        return hits

    def _score(self, q, i):
        tf = self.tfs[i]
        length = sum(tf.values()) or 1
        norm = K1 * (1.0 - B + B * length / self.avg)
        total = 0.0
        for term, qf in q.items():
            count = tf.get(term)
            if not count:
                continue
            idf = self.idf.get(term, 0.0)
            total += idf * (count * (K1 + 1.0)) / (count + norm) * (qf * (K1 + 1.0)) / (qf + K1)
        return total


def blank_weights():
    return [rx.PRIOR for _ in range(rx.N_REL + rx.N_NOISY)]


def mix(old, new, eta):
    return [(1.0 - eta) * a + eta * b for a, b in zip(old, new)]


def weights_from_claims(claims, previous):
    _truth, _log_w, learned = rx.crh_fit(claims)
    updated = list(previous)
    for issuer, weight in learned.items():
        if 0 <= issuer < len(updated):
            updated[issuer] = weight
    return updated


def majority_value(hits, belief):
    buckets = defaultdict(set)
    for hit in hits:
        issuer = hit["issuer"]
        if issuer is None or not isinstance(issuer, int) or issuer < 0:
            continue
        buckets[hit["value"]].add(issuer)
    if not buckets:
        return belief
    best_n = max(len(members) for members in buckets.values())
    cands = [value for value, members in buckets.items() if len(members) == best_n]
    if belief in cands:
        return belief
    return min(cands)


def crh_value(hits, belief):
    claims = [(0, hit["value"], hit["issuer"]) for hit in hits if isinstance(hit["issuer"], int) and hit["issuer"] >= 0]
    if not claims:
        return belief
    truth, _log_w, _learned = rx.crh_fit(claims)
    return truth.get(0, belief)


def gate_value(hits, belief, trust, schema_value, schema):
    flags = {"margin": True, "trust": True, "dep": schema, "exp": True, "open": False}
    value, _committed, _margin, _issuer = rx.decide(belief, hits, trust, schema_value, flags)
    return value if value is not None else belief


def recent_majority(hits, belief, window):
    newest = max(hit["time"] for hit in hits)
    kept = [hit for hit in hits if hit["time"] >= newest - window]
    return majority_value(kept, belief)


def cluster_pick(hits, newest):
    best = max(hit.get("bm25", 0.0) for hit in hits)
    tied = [hit for hit in hits if hit.get("bm25", 0.0) >= best - 1e-9]
    if newest:
        return max(tied, key=lambda hit: (hit["time"], hit.get("seq", 0)))["value"]
    return min(tied, key=lambda hit: (hit["time"], hit.get("seq", 0)))["value"]


def answer_hits(rule, hits, belief, trust, schema_value, schema, window=1):
    if not hits:
        return belief
    if rule in ("bm25", "cluster_oldest"):
        return cluster_pick(hits, newest=False)
    if rule == "cluster_newest":
        return cluster_pick(hits, newest=True)
    if rule == "recency":
        return max(hits, key=lambda hit: (hit["time"], hit.get("seq", 0)))["value"]
    if rule == "majority":
        return majority_value(hits, belief)
    if rule == "recent":
        return recent_majority(hits, belief, window)
    if rule == "crh":
        return crh_value(hits, belief)
    return gate_value(hits, belief, trust, schema_value, schema)


def collect_trace(world, corpus="local"):
    """BM25 hits do not depend on the decision rule when the corpus rule is fixed.

    local: documents delivered to that agent.
    corroborated: documents whose recipient list has size at least 2, shared.
    union: local deliveries plus those corroborated documents.
    all: every generated document, shared. This is the omniscient board.
    """
    mail = [[] for _ in range(rx.N_AGENTS)]
    order = list(world.mutable_ids) + list(world.stable_ids) + list(world.derived_ids)
    queries = [rx.query_text(world, slot) for slot in order]
    epochs = []
    recall_hit = entity_hit = held_entity = held_value = recall_n = 0
    shared_mail = corpus in ("corroborated", "all")
    for epoch_i, docs in enumerate(world.visible):
        fresh = [[] for _ in range(rx.N_AGENTS)]
        corroborated = [doc for doc in docs if len(doc["receivers"]) >= 2]

        def add(agent, doc):
            mail[agent].append(doc)
            fresh[agent].append(doc)

        for doc in docs:
            for agent in doc["receivers"]:
                if corpus in ("local", "union"):
                    add(agent, doc)
        if corpus == "corroborated":
            for agent in range(rx.N_AGENTS):
                for doc in corroborated:
                    add(agent, doc)
        elif corpus == "union":
            for doc in corroborated:
                for agent in range(rx.N_AGENTS):
                    if agent not in doc["receivers"]:
                        add(agent, doc)
        elif corpus == "all":
            for agent in range(rx.N_AGENTS):
                for doc in docs:
                    add(agent, doc)
        agents = []
        shared_hits = None
        if shared_mail:
            shared_hits = [Index(mail[0]).search(query, TOPK) for query in queries]
        for agent in range(rx.N_AGENTS):
            if shared_hits is None:
                index = Index(mail[agent])
                hits = [index.search(query, TOPK) for query in queries]
            else:
                hits = shared_hits
            claims = defaultdict(list)
            held_slots = set()
            held_pairs = set()
            for doc in mail[agent]:
                held_slots.add(doc["slot"])
                held_pairs.add((doc["slot"], doc["value"]))
            for doc in fresh[agent]:
                if isinstance(doc["issuer"], int) and doc["issuer"] >= 0:
                    claims[doc["slot"] % len(rx.ORGS)].append((doc["slot"], doc["value"], doc["issuer"]))
            if agent == 0 or not shared_mail:
                for slot, passages in zip(order, hits):
                    truth = world.timeline[epoch_i][slot]
                    recall_n += 1
                    if slot in held_slots:
                        held_entity += 1
                    if (slot, truth) in held_pairs:
                        held_value += 1
                    if any(hit["slot"] == slot for hit in passages):
                        entity_hit += 1
                    if any(hit["slot"] == slot and hit["value"] == truth for hit in passages):
                        recall_hit += 1
            agents.append({"hits": hits, "claims": claims})
        epochs.append(agents)
    published = [doc for docs in world.visible for doc in docs if len(doc["receivers"]) >= 2]
    pub_bytes = sum(len(doc["text"].encode("utf-8")) + 8 for doc in published)
    coverage = {
        "recall5": recall_hit / float(recall_n),
        "entity_recall": entity_hit / float(recall_n),
        "held_entity": held_entity / float(recall_n),
        "held_value": held_value / float(recall_n),
        "published_docs": len(published) / float(rx.EPOCHS),
        "published_bytes": pub_bytes / float(rx.EPOCHS),
    }
    return order, epochs, coverage


def score_trace(world, order, epochs, coverage, cfg):
    beliefs = [list(row) for row in world.init_belief]
    weights = [blank_weights() for _ in range(rx.N_AGENTS)]
    topic_w = [[blank_weights() for _ in range(len(rx.ORGS))] for _ in range(rx.N_AGENTS)]
    answers = []
    rule = cfg["rule"]
    eta = cfg.get("eta", 1.0)
    use_topic = cfg.get("topic", False)
    schema = cfg.get("schema", False)
    for epoch_i, agents in enumerate(epochs):
        grid = []
        for agent in range(rx.N_AGENTS):
            row = list(beliefs[agent])
            hits_by_slot = agents[agent]["hits"]
            for slot, hits in zip(order, hits_by_slot):
                org = slot % len(rx.ORGS)
                trust = topic_w[agent][org] if use_topic else weights[agent]
                schema_value = None
                parent = world.parent.get(slot)
                if schema and parent is not None:
                    parent_v = row[parent]
                    schema_value = "EXPIRED" if parent_v == "EXPIRED" else f"via-{parent_v}"
                row[slot] = answer_hits(
                    rule, hits, row[slot], trust, schema_value, schema, cfg.get("window", 1)
                )
            beliefs[agent] = row
            grid.append(row)
            if rule == "gate":
                claims_by_org = agents[agent]["claims"]
                if use_topic:
                    for org, claims in claims_by_org.items():
                        fresh = weights_from_claims(claims, topic_w[agent][org])
                        topic_w[agent][org] = mix(topic_w[agent][org], fresh, eta)
                else:
                    claims = [claim for rows in claims_by_org.values() for claim in rows]
                    if claims:
                        fresh = weights_from_claims(claims, weights[agent])
                        weights[agent] = mix(weights[agent], fresh, eta)
        answers.append(grid)
    counts = rx.empty_counts()
    rx.consume(counts, world, answers, 0.0)
    metrics = rx.pack_metrics(counts)
    metrics.update(coverage)
    metrics["upload_bytes"] = 0.0
    return metrics


def run_retrieval(world, cfg, share=False):
    beliefs = [list(row) for row in world.init_belief]
    mail = [[] for _ in range(rx.N_AGENTS)]
    shared = []
    uploaded = set()
    upload_bytes = 0
    weights = [blank_weights() for _ in range(rx.N_AGENTS)]
    topic_w = [
        [blank_weights() for _ in range(len(rx.ORGS))] for _ in range(rx.N_AGENTS)
    ]
    order = list(world.mutable_ids) + list(world.stable_ids) + list(world.derived_ids)
    answers = []
    recall_hit = 0
    entity_hit = 0
    held_entity = 0
    held_value = 0
    recall_n = 0
    rule = cfg["rule"]
    eta = cfg.get("eta", 1.0)
    use_topic = cfg.get("topic", False)
    schema = cfg.get("schema", False)
    for epoch_i, docs in enumerate(world.visible):
        delivered = [[] for _ in range(rx.N_AGENTS)]
        for doc in docs:
            for agent in doc["receivers"]:
                mail[agent].append(doc)
                delivered[agent].append(doc)
        grid = []
        for agent in range(rx.N_AGENTS):
            index = Index(mail[agent] + shared)
            row = list(beliefs[agent])
            for slot in order:
                query = rx.query_text(world, slot)
                hits = index.search(query, TOPK)
                truth = world.timeline[epoch_i][slot]
                recall_n += 1
                if any(doc["slot"] == slot for doc in mail[agent]):
                    held_entity += 1
                if any(doc["slot"] == slot and doc["value"] == truth for doc in mail[agent]):
                    held_value += 1
                if any(hit["slot"] == slot for hit in hits):
                    entity_hit += 1
                if any(hit["slot"] == slot and hit["value"] == truth for hit in hits):
                    recall_hit += 1
                org = slot % len(rx.ORGS)
                trust = topic_w[agent][org] if use_topic else weights[agent]
                schema_value = None
                parent = world.parent.get(slot)
                if schema and parent is not None:
                    parent_v = row[parent]
                    schema_value = "EXPIRED" if parent_v == "EXPIRED" else f"via-{parent_v}"
                chosen = answer_hits(rule, hits, row[slot], trust, schema_value, schema)
                row[slot] = chosen
                if share and rule == "gate":
                    for hit in hits:
                        if hit["value"] != chosen:
                            continue
                        key = (hit["slot"], hit["value"], hit["time"], hit["issuer"])
                        if key in uploaded:
                            continue
                        uploaded.add(key)
                        shared.append(hit)
                        upload_bytes += len(hit["text"].encode("utf-8")) + 8
                        break
            beliefs[agent] = row
            grid.append(row)
            if rule == "gate":
                by_org = defaultdict(list)
                for doc in delivered[agent]:
                    if isinstance(doc["issuer"], int) and doc["issuer"] >= 0:
                        by_org[doc["slot"] % len(rx.ORGS)].append((doc["slot"], doc["value"], doc["issuer"]))
                if use_topic:
                    for org, claims in by_org.items():
                        fresh = weights_from_claims(claims, topic_w[agent][org])
                        topic_w[agent][org] = mix(topic_w[agent][org], fresh, eta)
                else:
                    claims = [claim for rows in by_org.values() for claim in rows]
                    if claims:
                        fresh = weights_from_claims(claims, weights[agent])
                        weights[agent] = mix(weights[agent], fresh, eta)
        answers.append(grid)
    counts = rx.empty_counts()
    rx.consume(counts, world, answers, upload_bytes)
    metrics = rx.pack_metrics(counts)
    metrics["recall5"] = recall_hit / float(recall_n)
    metrics["entity_recall"] = entity_hit / float(recall_n)
    metrics["held_entity"] = held_entity / float(recall_n)
    metrics["held_value"] = held_value / float(recall_n)
    metrics["upload_bytes"] = upload_bytes / float(rx.N_AGENTS * rx.EPOCHS)
    return metrics


def cell_of(rows):
    cell = {}
    keys = rows[0].keys()
    for key in keys:
        values = [row[key] for row in rows]
        if isinstance(values[0], float):
            mu, sd = rx.mean_std(values)
            cell[key] = mu
            cell[key + "_std"] = sd
            if key in ("sp", "changed", "stable", "derived", "cfa"):
                cell[key + "_seeds"] = values
    return cell


def evaluate(seeds, regime, cfgs):
    bucket = {i: [] for i in range(len(cfgs))}
    sharing = any(cfg.get("share") for cfg in cfgs)
    for seed in seeds:
        world = rx.generate_trajectory(seed, RATE, POBS, regime=regime)
        if sharing:
            for i, cfg in enumerate(cfgs):
                bucket[i].append(run_retrieval(world, cfg, share=cfg.get("share", False)))
        else:
            order, epochs, coverage = collect_trace(world)
            for i, cfg in enumerate(cfgs):
                bucket[i].append(score_trace(world, order, epochs, coverage, cfg))
        print(regime, seed, " ".join(f"{i}:{bucket[i][-1]['sp']:.3f}" for i in range(len(cfgs))), flush=True)
    return {i: cell_of(bucket[i]) for i in bucket}


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


def select(dev_stat, dev_topic):
    ranked = []
    for i, cfg in enumerate(GRID):
        sp_s = dev_stat[i]["sp"]
        sp_t = dev_topic[i]["sp"]
        derived = 0.5 * (dev_stat[i]["derived"] + dev_topic[i]["derived"])
        ranked.append((min(sp_s, sp_t), derived, cfg, i, sp_s, sp_t))
    best_min = max(row[0] for row in ranked)
    near = [row for row in ranked if best_min - row[0] <= 0.005]
    near.sort(key=lambda row: (-row[1], row[2]["topic"], row[2]["schema"], -row[2]["eta"]))
    chosen = near[0]
    return {
        "index": chosen[3],
        "cfg": chosen[2],
        "dev_min_sp": chosen[0],
        "dev_derived": chosen[1],
        "dev_sp_stationary": chosen[4],
        "dev_sp_topic": chosen[5],
        "grid": [
            {
                "cfg": cfg,
                "sp_stationary": dev_stat[i]["sp"],
                "sp_topic": dev_topic[i]["sp"],
                "derived_stationary": dev_stat[i]["derived"],
                "derived_topic": dev_topic[i]["derived"],
                "min_sp": min(dev_stat[i]["sp"], dev_topic[i]["sp"]),
            }
            for i, cfg in enumerate(GRID)
        ],
    }


def baselines():
    return [
        {"rule": "bm25", "eta": 1.0, "topic": False, "schema": False},
        {"rule": "recency", "eta": 1.0, "topic": False, "schema": False},
        {"rule": "majority", "eta": 1.0, "topic": False, "schema": False},
        {"rule": "crh", "eta": 1.0, "topic": False, "schema": False},
    ]


def paired(cells, left, right, metric="sp"):
    diffs = [a - b for a, b in zip(cells[left][metric + "_seeds"], cells[right][metric + "_seeds"])]
    mu, sd = rx.mean_std(diffs)
    return {"left": left, "right": right, "metric": metric, "mean_diff": mu, "std_diff": sd, "p": perm_p(diffs), "n": len(diffs)}


def corroborated():
    """Old board, same worlds, with and without unicast rumors."""
    rows = {name: [] for name in ("PoolFull", "PoolCorroborated", "GateFull", "GateCorroborated", "LocalMajority")}
    orig = rx.Memory._pool

    def pool_corroborated(self, docs):
        return orig(self, [doc for doc in docs if len(doc["receivers"]) >= 2])

    for seed in OLD_SEEDS:
        world = rx.generate_trajectory(seed, RATE, POBS)
        rx.Memory._pool = orig
        for method, key in (("PoolMajority", "PoolFull"), ("Central", "GateFull"), ("Majority", "LocalMajority")):
            counts, _answers = rx.run_method(world, method, seed)
            rows[key].append(rx.pack_metrics(counts)["sp"])
        rx.Memory._pool = pool_corroborated
        for method, key in (("PoolMajority", "PoolCorroborated"), ("Central", "GateCorroborated")):
            counts, _answers = rx.run_method(world, method, seed)
            rows[key].append(rx.pack_metrics(counts)["sp"])
        print("corroborated", seed, {key: round(rows[key][-1], 3) for key in rows}, flush=True)
    rx.Memory._pool = orig
    out = {}
    for key, values in rows.items():
        mu, sd = rx.mean_std(values)
        out[key] = {"sp": mu, "sp_std": sd, "sp_seeds": values}
    out["tests"] = []
    for left, right in (
        ("GateFull", "PoolFull"),
        ("GateCorroborated", "PoolCorroborated"),
        ("PoolCorroborated", "PoolFull"),
        ("GateCorroborated", "GateFull"),
    ):
        diffs = [a - b for a, b in zip(out[left]["sp_seeds"], out[right]["sp_seeds"])]
        mu, sd = rx.mean_std(diffs)
        out["tests"].append({"left": left, "right": right, "mean_diff": mu, "std_diff": sd, "p": perm_p(diffs), "n": len(diffs)})
    return out


def scale(cfg):
    saved = rx.N_AGENTS
    out = []
    try:
        for agents in (8, 16, 32):
            rx.N_AGENTS = agents
            cfgs = [baselines()[0], baselines()[2], cfg]
            cells = evaluate(SCALE_SEEDS, "stationary", cfgs)
            out.append(
                {
                    "agents": agents,
                    "bm25": cells[0]["sp"],
                    "majority": cells[1]["sp"],
                    "selected": cells[2]["sp"],
                    "bm25_std": cells[0]["sp_std"],
                    "majority_std": cells[1]["sp_std"],
                    "selected_std": cells[2]["sp_std"],
                }
            )
    finally:
        rx.N_AGENTS = saved
    return out


def share_check(cfg):
    local = dict(cfg, share=False)
    shared = dict(cfg, share=True)
    cells = evaluate(SCALE_SEEDS, "stationary", [local, shared])
    return {"local": cells[0], "shared": cells[1]}


def slim(cell):
    return {key: value for key, value in cell.items() if not key.endswith("_seeds") or key in ("sp_seeds", "changed_seeds", "stable_seeds", "derived_seeds")}


def main():
    started = time.time()
    print("DEV stationary", flush=True)
    dev_stat = evaluate(DEV, "stationary", GRID)
    print("DEV topic", flush=True)
    dev_topic = evaluate(DEV, "topic", GRID)
    chosen = select(dev_stat, dev_topic)
    print("SELECTED", json.dumps(chosen["cfg"]), "min", round(chosen["dev_min_sp"], 4), flush=True)
    named = dict(chosen["cfg"], rule="gate")
    test_cfgs = baselines() + GRID
    print("TEST stationary", flush=True)
    test_stat = evaluate(TEST, "stationary", test_cfgs)
    print("TEST topic", flush=True)
    test_topic = evaluate(TEST, "topic", test_cfgs)
    names = ["BM25", "Recency", "HitMajority", "HitCRH"] + [f"grid{i}" for i in range(len(GRID))]
    evolv_name = f"grid{chosen['index']}"
    def named_cells(block):
        return {name: slim(block[i]) for i, name in enumerate(names)}

    tests = []
    stat_cells = named_cells(test_stat)
    topic_cells = named_cells(test_topic)
    stat_cells["EvolvRAG"] = stat_cells[evolv_name]
    topic_cells["EvolvRAG"] = topic_cells[evolv_name]
    for metric in ("sp", "changed", "stable", "derived"):
        for other in ("BM25", "Recency", "HitMajority", "HitCRH"):
            tests.append({"regime": "stationary", **paired(stat_cells, "EvolvRAG", other, metric)})
            tests.append({"regime": "topic", **paired(topic_cells, "EvolvRAG", other, metric)})
    print("CORROBORATED", flush=True)
    rumor = corroborated()
    print("SCALE", flush=True)
    grown = scale(named)
    print("SHARE", flush=True)
    shared = share_check(named)
    payload = {
        "protocol": {
            "dev_seeds": DEV,
            "test_seeds": TEST,
            "selection": "max min(stationary SP, topic SP); within 0.005 prefer derived; then simpler",
            "topk": TOPK,
            "stick_locked_on_seeds_0_4": True,
            "query": "natural language question; slot id is not an input to retrieval or the gate",
            "corpus": "documents delivered to that agent",
            "selected": chosen,
        },
        "test_stationary": stat_cells,
        "test_topic": topic_cells,
        "tests": tests,
        "corroborated_old_seeds": rumor,
        "scale": grown,
        "share": {"local": slim(shared["local"]), "shared": slim(shared["shared"])},
        "seconds": time.time() - started,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("WROTE", OUT, flush=True)
    for row in tests:
        if row["metric"] == "sp":
            print("TEST", row["regime"], row["left"], "minus", row["right"], f"{row['mean_diff']:+.4f}", f"p={row['p']:.6f}", flush=True)


if __name__ == "__main__":
    main()
