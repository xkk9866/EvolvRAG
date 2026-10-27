"""Record scope chosen by held-out predictive likelihood.

Each entity j gets a society record and an organization record that were fit
without j. Each record is scored by the marginal likelihood it assigns to the
round's testimony about j under a one-coin witness model. For entity k, the
organization record is used when the summed held-out log-likelihood ratio over
the organization's other entities is positive. No count of cleared issuers and
no margin enter the choice. Tau, eta, copy collapse, and the schema are locked.
Development is seeds 800-815. Test seeds 900-931 are read once afterwards.
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
import run_heldout as held

DEV = list(range(800, 816))
TEST = list(range(900, 932))
REGIMES = ("stationary", "topic", "open_rumor", "sybil")
CLIP = 1e-3
SCORE = "dec"


def log_marginal(claims, record):
    """log P(claims) with a uniform truth over claimed values plus one unseen value."""
    if len(claims) < 2:
        return 0.0
    values = sorted({value for value, _issuer in claims})
    cands = values + [None]
    wrong = len(cands) - 1
    terms = []
    for truth in cands:
        total = -math.log(len(cands))
        for value, issuer in claims:
            w = fair.weight_of(record, issuer)
            w = min(1.0 - CLIP, max(CLIP, w))
            total += math.log(w) if value == truth else math.log((1.0 - w) / wrong)
        terms.append(total)
    top = max(terms)
    return top + math.log(sum(math.exp(t - top) for t in terms))


def witness_score(hits, fresh, record, tau=0.55):
    """Leave-one-witness-out accuracy of the adoption rule under one record."""
    score = 0.0
    for issuer, value in fresh.items():
        best = None
        for hit in hits:
            if hit["issuer"] == issuer:
                continue
            w = fair.weight_of(record, hit["issuer"])
            if w is None or w + 1e-9 < tau:
                continue
            key = (hit["time"], hit.get("seq", 0))
            if best is None or key > best[0]:
                best = (key, hit["value"])
        if best is not None:
            score += 1.0 if best[1] == value else -1.0
    return score


def fit_pooled(claims, society_vec):
    """Organization record shrunk toward the issuer's held-out society weight."""
    out = list(society_vec)
    if not claims:
        return out
    center = {i: fair.weight_of(society_vec, i) for i in {c[2] for c in claims}}
    _t, _l, learned = rx.crh_fit(claims, center=center)
    for issuer, weight in learned.items():
        while issuer >= len(out):
            out.append(rx.PRIOR)
        out[issuer] = weight
    return out


def trust_from(vec):
    def trust(issuer, vec=vec):
        return fair.weight_of(vec, issuer)
    return trust


def run_world(world, modes=("loo", "xval")):
    cfg = next(item for item in fair.RULES if item["name"] == "hier")
    names = world.names
    order = list(world.mutable_ids) + list(world.stable_ids) + list(world.derived_ids)
    belief = {key: list(world.init_belief[0]) for key in modes}
    answers = {key: [] for key in modes}
    mail = []
    society = held.rag_blank()
    n_org = len(rx.ORGS)
    eta = cfg.get("eta", 0.5)
    picks = defaultdict(int)
    for epoch_docs in world.visible:
        mail.extend(epoch_docs)
        find = fair.copy_find(mail, names)
        usable = fair.collapsed(mail, find)
        fresh_c = fair.collapsed(epoch_docs, find)
        hist_rows, hist_max = held.claims_from(usable, names)
        fresh_rows, fresh_max = held.claims_from(fresh_c, names)
        by_org = [[] for _ in range(n_org)]
        org_slots = [set() for _ in range(n_org)]
        for slot, value, issuer in hist_rows:
            by_org[slot % n_org].append((slot, value, issuer))
            org_slots[slot % n_org].add(slot)
        fresh_by_slot = defaultdict(dict)
        for slot, value, issuer in fresh_rows:
            fresh_by_slot[slot][issuer] = value
        epoch_slots = set(fresh_by_slot)
        buckets = fair.index_hits(usable, names)
        pre = society
        blank = fair.pad(held.rag_blank(), hist_max + 1)
        soc_full = held.fit_vector(fresh_rows, pre, eta, max(len(pre) - 1, fresh_max))
        org_full = [held.fit_vector(by_org[g], blank, 1.0, hist_max) for g in range(n_org)]
        glob_of = {}
        orgv_of = {}
        gain = {}
        for slot in order:
            g = slot % n_org
            if slot in epoch_slots:
                kept = [item for item in fresh_rows if item[0] != slot]
                ids = [fresh_max] + [issuer for _s, _v, issuer in kept]
                glob_of[slot] = held.fit_vector(kept, pre, eta, max(len(pre) - 1, max(ids)))
            else:
                glob_of[slot] = soc_full
            if slot in org_slots[g]:
                kept = [item for item in by_org[g] if item[0] != slot]
                orgv_of[slot] = held.fit_vector(kept, blank, 1.0, hist_max)
            else:
                orgv_of[slot] = org_full[g]
            claims = [(value, issuer) for issuer, value in fresh_by_slot.get(slot, {}).items()]
            if SCORE == "lik":
                gain[slot] = log_marginal(claims, orgv_of[slot]) - log_marginal(claims, glob_of[slot])
            else:
                fresh = fresh_by_slot.get(slot, {})
                hits_j = buckets.get(slot, [])
                gain[slot] = witness_score(hits_j, fresh, orgv_of[slot]) - witness_score(hits_j, fresh, glob_of[slot])
        org_gain = [0.0] * n_org
        for slot, value in gain.items():
            org_gain[slot % n_org] += value
        row = {key: list(belief[key]) for key in modes}
        for slot in order:
            g = slot % n_org
            hits = buckets.get(slot, [])
            glob, org_vec = glob_of[slot], orgv_of[slot]
            for key in modes:
                schema_v = fair.schema_value(world, row[key], slot)
                if key == "loo":
                    trust = held.trust_for(glob, org_vec, cfg, False)
                elif key == "pool":
                    kept = [item for item in by_org[g] if item[0] != slot]
                    trust = trust_from(fit_pooled(kept, glob))
                else:
                    use_org = org_gain[g] - gain[slot] > 0.0
                    picks["org" if use_org else "soc"] += 1
                    trust = trust_from(org_vec if use_org else glob)
                row[key][slot] = fair.answer_slot(cfg, hits, row[key][slot], trust, schema_v)
        for key in modes:
            belief[key] = row[key]
            answers[key].append([list(row[key]) for _ in range(rx.N_AGENTS)])
        society = soc_full
    packed = {}
    for key, grid in answers.items():
        counts = rx.empty_counts()
        rx.consume(counts, world, grid, 0.0)
        packed[key] = rx.pack_metrics(counts)
    packed["_picks"] = dict(picks)
    return packed


def balanced(m):
    return (m["changed"] + m["stable"] + m["derived"]) / 3.0


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "probe"
    seeds = {"probe": [800, 801], "dev": DEV, "compare": list(range(700, 732)), "test": TEST}[task]
    modes = ("loo", "xval")
    stored = {m: {r: [] for r in REGIMES} for m in modes}
    picks = {r: defaultdict(int) for r in REGIMES}
    for regime in REGIMES:
        for seed in seeds:
            packed = run_world(fair.load_world(seed, regime), modes)
            for k, v in packed["_picks"].items():
                picks[regime][k] += v
            bits = []
            for m in modes:
                stored[m][regime].append(
                    {k: packed[m][k] for k in ("sp", "changed", "stable", "derived", "obsolete", "delay", "cfa")}
                )
                bits.append(f"{m}:{balanced(packed[m]):.3f}/{packed[m]['sp']:.3f}")
            print(regime, seed, " ".join(bits), flush=True)
    summary = {}
    for m in modes:
        by = {r: sum(balanced(x) for x in rows) / len(rows) for r, rows in stored[m].items()}
        summary[m] = {"by_regime": by, "mean": sum(by.values()) / len(by)}
    summary["org_share"] = {r: p["org"] / max(1, p["org"] + p["soc"]) for r, p in picks.items()}
    out = rx.ROOT / "results" / f"xval_{task}.json"
    out.write_text(json.dumps({"task": task, "summary": summary, "seeds": stored}), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
