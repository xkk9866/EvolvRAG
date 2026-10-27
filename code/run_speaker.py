"""Choose the record whose eligible speaker is more trusted.

The count rule adopts the local record whenever it clears more issuers.
A local record can clear two issuers at 0.60 while the society record
clears one issuer at 0.88. Speaker routing publishes the record whose
eligible speaker has the higher weight. Ties stay with the society record.
The floor stays 0.55. This file does not rewrite hier.json.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_decentral as dec
import run_experiment as rx
import run_fair as fair
import run_fusion as fusion

ROOT = rx.ROOT
OUT = ROOT / "results" / "speaker.json"
TAU = 0.55
DEV = list(range(600, 616))
REGIMES = fair.SELECT_REGIMES
HIER = next(cfg for cfg in fair.RULES if cfg["name"] == "hier")


def perm_p(diffs):
    obs = abs(sum(diffs))
    if obs <= 1e-12:
        return 1.0
    wins = sum(diff > 0 for diff in diffs)
    losses = sum(diff < 0 for diff in diffs)
    rng = random.Random(3157)
    count = 1
    for _ in range(99999):
        plus_wins = rng.getrandbits(wins).bit_count() if wins else 0
        plus_losses = rng.getrandbits(losses).bit_count() if losses else 0
        signed = (2 * plus_wins - wins) - (2 * plus_losses - losses)
        if abs(signed) + 1e-9 >= obs:
            count += 1
    return max(count / 100000.0, 1e-5)


def mean_std(values):
    n = len(values)
    mu = sum(values) / n
    var = sum((value - mu) ** 2 for value in values) / (n - 1)
    return mu, var ** 0.5


def nominate(weights, pairs):
    best = None
    best_key = None
    for value, issuer in pairs:
        weight = weights.get(issuer, fusion.PRIOR)
        if weight + 1e-9 < TAU:
            continue
        key = (weight, issuer)
        if best_key is None or key > best_key:
            best = value
            best_key = key
    return best, best_key


def fusion_corpus(name, days):
    book = fusion.Books()
    hits = {key: [] for key in ("society", "route", "speaker")}
    switched = 0
    for index, day in enumerate(days):
        if index == 0:
            book.update(day["claims"])
            continue
        by_slot = defaultdict(list)
        org_of = {}
        for slot, value, issuer, org in day["claims"]:
            by_slot[slot].append((value, issuer))
            org_of[slot] = org
        for slot, answer in day["gold"].items():
            pairs = by_slot.get(slot, [])
            org = org_of.get(slot)
            if org is None:
                continue
            society = book.weights(org, "society")
            local = book.weights(org, "org")
            route_w = book.weights(org, "hier")
            soc_value, soc_key = nominate(society, pairs)
            org_value, org_key = nominate(local, pairs)
            route_value, _route_key = nominate(route_w, pairs)
            if soc_value is None:
                soc_value = book.belief["society"].get(slot)
            if org_value is None:
                org_value = book.belief["org"].get(slot)
            if route_value is None:
                route_value = book.belief["hier"].get(slot)
            if org_key is not None and (soc_key is None or org_key[0] > soc_key[0]):
                speaker_value = org_value
                switched += 1
            else:
                speaker_value = soc_value
            hits["society"].append(int(soc_value == answer))
            hits["route"].append(int(route_value == answer))
            hits["speaker"].append(int(speaker_value == answer))
            if soc_value is not None:
                book.belief["society"][slot] = soc_value
            if org_value is not None:
                book.belief["org"][slot] = org_value
            if route_value is not None:
                book.belief["hier"][slot] = route_value
            if speaker_value is not None:
                book.belief.setdefault("speaker", {})[slot] = speaker_value
        book.update(day["claims"])
        print(name, day["day"], "speaker", round(sum(hits["speaker"]) / len(hits["speaker"]), 3), flush=True)
    summary = {key: sum(values) / len(values) for key, values in hits.items()}
    tests = {}
    for left, right in (("speaker", "society"), ("speaker", "route")):
        diffs = [a - b for a, b in zip(hits[left], hits[right])]
        mu, sd = mean_std(diffs)
        tests[f"{left}_minus_{right}"] = {"mean": mu, "std": sd, "p": perm_p(diffs), "n": len(diffs)}
    return {"em": summary, "switched": switched, "n": len(hits["society"]), "tests": tests}


def latest_eligible(hits, trust):
    best = None
    for hit in hits:
        weight = fair.weight_of(trust, hit["issuer"])
        if weight is None or weight + 1e-9 < TAU:
            continue
        key = (hit["time"], hit.get("seq", 0))
        if best is None or key > best[0]:
            best = (key, weight, hit["value"])
    return best


def speaker_value(book, hits, belief, schema_v):
    if book.org is None:
        return fair.answer_slot(HIER, hits, belief, book.for_slot(0), schema_v)
    return belief


def run_speaker_world(world):
    names = world.names
    order = list(world.mutable_ids) + list(world.stable_ids) + list(world.derived_ids)
    belief = list(world.init_belief[0])
    book = fair.TrustBook(HIER)
    mail = []
    answers = []
    for epoch_docs in world.visible:
        mail.extend(epoch_docs)
        find = fair.copy_find(mail, names)
        usable = fair.collapsed(mail, find)
        buckets = fair.index_hits(usable, names)
        row = list(belief)
        fresh = fair.collapsed(epoch_docs, find)
        # Weights used for the decision are those fit through the previous epoch.
        for slot in order:
            schema_v = fair.schema_value(world, row, slot)
            hits = buckets.get(slot, [])
            if book.org is None:
                row[slot] = fair.answer_slot(HIER, hits, row[slot], book.for_slot(slot), schema_v)
                continue
            org_i = slot % len(rx.ORGS)
            soc = latest_eligible(hits, book.plain)
            loc = latest_eligible(hits, book.org[org_i])
            if loc is not None and (soc is None or loc[1] > soc[1]):
                row[slot] = loc[2]
            elif soc is not None:
                row[slot] = soc[2]
            elif schema_v is not None:
                row[slot] = schema_v
            else:
                row[slot] = row[slot]
        belief = row
        answers.append([list(row) for _ in range(rx.N_AGENTS)])
        book.update(fresh, usable, names)
    counts = rx.empty_counts()
    rx.consume(counts, world, answers, 0.0)
    return rx.pack_metrics(counts)


def synthetic():
    stored = {name: {regime: [] for regime in REGIMES} for name in ("hier", "speaker")}
    for regime in REGIMES:
        for seed in DEV:
            world = fair.load_world(seed, regime)
            stored["hier"][regime].append(dec.metrics_of(world, HIER))
            stored["speaker"][regime].append(run_speaker_world(world))
            print(
                regime,
                seed,
                "hier",
                round(dec.balanced(stored["hier"][regime][-1]), 3),
                "speaker",
                round(dec.balanced(stored["speaker"][regime][-1]), 3),
                flush=True,
            )
    summary = {}
    tests = []
    for name in stored:
        flat = []
        per = {}
        for regime in REGIMES:
            values = [dec.balanced(row) for row in stored[name][regime]]
            mu, sd = mean_std(values)
            per[regime] = {"balanced": mu, "std": sd, "sp": mean_std([row["sp"] for row in stored[name][regime]])[0]}
            flat.extend(values)
        mu, sd = mean_std(flat)
        summary[name] = {"balanced": mu, "std": sd, "per_regime": per}
    for regime in REGIMES:
        diffs = [
            dec.balanced(a) - dec.balanced(b)
            for a, b in zip(stored["speaker"][regime], stored["hier"][regime])
        ]
        mu, sd = mean_std(diffs)
        tests.append({"regime": regime, "mean": mu, "std": sd, "p": fair.perm_p(diffs), "n": len(diffs)})
    flat_s = []
    flat_h = []
    for regime in REGIMES:
        flat_s.extend(dec.balanced(row) for row in stored["speaker"][regime])
        flat_h.extend(dec.balanced(row) for row in stored["hier"][regime])
    diffs = [a - b for a, b in zip(flat_s, flat_h)]
    mu, sd = mean_std(diffs)
    tests.append({"regime": "four", "mean": mu, "std": sd, "p": fair.perm_p(diffs), "n": len(diffs)})
    return {"summary": summary, "tests": tests}


def main():
    print("stock", flush=True)
    stock = fusion_corpus("stock", fusion.stock_days())
    print("flight", flush=True)
    flight = fusion_corpus("flight", fusion.flight_days())
    print("synthetic", flush=True)
    synth = synthetic()
    payload = {"stock": stock, "flight": flight, "dev": synth, "tau": TAU}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
