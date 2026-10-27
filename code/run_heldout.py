"""Revision with the judged entity held out of the reliability fit.

Source records answer whether a witness agrees with other witnesses. The
entity being revised is not one of those witnesses. Testimony about every
other entity, including testimony that arrived in the current round, is in
the fit. The same clearance rule then adopts the latest eligible passage.
Development seeds 600-615 are the selection set. Test seeds stay unread
until that comparison is finished.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as rx
import run_fair as fair

OUT = rx.ROOT / "results" / "heldout_dev.json"
DEV = list(range(600, 616))
REGIMES = ("stationary", "topic", "open_rumor", "sybil")


def claims_from(docs, names):
    rows = []
    max_id = rx.N_REL + rx.N_NOISY - 1
    for doc in docs:
        issuer = doc["issuer"]
        if isinstance(issuer, int) and issuer > max_id:
            max_id = issuer
        slot = fair.entity_of_text(doc["text"], names)
        if slot is None or not isinstance(issuer, int) or issuer < 0:
            continue
        rows.append((slot, doc["value"], issuer))
    return rows, max_id


def fit_vector(claims, previous, eta, max_id):
    import run_rag as rag

    base = fair.pad(previous, max_id + 1)
    if not claims:
        return base
    fresh = rag.weights_from_claims(claims, base)
    return rag.mix(base, fresh, eta)


def trust_for(glob, org_vec, cfg, society_only):
    tau = float(cfg.get("cutoff", 0.55))
    margin = int(cfg.get("margin", 1))
    if society_only:
        def trust(issuer, glob=glob):
            return fair.weight_of(glob, issuer)
        return trust
    width = max(len(glob), len(org_vec))
    soc_high = 0
    org_high = 0
    for issuer in range(width):
        if fair.weight_of(glob, issuer) + 1e-9 >= tau:
            soc_high += 1
        if fair.weight_of(org_vec, issuer) + 1e-9 >= tau:
            org_high += 1
    use_org = org_high >= soc_high + margin

    def trust(issuer, glob=glob, org_vec=org_vec, use_org=use_org):
        local_w = fair.weight_of(org_vec, issuer)
        global_w = fair.weight_of(glob, issuer)
        if use_org and local_w is not None:
            return local_w
        return global_w

    return trust


def run_loo(world):
    """One pass returns the routed held-out decision and the society-only held-out decision."""
    cfg = next(item for item in fair.RULES if item["name"] == "hier")
    names = world.names
    order = list(world.mutable_ids) + list(world.stable_ids) + list(world.derived_ids)
    belief = {"loo": list(world.init_belief[0]), "loo_global": list(world.init_belief[0])}
    mail = []
    society = rag_blank()
    answers = {key: [] for key in belief}
    n_org = len(rx.ORGS)
    eta = cfg.get("eta", 0.5)
    for epoch_docs in world.visible:
        mail.extend(epoch_docs)
        find = fair.copy_find(mail, names)
        usable = fair.collapsed(mail, find)
        fresh_c = fair.collapsed(epoch_docs, find)
        hist_rows, hist_max = claims_from(usable, names)
        fresh_rows, fresh_max = claims_from(fresh_c, names)
        by_org = [[] for _ in range(n_org)]
        org_slots = [set() for _ in range(n_org)]
        for slot, value, issuer in hist_rows:
            org = slot % n_org
            by_org[org].append((slot, value, issuer))
            org_slots[org].add(slot)
        epoch_slots = {slot for slot, _value, _issuer in fresh_rows}
        buckets = fair.index_hits(usable, names)
        pre = society
        blank = fair.pad(rag_blank(), hist_max + 1)
        soc_full = fit_vector(fresh_rows, pre, eta, max(len(pre) - 1, fresh_max))
        org_full = [fit_vector(by_org[org], blank, 1.0, hist_max) for org in range(n_org)]
        row = {key: list(belief[key]) for key in belief}
        for slot in order:
            org = slot % n_org
            if slot in epoch_slots:
                kept_epoch = [item for item in fresh_rows if item[0] != slot]
                epoch_ids = [fresh_max]
                epoch_ids.extend(issuer for _s, _v, issuer in kept_epoch)
                glob = fit_vector(kept_epoch, pre, eta, max(len(pre) - 1, max(epoch_ids)))
            else:
                glob = soc_full
            if slot in org_slots[org]:
                kept_org = [item for item in by_org[org] if item[0] != slot]
                org_vec = fit_vector(kept_org, blank, 1.0, hist_max)
            else:
                org_vec = org_full[org]
            hits = buckets.get(slot, [])
            for key, society_only in (("loo", False), ("loo_global", True)):
                schema_v = fair.schema_value(world, row[key], slot)
                row[key][slot] = fair.answer_slot(
                    cfg, hits, row[key][slot], trust_for(glob, org_vec, cfg, society_only), schema_v
                )
        for key in belief:
            belief[key] = row[key]
            answers[key].append([list(row[key]) for _ in range(rx.N_AGENTS)])
        society = soc_full
    packed = {}
    for key, grid in answers.items():
        counts = rx.empty_counts()
        rx.consume(counts, world, grid, 0.0)
        packed[key] = rx.pack_metrics(counts)
    return packed


def rag_blank():
    import run_rag as rag
    return rag.blank_weights()


def balanced(metrics):
    return (metrics["changed"] + metrics["stable"] + metrics["derived"]) / 3.0


def store(seed_rows, mode, regime, metrics):
    seed_rows[mode][regime].append(
        {key: metrics[key] for key in ("sp", "changed", "stable", "derived", "obsolete", "delay", "cfa")}
    )


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if task == "verify":
        modes = ("loo", "loo_global")
        seeds = [600]
        regimes = ("stationary",)
    elif task == "dev":
        modes = ("hier", "global", "loo", "loo_global")
        seeds = DEV
        regimes = REGIMES
    elif task == "test":
        modes = ("hier", "global", "loo", "loo_global")
        seeds = list(range(700, 732))
        regimes = REGIMES
    else:
        raise SystemExit("task must be verify, dev, or test")
    summary = {mode: {} for mode in modes}
    seed_rows = {mode: {regime: [] for regime in regimes} for mode in modes}
    hier_cfg = next(item for item in fair.RULES if item["name"] == "hier")
    global_cfg = next(item for item in fair.RULES if item["name"] == "global")
    for regime in regimes:
        for seed in seeds:
            world = fair.load_world(seed, regime)
            parts = []
            if "hier" in modes:
                metrics = fair.run_cfg(world, hier_cfg)
                store(seed_rows, "hier", regime, metrics)
                parts.append(f"hier:{balanced(metrics):.3f}/{metrics['sp']:.3f}")
            if "global" in modes:
                metrics = fair.run_cfg(world, global_cfg)
                store(seed_rows, "global", regime, metrics)
                parts.append(f"global:{balanced(metrics):.3f}/{metrics['sp']:.3f}")
            held = run_loo(world)
            for mode in ("loo", "loo_global"):
                if mode not in modes:
                    continue
                metrics = held[mode]
                store(seed_rows, mode, regime, metrics)
                parts.append(f"{mode}:{balanced(metrics):.3f}/{metrics['sp']:.3f}")
            print(regime, seed, " ".join(parts), flush=True)
    for mode in modes:
        by = {}
        for regime, rows in seed_rows[mode].items():
            by[regime] = sum(balanced(row) for row in rows) / len(rows)
        summary[mode] = {"by_regime": by, "mean": sum(by.values()) / len(by)}
    payload = {"task": task, "summary": summary, "seeds": seed_rows}
    out = rx.ROOT / "results" / f"heldout_{task}.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
