"""Decide with records fit on the current round, including and excluding the judged entity."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as rx
import run_fair as fair


def run(world, leave_out):
    cfg = next(item for item in fair.RULES if item["name"] == "hier")
    names = world.names
    order = list(world.mutable_ids) + list(world.stable_ids) + list(world.derived_ids)
    belief = list(world.init_belief[0])
    mail = []
    book = fair.TrustBook(cfg)
    answers = []
    for epoch_docs in world.visible:
        fresh = list(epoch_docs)
        mail.extend(epoch_docs)
        find = fair.copy_find(mail, names)
        usable = fair.collapsed(mail, find)
        fresh_c = fair.collapsed(fresh, find)
        book.update(fresh_c, usable, names)
        buckets = fair.index_hits(usable, names)
        row = list(belief)
        if not leave_out:
            for slot in order:
                schema_v = fair.schema_value(world, row, slot)
                row[slot] = fair.answer_slot(cfg, buckets.get(slot, []), row[slot], book.for_slot(slot), schema_v)
        else:
            for slot in order:
                held = [doc for doc in usable if fair.entity_of_text(doc["text"], names) != slot]
                held_fresh = [doc for doc in fresh_c if fair.entity_of_text(doc["text"], names) != slot]
                local = fair.TrustBook(cfg)
                local.update(held_fresh, held, names)
                schema_v = fair.schema_value(world, row, slot)
                row[slot] = fair.answer_slot(cfg, buckets.get(slot, []), row[slot], local.for_slot(slot), schema_v)
        belief = row
        answers.append([list(row) for _ in range(rx.N_AGENTS)])
    counts = rx.empty_counts()
    rx.consume(counts, world, answers, 0.0)
    packed = rx.pack_metrics(counts)
    return (packed["changed"] + packed["stable"] + packed["derived"]) / 3, packed["sp"], packed["changed"], packed["stable"]


cfg = next(item for item in fair.RULES if item["name"] == "hier")
for regime in ("stationary", "topic", "open_rumor", "sybil"):
    for seed in (600, 601):
        world = fair.load_world(seed, regime)
        hier = fair.run_cfg(world, cfg)
        hb = (hier["changed"] + hier["stable"] + hier["derived"]) / 3
        same = run(world, False)
        print(
            regime, seed,
            "hier", round(hb, 3), round(hier["sp"], 3),
            "same-round", [round(x, 3) for x in same],
            flush=True,
        )
