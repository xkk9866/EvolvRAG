"""Find a held-out revision where the organization record supplies the current value."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as rx
import run_fair as fair
import run_heldout as held

world = fair.load_world(700, "topic")
names = world.names
cfg = next(item for item in fair.RULES if item["name"] == "hier")
order = list(world.mutable_ids) + list(world.stable_ids) + list(world.derived_ids)
belief = list(world.init_belief[0])
mail = []
society = held.rag_blank()
shown = 0
for epoch_i, epoch_docs in enumerate(world.visible):
    mail.extend(epoch_docs)
    find = fair.copy_find(mail, names)
    usable = fair.collapsed(mail, find)
    fresh_c = fair.collapsed(epoch_docs, find)
    hist_rows, hist_max = held.claims_from(usable, names)
    fresh_rows, fresh_max = held.claims_from(fresh_c, names)
    buckets = fair.index_hits(usable, names)
    pre = society
    blank = fair.pad(held.rag_blank(), hist_max + 1)
    row = list(belief)
    for slot in order:
        if slot not in world.mutable_ids:
            schema_v = fair.schema_value(world, row, slot)
            org = slot % 8
            glob = held.fit_vector([item for item in fresh_rows if item[0] != slot], pre, 0.5, max(len(pre) - 1, fresh_max))
            org_vec = held.fit_vector([item for item in hist_rows if item[0] % 8 == org and item[0] != slot], blank, 1.0, hist_max)
            row[slot] = fair.answer_slot(cfg, buckets.get(slot, []), row[slot], held.trust_for(glob, org_vec, cfg, False), schema_v)
            continue
        truth = world.timeline[epoch_i][slot]
        org = slot % 8
        glob = held.fit_vector([item for item in fresh_rows if item[0] != slot], pre, 0.5, max(len(pre) - 1, fresh_max))
        org_vec = held.fit_vector([item for item in hist_rows if item[0] % 8 == org and item[0] != slot], blank, 1.0, hist_max)
        trust = held.trust_for(glob, org_vec, cfg, False)
        schema_v = fair.schema_value(world, row, slot)
        hits = buckets.get(slot, [])
        answer = fair.answer_slot(cfg, hits, row[slot], trust, schema_v)
        newest = max(hits, key=lambda hit: (hit["time"], hit.get("seq", 0)))["value"] if hits else None
        if shown < 6 and truth != "v0" and answer == truth and newest != truth and epoch_i >= 2:
            cleared = [(i, round(trust(i), 2)) for i in range(10) if trust(i) >= 0.55]
            org_only = [(i, round(org_vec[i], 2), round(glob[i] if i < len(glob) else 0.4, 2)) for i in range(10)]
            print("epoch", epoch_i + 1, "slot", slot, "org", org, "truth", truth, "newest", newest, "cleared", cleared)
            print("  weights org/soc", org_only)
            shown += 1
        row[slot] = answer
    belief = row
    society = held.fit_vector(fresh_rows, pre, 0.5, max(len(pre) - 1, fresh_max))
    print("finished epoch", epoch_i + 1, flush=True)
