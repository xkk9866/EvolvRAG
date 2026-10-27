"""Weight histogram of contemporaneous accuracy at the last epoch."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_acc as acc
import run_experiment as rx
import run_fair as fair
import run_heldout as held


def main():
    world = fair.load_world(800, "topic")
    names = world.names
    mail = []
    epoch_org = []
    fresh_max = 0
    for epoch_docs in world.visible:
        mail.extend(epoch_docs)
        find = fair.copy_find(mail, names)
        fresh_c = fair.collapsed(epoch_docs, find)
        fresh_rows, fresh_max = held.claims_from(fresh_c, names)
        by_g = [[] for _ in range(len(rx.ORGS))]
        for slot, value, issuer in fresh_rows:
            by_g[slot % len(rx.ORGS)].append((slot, value, issuer))
        epoch_org.append(by_g)
    for g in (0, 5):
        rows = [epoch_org[t][g] for t in range(len(epoch_org))]
        vec = acc.accuracy_vector(rows, fresh_max, exclude=-1)
        print("org", g, [round(w, 3) for w in vec[:10]], "claims", sum(len(r) for r in rows), flush=True)
        # per round support
        for t, r in enumerate(rows):
            slots = defaultdict(int)
            for slot, _v, _i in r:
                slots[slot] += 1
            multi = sum(1 for n in slots.values() if n >= 2)
            print("  epoch", t + 1, "claims", len(r), "slots_with_2", multi, flush=True)


if __name__ == "__main__":
    main()
