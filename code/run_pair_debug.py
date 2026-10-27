"""Contemporaneous pairwise agreement inside two topic organizations."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as rx
import run_fair as fair
import run_heldout as held


def main():
    world = fair.load_world(800, "topic")
    names = world.names
    mail = []
    pairs = [defaultdict(lambda: [0, 0]) for _ in range(len(rx.ORGS))]
    for epoch_docs in world.visible:
        mail.extend(epoch_docs)
        find = fair.copy_find(mail, names)
        fresh = fair.collapsed(epoch_docs, find)
        rows, _max = held.claims_from(fresh, names)
        by_slot = defaultdict(list)
        for slot, value, issuer in rows:
            if issuer < 10:
                by_slot[slot].append((issuer, value))
        for slot, claims in by_slot.items():
            g = slot % len(rx.ORGS)
            for i, (a, va) in enumerate(claims):
                for b, vb in claims[i + 1 :]:
                    if a == b:
                        continue
                    rec = pairs[g][(min(a, b), max(a, b))]
                    rec[1] += 1
                    rec[0] += int(va == vb)
    for g in (0, 5):
        print("org", g, flush=True)
        agree = [[0.0] * 10 for _ in range(10)]
        for (a, b), (hit, n) in pairs[g].items():
            if n:
                agree[a][b] = agree[b][a] = hit / n
        for i in range(10):
            print(i, [round(x, 2) for x in agree[i]], flush=True)


if __name__ == "__main__":
    main()
