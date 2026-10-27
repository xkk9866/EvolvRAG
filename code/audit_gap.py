"""Audit whether a cleared weight can sit just above 0.55.

The organization record is a cumulative CRH fit. The society record is an
epoch mixture. This script only reads seeds 700-731. It does not change the rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as rx
import run_fair as fair

TAU = 0.55
CFG = next(cfg for cfg in fair.RULES if cfg["name"] == "hier")


def cleared(vec):
    return [fair.weight_of(vec, i) for i in range(len(vec)) if fair.weight_of(vec, i) + 1e-9 >= TAU]


def main():
    gap_org = 0
    gap_soc = 0
    decisions = 0
    count_differ = 0
    count_wins_lower_sum = 0
    org_cleared_min = 1.0
    org_blocked_max = 0.0
    soc_cleared_min = 1.0
    for regime in ("stationary", "topic", "open_rumor", "sybil"):
        for seed in range(700, 732):
            world = fair.load_world(seed, regime)
            names = world.names
            book = fair.TrustBook(CFG)
            mail = []
            for epoch_docs in world.visible:
                if book.org is not None:
                    for org_i, org_vec in enumerate(book.org):
                        decisions += 1
                        oc = cleared(org_vec)
                        sc = cleared(book.plain)
                        if oc and min(oc) < 0.60 - 1e-9:
                            gap_org += 1
                        if sc and min(sc) < 0.60 - 1e-9:
                            gap_soc += 1
                        if oc:
                            org_cleared_min = min(org_cleared_min, min(oc))
                        below = [fair.weight_of(org_vec, i) for i in range(len(org_vec)) if fair.weight_of(org_vec, i) + 1e-9 < TAU]
                        if below:
                            org_blocked_max = max(org_blocked_max, max(below))
                        if sc:
                            soc_cleared_min = min(soc_cleared_min, min(sc))
                        if len(oc) != len(sc):
                            count_differ += 1
                            winner = oc if len(oc) > len(sc) else sc
                            loser = sc if len(oc) > len(sc) else oc
                            if sum(winner) + 1e-12 < sum(loser):
                                count_wins_lower_sum += 1
                fresh = list(epoch_docs)
                mail.extend(fresh)
                find = fair.copy_find(mail, names)
                usable = fair.collapsed(mail, find)
                book.update(fair.collapsed(fresh, find), usable, names)
    print("decisions", decisions)
    print("org cleared below 0.60", gap_org)
    print("soc cleared below 0.60", gap_soc)
    print("org cleared min", round(org_cleared_min, 4))
    print("org blocked max", round(org_blocked_max, 4))
    print("soc cleared min", round(soc_cleared_min, 4))
    print("counts differ", count_differ)
    print("more issuers but lower sum", count_wins_lower_sum)


if __name__ == "__main__":
    main()
