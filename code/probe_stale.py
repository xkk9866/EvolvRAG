"""How much of the disagreement in Stock and Flight is staleness?

For every claim on a gold item, classify it as correct, stale (equal to the
gold value of the same item on an earlier day, lag 1..3) or other. Also
report how often each attribute's gold value changes from one day to the next.
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_fusion as fu


def main():
    for name, fn in (("stock", fu.stock_days), ("flight", fu.flight_days)):
        days = fn()
        kind = Counter()
        per_attr = defaultdict(Counter)
        per_src = defaultdict(Counter)
        change = defaultdict(lambda: [0, 0])
        for d, day in enumerate(days):
            gold = day["gold"]
            for slot, v in gold.items():
                attr = slot.split("|")[1]
                if d > 0 and slot in days[d - 1]["gold"]:
                    change[attr][0] += days[d - 1]["gold"][slot] != v
                    change[attr][1] += 1
            for slot, v, src, _org in day["claims"]:
                if slot not in gold:
                    continue
                attr = slot.split("|")[1]
                if v == gold[slot]:
                    k = "correct"
                else:
                    k = "other"
                    for lag in (1, 2, 3):
                        if d - lag >= 0 and days[d - lag]["gold"].get(slot) == v and v != gold[slot]:
                            k = f"stale{lag}"
                            break
                kind[k] += 1
                per_attr[attr][k] += 1
                per_src[src][k] += 1
        tot = sum(kind.values())
        print(name, {k: round(c / tot, 4) for k, c in kind.most_common()})
        for a, c in per_attr.items():
            t = sum(c.values())
            ch = change[a]
            print("  ", a, "change/day", round(ch[0] / max(ch[1], 1), 3), {k: round(x / t, 3) for k, x in c.most_common()})
        stale_src = sorted(((c["stale1"] + c["stale2"] + c["stale3"]) / max(sum(c.values()), 1), s) for s, c in per_src.items())
        print("   most stale sources", [(s, round(x, 2)) for x, s in stale_src[-6:]])


if __name__ == "__main__":
    main()
