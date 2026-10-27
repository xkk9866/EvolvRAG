"""Print the held-out table from results/hier.json. Does not rescore."""
import json
from pathlib import Path

p = json.loads(Path("results/hier.json").read_text(encoding="utf-8"))
rules = p["protocol"]["rules"]
test = p["test"]
names = [cfg["name"] for cfg in rules]


def cell(regime, i, key):
    return test[regime][str(i)][key]


print("SELECTED", p["protocol"]["selection"]["cfg"]["name"], round(p["protocol"]["selection"]["dev_mean_balanced"], 4))
print("TEST means")
header = f"{'rule':16} {'SPsp':6} {'ch':6} {'st':6} {'de':6} {'Tsp':6} {'Osp':6} {'Ysp':6} {'Vsp':6}"
print(header)
for i, name in enumerate(names):
    print(
        f"{name:16} "
        f"{cell('stationary', i, 'sp'):6.3f} "
        f"{cell('stationary', i, 'changed'):6.3f} "
        f"{cell('stationary', i, 'stable'):6.3f} "
        f"{cell('stationary', i, 'derived'):6.3f} "
        f"{cell('topic', i, 'sp'):6.3f} "
        f"{cell('open_rumor', i, 'sp'):6.3f} "
        f"{cell('sybil', i, 'sp'):6.3f} "
        f"{cell('overlap', i, 'sp'):6.3f}"
    )

label = p["protocol"]["selection"]["index"]
print("\nfour-regime balanced means")
for i, name in enumerate(names):
    vals = []
    for regime in ("stationary", "topic", "open_rumor", "sybil"):
        c = test[regime][str(i)]
        vals.append((c["changed"] + c["stable"] + c["derived"]) / 3)
    print(f"{name:16} {sum(vals)/4:.4f} " + " ".join(f"{v:.3f}" for v in vals))

print("\npaired vs selected, sp")
for row in p.get("tests", []):
    if row["metric"] != "sp":
        continue
    if row["left"] != label:
        continue
    other = names[row["right"]]
    if other not in ("newest", "majority", "crh", "gate", "global", "org", "hier-nocopy", "hier-noschema", "hier-m0", "hier-m2", "hier-50", "shortcut-newest"):
        continue
    if row["regime"] not in ("stationary", "topic", "open_rumor", "sybil", "overlap"):
        continue
    print(f"{row['regime']:12} {other:16} {row['mean_diff']:+.3f} sd {row['std_diff']:.3f} p {row['p']:.4g} n {row['n']}")

print("\nstationary changed/stable/derived vs gate global org")
for metric in ("changed", "stable", "derived", "sp"):
    for row in p.get("tests", []):
        if row["regime"] != "stationary" or row["metric"] != metric or row["left"] != label:
            continue
        other = names[row["right"]]
        if other in ("gate", "global", "org", "newest", "hier-noschema", "shortcut-newest"):
            print(f"{metric:8} {other:16} {row['mean_diff']:+.3f} sd {row['std_diff']:.3f} p {row['p']:.4g}")

audit = p.get("address_audit_dev")
if audit:
    print("audit", audit)
tie = p.get("tie_audit_seed_700")
if tie:
    print("tie", tie)
print("keys", sorted(p.keys()))
