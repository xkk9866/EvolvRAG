import json
from pathlib import Path

payload = json.loads(Path("results/fair.json").read_text(encoding="utf-8"))
rules = payload["protocol"]["rules"]
names = [cfg["name"] for cfg in rules]
print("SELECTED", payload["protocol"]["selection"]["cfg"]["name"], "dev", round(payload["protocol"]["selection"]["dev_mean_balanced"], 3))
print("DEV", {k: round(v, 3) for k, v in payload["protocol"]["selection"]["dev_balanced"].items()})
print("AUDIT", payload["address_audit_dev"])
print("TIE", payload["tie_audit_seed_500"])
print("--- TEST ---")
for regime, cells in payload["test"].items():
    print("REGIME", regime)
    for i, name in enumerate(names):
        cell = cells[str(i)]
        bits = " ".join(f"{k}={cell[k]:.3f}" for k in ("sp", "changed", "stable", "derived", "delay", "obsolete", "cfa"))
        print(f"  {i:2d} {name:20s} {bits}")
print("--- PAIRED sp/changed vs 4 ---")
for row in payload["tests"]:
    if row["left"] != 4:
        continue
    if row["metric"] not in ("sp", "changed", "stable", "derived"):
        continue
    if row["right"] not in (0, 1, 2, 3, 8, 9, 10, 11, 12) and row["metric"] != "sp":
        continue
    if row["metric"] == "sp" or row["right"] in (0, 3, 8, 12):
        print(f"{row['regime']:16s} {names[row['right']]:20s} {row['metric']:8s} {row['mean_diff']:+.3f} sd {row['std_diff']:.3f} p {row['p']:.4f}")
print("--- SENS ---")
for row in payload["sensitivity_cutoff_not_used_for_selection"]:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
print("--- STRESS ---")
for name, block in payload["stress"].items():
    print(name)
    for regime, row in block.items():
        print(" ", regime, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != "paired_sp"}, "p", round(row["paired_sp"]["p"], 4), "n", row["paired_sp"]["n"])
print("--- POBS ---")
for row in payload["p_obs"]:
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})
print("seconds", round(payload["seconds"]))
