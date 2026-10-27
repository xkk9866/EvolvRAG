import json
from pathlib import Path

payload = json.loads(Path("results/causal.json").read_text(encoding="utf-8"))
keys = ["sp", "cfa", "changed", "stable", "derived", "comm", "upload_distinct", "upload_naive", "total_distinct", "total_naive", "total_distinct_bytes", "total_naive_bytes"]
print("ACCOUNT")
for k, v in payload["account"].items():
    if isinstance(v, float):
        print(f"  {k} {v:.4f}")
    else:
        print(f"  {k} {v}")
print("STATIONARY")
for method, cell in payload["stationary"].items():
    bits = []
    for key in keys:
        if key in cell:
            bits.append(f"{key}={cell[key]:.4f}")
    print(method, " ".join(bits))
print("TESTS")
for row in payload["tests"]:
    if row["metric"] in ("sp", "changed", "stable") and row["right"] in ("PoolMajority", "Central"):
        print(f"  {row['metric']} {row['left']}-{row['right']} {row['mean_diff']:+.4f} sd={row['std_diff']:.4f} p={row['p']:.6f} n={row['n']}")
print("LAMBDA")
for method in ("PoolMajority", "PoolUniform", "PoolOpen", "PoolWeight", "Central", "NoShrink", "Majority"):
    rows = payload["lambda"][method]
    shown = ", ".join(f"{row['lambda']:.1f}:{row['score']:.3f}" for row in rows if row["lambda"] in (0.0, 0.25, 0.5, 0.75, 1.0))
    print(method, shown)
print("SELECTIVE", payload["selective"]["mean_uploads_per_agent_epoch"])
for method, cell in payload["selective"]["cells"].items():
    print(" ", method, f"sp={cell['sp']:.4f}", f"changed={cell['changed']:.4f}", f"stable={cell['stable']:.4f}", f"comm={cell['comm']:.4f}")
for regime, block in payload["stress"].items():
    print("REGIME", regime)
    for method, cell in block["cells"].items():
        print(f"  {method} sp={cell['sp']:.4f} cfa={cell['cfa']:.4f} changed={cell['changed']:.4f} stable={cell['stable']:.4f}")
    for row in block["tests"]:
        print(f"  test {row['left']}-{row['right']} {row['mean_diff']:+.4f} p={row['p']:.4f}")
for row in payload["scale_agents"]:
    print("AGENTS", row["agents"])
    for method, cell in row["cells"].items():
        print(f"  {method} sp={cell['sp']:.4f} changed={cell['changed']:.4f} stable={cell['stable']:.4f} comm={cell['comm']:.4f}")
for row in payload["scale_keys"]:
    print("KEYS", row["mutable"])
    for method, cell in row["cells"].items():
        print(f"  {method} sp={cell['sp']:.4f} changed={cell['changed']:.4f} stable={cell['stable']:.4f}")
