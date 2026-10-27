import json
from pathlib import Path

p = json.loads(Path("results/rag_v2.json").read_text(encoding="utf-8"))
sel = p["protocol"]["selected"]
print("SELECTED", sel["corpus"], sel["cfg"], "min", round(sel["dev_min_sp"], 4))
print("DEV SP", {k: round(v, 4) for k, v in sel["dev_sp"].items()})
print("--- DEV GRID min / stat / topic / open ---")
rows = sorted(sel["grid"], key=lambda r: -r["min_sp"])
for r in rows:
    sp = r["sp"]
    print(
        f"{r['min_sp']:.3f}  {sp['stationary']:.3f} {sp['topic']:.3f} {sp['open_rumor']:.3f}  "
        f"{r['corpus']:14} {r['cfg']['rule']:16} "
        f"der {r['derived']['stationary']:.3f}/{r['derived']['topic']:.3f}/{r['derived']['open_rumor']:.3f}"
    )
print("--- TEST selected and neighbors ---")
label = f"{sel['corpus']}|{sel['rule_index']}"
rules = p["protocol"]["rules"]
for regime in ("stationary", "topic", "open_rumor"):
    print("REGIME", regime)
    cells = p["test"][regime]
    cov = p["test_coverage"][regime][sel["corpus"]]
    print(" coverage", {k: round(v, 3) for k, v in cov.items()})
    for corpus in ("local", "corroborated", "union", "all"):
        for i, cfg in enumerate(rules):
            if corpus != sel["corpus"] and i != sel["rule_index"] and not (
                cfg["rule"] in ("majority", "cluster_newest", "cluster_oldest", "recency", "gate")
                and corpus in ("local", "corroborated", "all")
            ):
                continue
            cell = cells[f"{corpus}|{i}"]
            mark = " *" if f"{corpus}|{i}" == label else ""
            print(
                f"  {corpus:14} {cfg['rule']:16} SP {cell['sp']:.3f} ch {cell['changed']:.3f} "
                f"st {cell['stable']:.3f} der {cell['derived']:.3f} obs {cell['obsolete']:.3f} "
                f"unnec {cell['unnec']:.3f} delay {cell['delay']:.3f}{mark}"
            )
print("--- TESTS sp ---")
for row in p["tests"]:
    if row["metric"] != "sp":
        continue
    print(
        f"{row['regime']:12} {row['left']} - {row['right']} {row['mean_diff']:+.4f} "
        f"sd {row['std_diff']:.4f} p {row['p']:.6f} n {row['n']}"
    )
