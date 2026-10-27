import json

d = json.load(open(r"results/metrics.json", encoding="utf-8"))
keys = ["cfa", "changed", "stable", "derived", "obsolete", "unnec", "delay", "comm"]
for c in d["cells"]:
    if abs(c["rate"] - 0.16) < 1e-9 and abs(c["p_obs"] - 0.45) < 1e-9:
        print(c["method"])
        for k in keys:
            print(f"  {k} {c[k]:.4f} +/- {c[k + '_std']:.4f}")
        if c["method"] in ("EvolvRAG", "LTM", "Static", "Central", "Timestamp", "Reset"):
            print("  epoch", " ".join(f"{x:.3f}" for x in c["epoch_cfa_mean"]))
            print("  estd", " ".join(f"{x:.3f}" for x in c["epoch_cfa_std"]))
print("TRACE")
print(json.dumps(d["trace"], indent=2))
