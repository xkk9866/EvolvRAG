"""Print the numbers the manuscript is allowed to use."""

import json
from pathlib import Path

from run_experiment import mean_std, signflip_p

ROOT = Path(__file__).resolve().parents[1]
payload = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))


def cell(method, rate=0.16, p_obs=0.45):
    for item in payload["cells"]:
        if item["method"] == method and abs(item["rate"] - rate) < 1e-9 and abs(item["p_obs"] - p_obs) < 1e-9:
            return item
    raise KeyError(method)


def paired(a, b, metric):
    diffs = [x - y for x, y in zip(cell(a)[metric + "_seeds"], cell(b)[metric + "_seeds"])]
    mu, sd = mean_std(diffs)
    return mu, sd, signflip_p(diffs)


methods = ["Static", "Timestamp", "Reset", "LTM", "Majority", "CRH", "PoolCRH", "Central", "EvolvRAG", "NoTrust", "NoRelay", "NoDep", "NoExpire"]
headers = ["sp", "cfa", "changed", "stable", "derived", "obsolete", "unnec", "delay", "comm"]
print("DEFAULT")
for method in methods:
    item = cell(method)
    nums = " ".join(f"{item[h]:.4f}" for h in headers)
    stds = " ".join(f"{item[h + '_std']:.4f}" for h in ("sp", "cfa", "changed", "stable"))
    print(method, nums, "std", stds)

print("PAIRS")
for metric in ("sp", "cfa", "changed", "stable"):
    for other in ("Majority", "CRH", "PoolCRH", "LTM", "EvolvRAG", "Static"):
        mu, sd, p = paired("Central", other, metric)
        print(f"Central-{other:10} {metric:8} {mu:+.4f} +/- {sd:.4f} p={p:.4f}")

print("EPOCH")
for method in ("Central", "EvolvRAG", "CRH", "Majority", "LTM", "PoolCRH", "Static"):
    item = cell(method)
    means = " ".join(f"{v:.3f}" for v in item["epoch_cfa_mean"])
    stds = " ".join(f"{v:.3f}" for v in item["epoch_cfa_std"])
    print(method, means, "|", stds)

sens = payload["sensitivity"]
sps = [row["sp"] for row in sens]
print("SENS n", len(sens), "min", min(sps), "max", max(sps))
chosen = payload["config"]["locked"]
print("CHOSEN", chosen)
near = [row for row in sens if max(sps) - row["sp"] <= 0.01]
print("NEAR", len(near), "of", len(sens))

print("BUDGET")
for row in payload["budget"]:
    print(row["budget"], f"{row['sp']:.4f}", f"{row['cfa']:.4f}", f"{row['stable']:.4f}", f"{row['changed']:.4f}", f"{row['comm']:.4f}")

print("OBS")
for p_obs in (0.25, 0.45, 0.70, 1.0):
    for method in ("Central", "EvolvRAG", "CRH", "Majority", "LTM"):
        item = cell(method, 0.16, p_obs)
        print(p_obs, method, f"{item['sp']:.4f}", f"{item['cfa']:.4f}", f"{item['changed']:.4f}", f"{item['stable']:.4f}", f"{item['comm']:.4f}")
