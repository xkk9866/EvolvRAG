import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_causal as rc
import run_experiment as rx

p = json.loads((rx.ROOT / "results" / "causal.json").read_text(encoding="utf-8"))
s = p["stationary"]
a = s["NoShrink"]["sp_seeds"]
b = s["Central"]["sp_seeds"]
d = [x - y for x, y in zip(a, b)]
mu, sd = rx.mean_std(d)
print("NoShrink-Central sp", round(mu, 4), round(sd, 4), rc.perm_p(d))
for name in ("sp", "changed", "stable", "cfa", "derived", "comm"):
    print("std", name, {m: round(s[m][name + "_std"], 4) for m in ("PoolMajority", "PoolCount", "PoolUniform", "PoolCRH", "PoolOpen", "PoolWeight", "NoShrink", "Central", "Majority", "EvolvRAG")})
topic = p["stress"]["topic"]["cells"]
d = [x - y for x, y in zip(topic["Majority"]["sp_seeds"], topic["Central"]["sp_seeds"])]
mu, sd = rx.mean_std(d)
print("topic Majority-Central", round(mu, 4), round(sd, 4), rc.perm_p(d))
# lambda dominance
for lam_row_i, lam in enumerate([i / 10 for i in range(11)]):
    scores = {m: p["lambda"][m][lam_row_i]["score"] for m in ("Central", "Majority", "PoolMajority", "NoShrink", "PoolOpen", "PoolUniform")}
    print(lam, {m: round(v, 3) for m, v in scores.items()})
