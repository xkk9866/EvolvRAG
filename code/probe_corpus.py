import time
import run_experiment as rx
import run_rag as r

GATE = {"rule": "gate", "eta": 0.25, "topic": True, "schema": True}
MAJ = {"rule": "majority", "eta": 1.0, "topic": False, "schema": False}
for corpus in ("local", "union", "corroborated", "all"):
    t = time.time()
    scores = {name: [] for name in ("majority", "gate")}
    for seed in (100, 101):
        world = rx.generate_trajectory(seed, 0.16, 0.45)
        order, epochs, cov = r.collect_trace(world, corpus)
        scores["majority"].append(r.score_trace(world, order, epochs, cov, MAJ)["sp"])
        scores["gate"].append(r.score_trace(world, order, epochs, cov, GATE)["sp"])
        print(corpus, seed, "held", round(cov["held_value"], 3), "recall", round(cov["recall5"], 3),
              "maj", round(scores["majority"][-1], 3), "gate", round(scores["gate"][-1], 3), flush=True)
    print(corpus, "seconds", round(time.time() - t, 1), flush=True)
