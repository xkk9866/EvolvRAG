import run_experiment as rx
import run_rag as rag

rules = [
    {"rule": "majority", "eta": 1, "topic": False, "schema": False},
    {"rule": "cluster_oldest", "eta": 1, "topic": False, "schema": False},
    {"rule": "cluster_newest", "eta": 1, "topic": False, "schema": False},
    {"rule": "recency", "eta": 1, "topic": False, "schema": False},
    {"rule": "gate", "eta": 0.25, "topic": True, "schema": True},
]
for regime in ("stationary", "open_rumor"):
    world = rx.generate_trajectory(100, 0.16, 0.45, regime=regime)
    for corpus in ("local", "corroborated", "all"):
        order, epochs, cov = rag.collect_trace(world, corpus)
        bits = [f"held {cov['held_value']:.3f}"]
        for cfg in rules:
            m = rag.score_trace(world, order, epochs, cov, cfg)
            bits.append(f"{cfg['rule'][:6]} {m['sp']:.3f}")
        print(regime, corpus, " ".join(bits), flush=True)
