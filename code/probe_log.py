import run_experiment as rx
import run_rag as rag

world = rx.generate_trajectory(5, 0.16, 0.45)
slot = next(s for s, name in enumerate(world.names) if "055" in name)
print("slot", slot, world.names[slot], "role", world.roles[slot])
print("truth", [world.timeline[t][slot] for t in range(rx.EPOCHS)])
order, epochs, cov = rag.collect_trace(world, "corroborated")
# replay gate beliefs at epoch 3 (index 2) is inside score; print hits instead
hits = epochs[2][0]["hits"][order.index(slot)]
print("query", rx.query_text(world, slot))
print("hits", [(h["time"], h["value"], h["issuer"], len(h.get("receivers", []))) for h in hits[:8]])
pos = order.index(slot)
for cfg, name in (
    ({"rule": "gate", "eta": 0.25, "topic": True, "schema": True}, "gate"),
    ({"rule": "cluster_newest", "eta": 1, "topic": False, "schema": False}, "newest"),
):
    beliefs = [list(row) for row in world.init_belief]
    weights = [rag.blank_weights() for _ in range(rx.N_AGENTS)]
    topic_w = [[rag.blank_weights() for _ in range(len(rx.ORGS))] for _ in range(rx.N_AGENTS)]
    for epoch_i, agents in enumerate(epochs):
        for agent in range(rx.N_AGENTS):
            row = list(beliefs[agent])
            for s, hits in zip(order, agents[agent]["hits"]):
                org = s % len(rx.ORGS)
                trust = topic_w[agent][org] if cfg.get("topic") else weights[agent]
                schema_value = None
                parent = world.parent.get(s)
                if cfg.get("schema") and parent is not None:
                    parent_v = row[parent]
                    schema_value = "EXPIRED" if parent_v == "EXPIRED" else f"via-{parent_v}"
                row[s] = rag.answer_hits(cfg["rule"], hits, row[s], trust, schema_value, cfg.get("schema"), 1)
            beliefs[agent] = row
            claims_by_org = agents[agent]["claims"]
            if cfg["rule"] == "gate":
                if cfg.get("topic"):
                    for org, claims in claims_by_org.items():
                        fresh = rag.weights_from_claims(claims, topic_w[agent][org])
                        topic_w[agent][org] = rag.mix(topic_w[agent][org], fresh, cfg["eta"])
                else:
                    claims = [c for rows in claims_by_org.values() for c in rows]
                    if claims:
                        fresh = rag.weights_from_claims(claims, weights[agent])
                        weights[agent] = rag.mix(weights[agent], fresh, cfg["eta"])
        if epoch_i == 2:
            answers = sorted({beliefs[a][slot] for a in range(rx.N_AGENTS)})
            print(name, "epoch3", answers, "truth", world.timeline[2][slot])
