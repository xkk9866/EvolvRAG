import run_experiment as rx
import run_rag as rag

world = rx.generate_trajectory(100, 0.16, 0.45)
order, epochs, cov = rag.collect_trace(world, "corroborated")
cfg = {"rule": "bm25", "eta": 1.0, "topic": False, "schema": False}
maj = {"rule": "majority", "eta": 1.0, "topic": False, "schema": False}
print("coverage", {k: round(cov[k], 3) if isinstance(cov[k], float) else cov[k] for k in cov})
for name, c in (("bm25", cfg), ("maj", maj)):
    m = rag.score_trace(world, order, epochs, cov, c)
    print(name, {k: round(m[k], 3) for k in ("sp", "changed", "stable", "derived", "cfa")})

# Inspect one mutable slot at the last epoch: top hit vs truth, and score ties.
slot = world.mutable_ids[0]
query = rx.query_text(world, slot)
index_docs = []
for epoch_docs in world.visible:
    index_docs.extend([d for d in epoch_docs if len(d["receivers"]) >= 2 and d["slot"] == slot])
print("slot", slot, "query", query, "n_corr", len(index_docs))
print("last truth", world.timeline[-1][slot])
hits = epochs[-1][0]["hits"][order.index(slot)]
print("n_hits", len(hits))
for hit in hits[:8]:
    print(" ", hit["time"], hit["value"], hit["issuer"], hit["text"][:80])
