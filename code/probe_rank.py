import run_experiment as rx
import run_rag as rag

world = rx.generate_trajectory(100, 0.16, 0.45)
order, epochs, cov = rag.collect_trace(world, "corroborated")
# Find a slot with several corroborated values.
from collections import defaultdict
by = defaultdict(list)
for epoch_docs in world.visible:
    for doc in epoch_docs:
        if len(doc["receivers"]) >= 2:
            by[doc["slot"]].append(doc)
multi = [(slot, docs) for slot, docs in by.items() if len({d["value"] for d in docs}) >= 2]
multi.sort(key=lambda item: -len(item[1]))
slot, docs = multi[0]
print("slot", slot, "n", len(docs), "values", sorted({d["value"] for d in docs}))
print("truths", [world.timeline[t][slot] for t in range(rx.EPOCHS)])
query = rx.query_text(world, slot)
# Rebuild index at last epoch the same way collect does.
mail = []
for epoch_i, epoch_docs in enumerate(world.visible):
    mail.extend([d for d in epoch_docs if len(d["receivers"]) >= 2])
    if epoch_i != rx.EPOCHS - 1:
        continue
    index = rag.Index(mail)
    hits = index.search(query)
    print("query", query)
    print("n_hits", len(hits), "top", [(h["time"], h["value"], h["issuer"]) for h in hits[:8]])
    q = __import__("collections").Counter(rag.tokens(query))
    scored = []
    for i, doc in enumerate(mail):
        if doc["slot"] != slot:
            continue
        scored.append((round(index._score(q, i), 4), doc["time"], doc["value"], doc["text"]))
    scored.sort(reverse=True)
    print("scores")
    for row in scored:
        print(" ", row)
