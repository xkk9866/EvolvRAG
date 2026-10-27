import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_fair as fair

world = fair.load_world(700, "topic")
mail = []
for epoch_i, docs in enumerate(world.visible[:3]):
    mail.extend(docs)
find = fair.copy_find(mail, world.names)
usable = fair.collapsed(mail, find)
hits = fair.index_hits(usable, world.names)[103]
print(world.names[103], world.roles[103], world.timeline[2][103])
for hit in sorted(hits, key=lambda item: (item["time"], item.get("seq", 0))):
    print(hit["issuer"], hit["value"], hit["time"], hit.get("seq"))
