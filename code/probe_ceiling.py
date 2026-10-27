"""Ceiling: how often the gold string is a span inside the margin passage."""

import json
import math
import re
from collections import Counter
from pathlib import Path

import run_hoh as hoh

rows = json.loads(Path("data/hoh_dev.json").read_text(encoding="utf-8"))


def norm(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower()).strip(" .")


def spans(text):
    found = set(re.findall(r'"([^"]+)"', text))
    for match in re.finditer(
        r"((?:more than |over |about |around |at least )?\d[\d,]*(?:\s*-\s*\d[\d,]*)?)",
        text,
        re.I,
    ):
        found.add(match.group(1).strip())
    for match in re.finditer(r"\b([A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+){0,6})", text):
        found.add(match.group(1))
    return [item for item in found if item and len(item.split()) <= 14]


docs = []
for qid, row in enumerate(rows):
    docs.append(
        {"qid": qid, "text": row["evidence"], "answer": row["answer"], "time": row["time"], "current": True}
    )
    for old in row["outdated"]:
        docs.append(
            {
                "qid": qid,
                "text": old["evidence"],
                "answer": old["answer"],
                "time": old["time"],
                "current": False,
            }
        )
tfs = []
df = Counter()
for doc in docs:
    counts = Counter(hoh.tokens(doc["text"]))
    tfs.append(counts)
    df.update(counts)
n = len(docs)
avg = sum(sum(counts.values()) for counts in tfs) / n
idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}
label = 0
ceiling = 0
for qid, row in enumerate(rows):
    scores = hoh.score_query(row["question"], tfs, idf, avg)
    ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)[:5]
    cands = []
    for index in ranked:
        doc = docs[index]
        cands.append(
            {
                "score": scores[index],
                "time": doc["time"],
                "text": doc["text"],
                "answer": doc["answer"],
                "current": doc["current"] and doc["qid"] == qid,
            }
        )
    picked = hoh.choose(cands, "margin", 0.85)
    gold = norm(row["answer"])
    label += int(norm(picked["answer"]) == gold)
    ceiling += int(any(norm(item) == gold for item in spans(picked["text"])))
print("n", len(rows))
print("margin label", round(label / len(rows), 3))
print("span ceiling", round(ceiling / len(rows), 3))
