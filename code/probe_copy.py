"""Choose a copy-from-revised-passage reader on HoH development rows only.

The official first 2000 filtered rows are not scored. The margin 0.85 and the
top-5 pool match the lexical check. The prompt is fixed before the run.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import run_hoh as hoh
import run_reader as reader
from run_public_rag import PublicReader, norm_answer

ROOT = Path(__file__).resolve().parents[1]
DEV = json.loads((ROOT / "data" / "hoh_dev.json").read_text(encoding="utf-8"))
N = 120
MARGIN = 0.85


def build(rows):
    docs = []
    for qid, row in enumerate(rows):
        docs.append({"qid": qid, "text": row["evidence"], "answer": row["answer"], "time": row["time"], "current": True})
        for old in row["outdated"]:
            docs.append({"qid": qid, "text": old["evidence"], "answer": old["answer"], "time": old["time"], "current": False})
    tfs = []
    from collections import Counter
    import math
    df = Counter()
    for doc in docs:
        counts = Counter(hoh.tokens(doc["text"]))
        tfs.append(counts)
        df.update(counts)
    n = len(docs)
    avg = sum(sum(counts.values()) for counts in tfs) / max(n, 1)
    idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}
    return docs, tfs, idf, avg


def top_passages(row, qid, docs, tfs, idf, avg):
    scores = hoh.score_query(row["question"], tfs, idf, avg)
    ranked = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:5]
    cands = []
    for index in ranked:
        doc = docs[index]
        cands.append({"score": scores[index], "time": doc["time"], "text": doc["text"], "answer": doc["answer"], "current": doc["current"] and doc["qid"] == qid})
    return cands


def copy_prompt(question, text):
    return (
        "Copy the shortest span from the passage that answers the question. "
        "Reply with that span and no other words.\n\n"
        f"Question: {question}\n\nPassage: {text}\n\nAnswer:"
    )


def later_of_pair(cands):
    best = None
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            a, b = cands[i], cands[j]
            overlap = SequenceMatcher(a=a["text"], b=b["text"]).ratio()
            if overlap < 0.45:
                continue
            later = a if a["time"] >= b["time"] else b
            rank = (later["time"], overlap)
            if best is None or rank > best[0]:
                best = (rank, later)
    if best is None:
        return max(cands, key=lambda item: (item["time"], item["score"]))
    return best[1]


def main():
    rows = DEV[:N]
    docs, tfs, idf, avg = build(rows)
    prompts = {"margin": [], "revised": []}
    golds = []
    label_hits = 0
    for qid, row in enumerate(rows):
        cands = top_passages(row, qid, docs, tfs, idf, avg)
        picked = hoh.choose(cands, "margin", MARGIN)
        revised = later_of_pair(cands)
        prompts["margin"].append(copy_prompt(row["question"], picked["text"]))
        prompts["revised"].append(copy_prompt(row["question"], revised["text"]))
        golds.append(norm_answer(row["answer"]))
        label_hits += int(norm_answer(picked["answer"]) == norm_answer(row["answer"]))
    device = reader.device_of()
    print("device", device, "margin label", round(label_hits / N, 3), flush=True)
    llm = PublicReader(device)
    for name, batch in prompts.items():
        outs = llm.complete(batch)
        hits = sum(int(norm_answer(text) == gold) for text, gold in zip(outs, golds))
        print(name, "em", round(hits / N, 3), flush=True)


if __name__ == "__main__":
    main()
