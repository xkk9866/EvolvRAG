"""Passage selection on a fixed prefix of the public HoH train split.

The society operator is not run here. This script asks a narrower question:
when a retrieved set mixes a current passage with outdated ones, does a
predeclared lexical margin change which answer is selected?

Metric is exact match between the selected passage's annotated answer and the
current HoH answer. It is not a generation score. The margin 0.85 is fixed
before the rows are scored. Neighboring margins are reported as a sensitivity
check, not as a second chance to pick a winner.
"""

from __future__ import annotations

import json
import math
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "hoh.json"
CACHE = ROOT / "data" / "hoh_rows.json"
DATASET = "russwest404/HoH-QAs"
TARGET = 1000
PAGE = 100
MARGIN = 0.85
NEIGHBORS = (0.70, 0.85, 1.00)
TOPK = 5
K1 = 1.5
B = 0.75


def fetch_rows():
    if CACHE.exists():
        cached = json.loads(CACHE.read_text(encoding="utf-8"))
        if len(cached) >= TARGET:
            return cached[:TARGET]
    rows = []
    offset = 0
    while len(rows) < TARGET:
        url = (
            "https://datasets-server.huggingface.co/rows"
            f"?dataset={DATASET}&config=default&split=train&offset={offset}&length={PAGE}"
        )
        with urllib.request.urlopen(url, timeout=120) as handle:
            payload = json.loads(handle.read().decode("utf-8"))
        batch = payload.get("rows") or []
        if not batch:
            break
        for item in batch:
            row = item.get("row", item)
            outdated = row.get("outdated_infos") or []
            if not outdated:
                continue
            rows.append(
                {
                    "question": row.get("question") or "",
                    "answer": row.get("answer") or "",
                    "evidence": row.get("evidence") or "",
                    "time": str(row.get("last_modified_time") or ""),
                    "outdated": [
                        {
                            "answer": old.get("answer") or "",
                            "evidence": old.get("evidence") or "",
                            "time": str(old.get("last_modified_time") or ""),
                        }
                        for old in outdated
                        if old.get("evidence")
                    ],
                }
            )
            if len(rows) >= TARGET:
                break
        offset += PAGE
        if offset > 5000:
            break
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def build(rows):
    docs = []
    for qid, row in enumerate(rows):
        docs.append(
            {
                "qid": qid,
                "text": row["evidence"],
                "answer": row["answer"],
                "time": row["time"],
                "current": True,
            }
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
    df = Counter()
    tfs = []
    for doc in docs:
        counts = Counter(tokens(doc["text"]))
        tfs.append(counts)
        df.update(counts.keys())
    n = len(docs)
    avg = sum(sum(counts.values()) for counts in tfs) / max(n, 1)
    idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}
    return docs, tfs, idf, avg


def score_query(query, tfs, idf, avg):
    qcounts = Counter(tokens(query))
    scores = []
    for counts in tfs:
        length = sum(counts.values()) or 1
        norm = K1 * (1 - B + B * length / avg)
        total = 0.0
        for term, qf in qcounts.items():
            if term not in counts:
                continue
            tf = counts[term]
            total += idf.get(term, 0.0) * (tf * (K1 + 1)) / (tf + norm) * (qf * (K1 + 1)) / (qf + K1)
        scores.append(total)
    return scores


def norm_answer(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def choose(cands, mode, margin):
    if mode == "bm25":
        return max(cands, key=lambda item: (item["score"], item["time"]))
    if mode == "recency":
        return max(cands, key=lambda item: (item["time"], item["score"]))
    best = max(cands, key=lambda item: (item["score"], item["time"]))
    floor = best["score"] * margin
    newer = [item for item in cands if item["score"] + 1e-9 >= floor and item["time"] >= best["time"]]
    if not newer:
        return best
    return max(newer, key=lambda item: (item["time"], item["score"]))


def evaluate(rows):
    docs, tfs, idf, avg = build(rows)
    methods = ["bm25", "recency"] + [f"margin_{m:.2f}" for m in NEIGHBORS]
    hits = {name: 0 for name in methods}
    recall = 0
    n = len(rows)
    for qid, row in enumerate(rows):
        scores = score_query(row["question"], tfs, idf, avg)
        ranked = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:TOPK]
        cands = []
        for index in ranked:
            doc = docs[index]
            cands.append(
                {
                    "score": scores[index],
                    "time": doc["time"],
                    "answer": doc["answer"],
                    "current": doc["current"] and doc["qid"] == qid,
                    "same": doc["qid"] == qid,
                }
            )
        if any(item["current"] for item in cands):
            recall += 1
        gold = norm_answer(row["answer"])
        picked = {
            "bm25": choose(cands, "bm25", MARGIN),
            "recency": choose(cands, "recency", MARGIN),
        }
        for margin in NEIGHBORS:
            picked[f"margin_{margin:.2f}"] = choose(cands, "margin", margin)
        for name, item in picked.items():
            if norm_answer(item["answer"]) == gold:
                hits[name] += 1
    return {
        "n": n,
        "passages": len(docs),
        "recall_at_5": recall / n,
        "accuracy": {name: hits[name] / n for name in methods},
        "declared_margin": MARGIN,
        "topk": TOPK,
    }


def main():
    rows = fetch_rows()
    report = evaluate(rows)
    report["source"] = DATASET
    report["sample"] = f"first {len(rows)} train rows with a non-empty outdated list"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
