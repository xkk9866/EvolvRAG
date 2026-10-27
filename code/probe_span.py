"""Development-only probe for revision spans. Does not score the official slices.

HoH development rows are the next 400 filtered training rows after the 2000
already used. StreamingQA development rows are the next 400 validation rows
after the official 1000. Grids here choose a rule. They are not test scores.
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import run_hoh as hoh

ROOT = Path(__file__).resolve().parents[1]
HOH_USED = ROOT / "data" / "hoh_rows_2000.json"
HOH_DEV = ROOT / "data" / "hoh_dev.json"
STREAM_DEV = ROOT / "data" / "streamingqa_dev.json"
DEV_N = 400


def norm(text):
    line = ((text or "").strip().splitlines() or [""])[0]
    line = line.strip().strip("\"'`").rstrip(" .")
    return re.sub(r"\s+", " ", line.lower())


def fetch_hoh_dev():
    if HOH_DEV.exists():
        rows = json.loads(HOH_DEV.read_text(encoding="utf-8"))
        if len(rows) >= DEV_N:
            return rows[:DEV_N]
    used = json.loads(HOH_USED.read_text(encoding="utf-8"))
    seen = {(row["question"], row["answer"], row["time"]) for row in used}
    rows = []
    offset = 0
    while len(rows) < DEV_N and offset <= 30000:
        url = (
            "https://datasets-server.huggingface.co/rows"
            f"?dataset={hoh.DATASET}&config=default&split=train&offset={offset}&length=100"
        )
        with urllib.request.urlopen(url, timeout=120) as handle:
            payload = json.loads(handle.read().decode("utf-8"))
        batch = payload.get("rows") or []
        if not batch:
            break
        for item in batch:
            row = item.get("row", item)
            outdated = [old for old in (row.get("outdated_infos") or []) if old.get("evidence")]
            if not outdated:
                continue
            packed = {
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
                ],
            }
            key = (packed["question"], packed["answer"], packed["time"])
            if key in seen:
                continue
            rows.append(packed)
            if len(rows) >= DEV_N:
                break
        offset += 100
        if offset % 1000 == 0:
            print("hoh offset", offset, "dev", len(rows), flush=True)
    HOH_DEV.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


def fetch_stream_dev():
    if STREAM_DEV.exists():
        rows = json.loads(STREAM_DEV.read_text(encoding="utf-8"))
        if len(rows) >= DEV_N:
            return rows[:DEV_N]
    official = json.loads((ROOT / "data" / "streamingqa_valid.json").read_text(encoding="utf-8"))
    seen = {(row["question"], row["answers"][0]) for row in official}
    rows = []
    offset = 0
    while len(rows) < DEV_N and offset <= 20000:
        url = (
            "https://datasets-server.huggingface.co/rows"
            "?dataset=Catkamakura/ts-qa&config=default&split=validation"
            f"&offset={offset}&length=100"
        )
        with urllib.request.urlopen(url, timeout=120) as handle:
            payload = json.loads(handle.read().decode("utf-8"))
        batch = payload.get("rows") or []
        if not batch:
            break
        for item in batch:
            row = item.get("row", item)
            if row.get("source") != "streamingqa" or row.get("is_hard_negative"):
                continue
            answers = [text for text in (row.get("answers") or []) if text]
            context = row.get("context") or ""
            if not answers or not context:
                continue
            if (row.get("question") or "", answers[0]) in seen:
                continue
            rows.append(
                {
                    "question": row.get("question") or "",
                    "answers": answers,
                    "context": context,
                    "time": str(row.get("evidence_timestamp") or ""),
                    "asked": str(row.get("question_timestamp") or ""),
                }
            )
            if len(rows) >= DEV_N:
                break
        offset += 100
    STREAM_DEV.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


def replaced_span(earlier, later):
    left = earlier.split()
    right = later.split()
    if not left or not right:
        return ""
    spans = []
    for tag, _i1, _i2, j1, j2 in SequenceMatcher(a=left, b=right).get_opcodes():
        if tag in ("replace", "insert") and j2 > j1:
            spans.append(" ".join(right[j1:j2]))
    if not spans:
        return ""
    return max(spans, key=len)


def quoted_revision(earlier, later):
    previous = {norm(piece) for piece in re.findall(r"\"([^\"]+)\"", earlier)}
    pieces = re.findall(r"\"([^\"]+)\"", later)
    fresh = [piece for piece in pieces if norm(piece) not in previous and norm(piece)]
    if not fresh:
        return ""
    return max(fresh, key=len)


def build_docs(rows):
    docs = []
    questions = []
    for qid, row in enumerate(rows):
        gold = len(docs)
        docs.append({"text": row["evidence"], "time": row["time"], "qid": qid})
        for old in row["outdated"]:
            docs.append({"text": old["evidence"], "time": old["time"], "qid": qid})
        questions.append({"question": row["question"], "answer": row["answer"], "gold": gold})
    return questions, docs


def bm25_order(question, docs):
    tfs = []
    df = Counter()
    for doc in docs:
        counts = Counter(hoh.tokens(doc["text"]))
        tfs.append(counts)
        df.update(counts)
    n = len(docs)
    avg = sum(sum(counts.values()) for counts in tfs) / max(n, 1)
    idf = {term: __import__("math").log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}
    scores = hoh.score_query(question, tfs, idf, avg)
    return sorted(range(n), key=lambda i: scores[i], reverse=True)


def span_predict(docs, order, k, ratio, kind):
    pool = order[:k]
    best = None
    for a in range(len(pool)):
        for b in range(a + 1, len(pool)):
            left, right = docs[pool[a]], docs[pool[b]]
            if left["time"] == right["time"]:
                earlier, later = left, right
            elif left["time"] < right["time"]:
                earlier, later = left, right
            else:
                earlier, later = right, left
            overlap = SequenceMatcher(a=earlier["text"], b=later["text"]).ratio()
            if overlap < ratio:
                continue
            span = replaced_span(earlier["text"], later["text"]) if kind == "diff" else quoted_revision(earlier["text"], later["text"])
            if not span:
                continue
            rank = (later["time"], overlap)
            if best is None or rank > best[0]:
                best = (rank, span)
    return "" if best is None else best[1]


def score_hoh(rows):
    questions, docs = build_docs(rows)
    orders = [bm25_order(row["question"], docs) for row in questions]
    print("hoh dev", len(rows), "docs", len(docs), flush=True)
    for kind in ("diff", "quote"):
        for k in (5, 20):
            for ratio in (0.45, 0.70):
                hits = 0
                blank = 0
                for row, order in zip(questions, orders):
                    pred = span_predict(docs, order, k, ratio, kind)
                    if not pred:
                        blank += 1
                        continue
                    hits += int(norm(pred) == norm(row["answer"]))
                print(kind, "k", k, "ratio", ratio, "em", round(hits / len(rows), 3), "blank", round(blank / len(rows), 3), flush=True)


def sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part for part in parts if part.strip()]


def score_stream(rows):
    docs = [{"text": row["context"], "time": row["time"]} for row in rows]
    print("stream dev", len(rows), flush=True)
    contain = 0
    for row_i, row in enumerate(rows):
        order = bm25_order(row["question"], docs)[:5]
        pool = []
        for index in order:
            pool.extend(sentences(docs[index]["text"]))
        if not pool:
            continue
        sent_docs = [{"text": sent, "time": ""} for sent in pool]
        top = sentences_rank(row["question"], sent_docs)
        gold = norm(row["answers"][0])
        if gold and gold in norm(top):
            contain += 1
    print("top sentence contains gold", round(contain / len(rows), 3), flush=True)


def sentences_rank(question, docs):
    if len(docs) == 1:
        return docs[0]["text"]
    order = bm25_order(question, docs)
    return docs[order[0]]["text"]


def main():
    hoh_rows = fetch_hoh_dev()
    score_hoh(hoh_rows)
    stream_rows = fetch_stream_dev()
    score_stream(stream_rows)


if __name__ == "__main__":
    main()
