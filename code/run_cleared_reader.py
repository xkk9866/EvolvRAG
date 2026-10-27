"""Reader on the single passage that clears the time margin.

HoH development rows chose this over the top five and over a revised pair.
The margin stays 0.85. StreamingQA uses the latest retrieved article whose
time is not after the question date; that choice is confirmed on development
rows before the official 1000 are scored.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import run_hoh as hoh
import run_reader as reader
from run_public_rag import (
    PublicReader,
    hoh_prompt,
    load_hoh,
    load_streaming,
    news_prompt,
    norm_answer,
    paired,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "cleared_reader.json"
STREAM_DEV = ROOT / "data" / "streamingqa_dev.json"


def index_hoh(rows):
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
    from collections import Counter
    import math

    tfs = []
    df = Counter()
    for doc in docs:
        counts = Counter(hoh.tokens(doc["text"]))
        tfs.append(counts)
        df.update(counts)
    n = len(docs)
    avg = sum(sum(counts.values()) for counts in tfs) / max(n, 1)
    idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}
    return docs, tfs, idf, avg


def hoh_views(rows):
    docs, tfs, idf, avg = index_hoh(rows)
    margin_prompts = []
    top_prompts = []
    golds = []
    outdated = []
    for qid, row in enumerate(rows):
        scores = hoh.score_query(row["question"], tfs, idf, avg)
        ranked = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:5]
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
        picked = hoh.choose(cands, "margin", hoh.MARGIN)
        margin_prompts.append(hoh_prompt(row["question"], [{"time": picked["time"], "text": picked["text"]}]))
        top_prompts.append(hoh_prompt(row["question"], [{"time": item["time"], "text": item["text"]} for item in cands]))
        golds.append([norm_answer(row["answer"])])
        outdated.append({norm_answer(old["answer"]) for old in row["outdated"]})
    return margin_prompts, top_prompts, golds, outdated


def stream_views(rows):
    from collections import Counter
    import math

    docs = [{"text": row["context"], "time": row["time"], "asked": row["asked"]} for row in rows]
    tfs = []
    df = Counter()
    for doc in docs:
        counts = Counter(hoh.tokens(doc["text"]))
        tfs.append(counts)
        df.update(counts)
    n = len(docs)
    avg = sum(sum(counts.values()) for counts in tfs) / max(n, 1)
    idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}
    cleared = []
    top = []
    golds = []
    for row_i, row in enumerate(rows):
        scores = hoh.score_query(row["question"], tfs, idf, avg)
        ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)[:5]
        cands = [{"score": scores[i], "time": docs[i]["time"], "text": docs[i]["text"]} for i in ranked]
        asked = row["asked"]
        eligible = [item for item in cands if item["time"] <= asked] or cands
        picked = max(eligible, key=lambda item: (item["time"], item["score"]))
        cleared.append(news_prompt(row["question"], [{"time": picked["time"], "text": picked["text"]}]))
        top.append(news_prompt(row["question"], cands))
        golds.append([norm_answer(answer) for answer in row["answers"]])
    return cleared, top, golds


def exact(texts, golds, outdated=None):
    em = []
    stale = []
    for i, text in enumerate(texts):
        pred = norm_answer(text)
        hit = int(pred in golds[i] and pred != "")
        em.append(hit)
        if outdated is not None:
            stale.append(int(pred in outdated[i] and not hit))
    return em, stale


def summarize(em, stale=None):
    packed = {"em": sum(em) / len(em), "n": len(em)}
    if stale:
        packed["stale"] = sum(stale) / len(stale)
    return packed


def run_pair(llm, left_prompts, right_prompts, golds, outdated=None):
    left = llm.complete(left_prompts)
    right = llm.complete(right_prompts)
    left_em, left_stale = exact(left, golds, outdated)
    right_em, right_stale = exact(right, golds, outdated)
    return {
        "cleared": summarize(left_em, left_stale),
        "top5": summarize(right_em, right_stale),
        "paired_em": paired(left_em, right_em),
    }


def main():
    started = time.time()
    device = reader.device_of()
    print("device", device, flush=True)
    llm = PublicReader(device)
    payload = {"slices": {}}
    if OUT.exists():
        payload = json.loads(OUT.read_text(encoding="utf-8"))
        payload.setdefault("slices", {})
    for name, which in (("hoh_first", "first"), ("hoh_next", "next")):
        if name in payload["slices"]:
            print("skip", name, flush=True)
            continue
        cache = json.loads(
            (ROOT / "data" / ("hoh_rows.json" if which == "first" else "hoh_rows_2000.json")).read_text(encoding="utf-8")
        )
        rows = cache if which == "first" else cache[1000:2000]
        print("slice", name, len(rows), flush=True)
        margin, top, golds, outdated = hoh_views(rows)
        payload["slices"][name] = run_pair(llm, margin, top, golds, outdated)
        print(json.dumps(payload["slices"][name], indent=2), flush=True)
        OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    dev_rows = json.loads(STREAM_DEV.read_text(encoding="utf-8"))[:40]
    dev_cleared, dev_top, dev_golds = stream_views(dev_rows)
    dev = run_pair(llm, dev_cleared, dev_top, dev_golds)
    payload["streaming_dev40"] = dev
    print("stream dev", json.dumps(dev), flush=True)
    if dev["paired_em"]["mean"] > 0:
        official_rows = json.loads((ROOT / "data" / "streamingqa_valid.json").read_text(encoding="utf-8"))[:1000]
        print("slice streamingqa", len(official_rows), flush=True)
        cleared, top, golds = stream_views(official_rows)
        payload["slices"]["streamingqa"] = run_pair(llm, cleared, top, golds)
        print(json.dumps(payload["slices"]["streamingqa"], indent=2), flush=True)
    payload["seconds"] = time.time() - started
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
