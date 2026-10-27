"""Retriever plus a generative reader on two public continual-QA corpora.

HoH slices are the same first 1000 and next 1000 training rows already used
for lexical passage selection. StreamingQA is the first 1000 validation rows
of the public ts-qa release whose source field is streamingqa and whose
hard-negative flag is false, read in dataset order.

Locked before either corpus is scored:
  embedding: BAAI/bge-small-en-v1.5, CLS, L2 norm, 128 tokens, query prefix
  reader: Qwen/Qwen2.5-3B-Instruct, greedy, 48 new tokens
  top 5 passages, each capped at 8000 characters
  BM25 ties keep the earlier index, matching the HoH lexical check
  dense ties keep the later time, then the earlier index
  exact match is case-folded, whitespace-collapsed, and stripped of one
  trailing period; StreamingQA matches any listed answer
  token F1 is the best overlap with a listed answer
  evidence hallucination is a non-empty generation that is not contained in
  a passage shown to the reader; a closed-book reply is not a generation
  nothing here retunes clearance, the floor, or the society reader
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as rx
import run_fair as fair
import run_hoh as hoh
import run_reader as reader

ROOT = rx.ROOT
OUT = ROOT / "results" / "public_rag.json"
STREAM_CACHE = ROOT / "data" / "streamingqa_valid.json"
HOH_FIRST = ROOT / "data" / "hoh_rows.json"
HOH_BOTH = ROOT / "data" / "hoh_rows_2000.json"
STREAM_DATASET = "Catkamakura/ts-qa"
STREAM_N = 1000
READ = 5
PASSAGE_CAP = 8000
NEW_TOKENS = 48
SYSTEMS = ("closed_llm", "bm25_llm", "dense_llm", "oracle_llm")


def norm_answer(text):
    line = ((text or "").strip().splitlines() or [""])[0]
    line = line.strip().strip("\"'`").rstrip(" .")
    return re.sub(r"\s+", " ", line.lower())


def token_f1(pred, golds):
    left = hoh.tokens(pred)
    if not left:
        return 0.0
    best = 0.0
    counted = Counter(left)
    for gold in golds:
        right = hoh.tokens(gold)
        if not right:
            continue
        overlap = sum((counted & Counter(right)).values())
        if overlap == 0:
            continue
        precision = overlap / len(left)
        recall = overlap / len(right)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def fold(text):
    return re.sub(r"\s+", " ", (text or "").lower())


def contained(pred, passages):
    folded = norm_answer(pred)
    if not folded:
        return False
    for doc in passages:
        if folded in fold(doc["text"][:PASSAGE_CAP]):
            return True
    return False


class PublicReader(reader.LLMReader):
    def complete(self, prompts):
        outs = [""] * len(prompts)
        if not prompts:
            return outs
        texts = [
            self.tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        for start in range(0, len(texts), 16):
            chunk = texts[start : start + 16]
            batch = self.tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=8192).to(self.device)
            with torch.inference_mode():
                gen = self.model.generate(
                    **batch,
                    max_new_tokens=NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=self.tok.pad_token_id,
                )
            new = gen[:, batch["input_ids"].shape[1] :]
            decoded = self.tok.batch_decode(new, skip_special_tokens=True)
            for offset, text in enumerate(decoded):
                outs[start + offset] = text
        return outs


def hoh_prompt(question, passages):
    lines = [f"[{i}] time {doc['time']}: {doc['text'][:PASSAGE_CAP]}" for i, doc in enumerate(passages, 1)]
    return (
        "Answer the question using only the passages. "
        "If they disagree, use the passage with the later time. "
        "Reply with the answer and no other words.\n\n"
        f"Question: {question}\n\nPassages:\n" + "\n".join(lines) + "\n\nAnswer:"
    )


def news_prompt(question, passages):
    lines = [f"[{i}] time {doc['time']}: {doc['text'][:PASSAGE_CAP]}" for i, doc in enumerate(passages, 1)]
    return (
        "Answer the question as of the date it states, using only the passages. "
        "Reply with the answer and no other words.\n\n"
        f"Question: {question}\n\nPassages:\n" + "\n".join(lines) + "\n\nAnswer:"
    )


def closed_prompt(question):
    return (
        "Answer the question from memory. Reply with the answer and no other words.\n\n"
        f"Question: {question}\n\nAnswer:"
    )


def load_hoh(which):
    first = json.loads(HOH_FIRST.read_text(encoding="utf-8"))
    both = json.loads(HOH_BOTH.read_text(encoding="utf-8"))
    if both[:1000] != first:
        raise RuntimeError("HoH cache prefix does not match the lexical slice")
    rows = first if which == "first" else both[1000:2000]
    docs = []
    questions = []
    for qid, row in enumerate(rows):
        gold = len(docs)
        docs.append({"text": row["evidence"], "time": row["time"], "qid": qid})
        for old in row["outdated"]:
            docs.append({"text": old["evidence"], "time": old["time"], "qid": qid})
        questions.append(
            {
                "question": row["question"],
                "answers": [row["answer"]],
                "outdated": [old["answer"] for old in row["outdated"]],
                "gold": gold,
            }
        )
    return questions, docs


def fetch_streaming():
    if STREAM_CACHE.exists():
        cached = json.loads(STREAM_CACHE.read_text(encoding="utf-8"))
        if len(cached) >= STREAM_N:
            return cached[:STREAM_N]
    rows = []
    offset = 0
    while len(rows) < STREAM_N:
        url = (
            "https://datasets-server.huggingface.co/rows"
            f"?dataset={STREAM_DATASET}&config=default&split=validation&offset={offset}&length=100"
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
            rows.append(
                {
                    "question": row.get("question") or "",
                    "answers": answers,
                    "context": context,
                    "time": str(row.get("evidence_timestamp") or ""),
                    "asked": str(row.get("question_timestamp") or ""),
                }
            )
            if len(rows) >= STREAM_N:
                break
        offset += 100
        if offset > 20000:
            break
    if len(rows) < STREAM_N:
        raise RuntimeError(f"streamingqa validation returned {len(rows)} rows")
    STREAM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    STREAM_CACHE.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


def load_streaming():
    rows = fetch_streaming()
    docs = []
    questions = []
    for qid, row in enumerate(rows):
        docs.append({"text": row["context"], "time": row["time"], "qid": qid})
        questions.append(
            {
                "question": row["question"],
                "answers": row["answers"],
                "outdated": [],
                "gold": qid,
            }
        )
    return questions, docs


def bm25_top(questions, docs):
    tfs = []
    df = Counter()
    for doc in docs:
        counts = Counter(hoh.tokens(doc["text"]))
        tfs.append(counts)
        df.update(counts.keys())
    n = len(docs)
    avg = sum(sum(counts.values()) for counts in tfs) / max(n, 1)
    idf = {term: __import__("math").log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}
    picked = []
    for row in questions:
        scores = hoh.score_query(row["question"], tfs, idf, avg)
        order = sorted(range(n), key=lambda i: scores[i], reverse=True)[:READ]
        picked.append([docs[i] for i in order])
        row["bm25_hit"] = int(row["gold"] in order)
    return picked


def dense_top(questions, docs, embedder):
    matrix = embedder.encode([doc["text"] for doc in docs], query=False)
    queries = embedder.encode([row["question"] for row in questions], query=True)
    picked = []
    for row_i, row in enumerate(questions):
        scores = (matrix @ queries[row_i]).tolist()
        order = list(range(len(docs)))
        order.sort(key=lambda i: docs[i]["time"], reverse=True)
        order.sort(key=lambda i: scores[i], reverse=True)
        top = order[:READ]
        picked.append([docs[i] for i in top])
        row["dense_hit"] = int(row["gold"] in top)
    return picked


def score_text(text, row, passages, generated):
    pred = norm_answer(text)
    golds = [norm_answer(answer) for answer in row["answers"]]
    em = int(pred in golds and pred != "")
    stale = int(pred in {norm_answer(answer) for answer in row["outdated"]} and not em)
    hall = None
    if generated:
        hall = int(not contained(pred, passages))
    return {"em": em, "f1": token_f1(pred, row["answers"]), "stale": stale, "hall": hall}


def paired(left, right):
    diffs = [a - b for a, b in zip(left, right)]
    mu, sd = rx.mean_std(diffs)
    return {"mean": mu, "std": sd, "p": fair.perm_p(diffs), "n": len(diffs)}


def pack(rows, key):
    values = [row[key] for row in rows]
    mu, sd = rx.mean_std(values)
    return {"mean": mu, "std": sd}


def run_slice(name, kind, questions, docs, embedder, llm):
    prompt_of = hoh_prompt if kind == "hoh" else news_prompt
    bm25 = bm25_top(questions, docs)
    dense = dense_top(questions, docs, embedder)
    oracle = [[docs[row["gold"]]] for row in questions]
    prompts = {system: [] for system in SYSTEMS}
    shown = {"bm25_llm": bm25, "dense_llm": dense, "oracle_llm": oracle}
    for row, bm, de, gold in zip(questions, bm25, dense, oracle):
        prompts["closed_llm"].append(closed_prompt(row["question"]))
        prompts["bm25_llm"].append(prompt_of(row["question"], bm))
        prompts["dense_llm"].append(prompt_of(row["question"], de))
        prompts["oracle_llm"].append(prompt_of(row["question"], gold))
    outputs = {system: llm.complete(prompts[system]) for system in SYSTEMS}
    stored = {system: [] for system in SYSTEMS}
    for i, row in enumerate(questions):
        stored["closed_llm"].append(score_text(outputs["closed_llm"][i], row, [], False))
        for system in ("bm25_llm", "dense_llm", "oracle_llm"):
            stored[system].append(score_text(outputs[system][i], row, shown[system][i], True))
    summary = {}
    for system, rows in stored.items():
        summary[system] = {
            "em": pack(rows, "em")["mean"],
            "f1": pack(rows, "f1")["mean"],
            "stale": pack(rows, "stale")["mean"],
        }
        halls = [item["hall"] for item in rows if item["hall"] is not None]
        summary[system]["hall"] = None if not halls else sum(halls) / len(halls)
    summary["bm25_llm"]["recall5"] = sum(row["bm25_hit"] for row in questions) / len(questions)
    summary["dense_llm"]["recall5"] = sum(row["dense_hit"] for row in questions) / len(questions)
    tests = []
    for metric in ("em", "f1"):
        for other in ("bm25_llm", "oracle_llm", "closed_llm"):
            tests.append(
                {
                    "metric": metric,
                    "other": other,
                    **paired([row[metric] for row in stored["dense_llm"]], [row[metric] for row in stored[other]]),
                }
            )
    tests.append(
        {
            "metric": "hall",
            "other": "bm25_llm",
            **paired(
                [row["hall"] for row in stored["dense_llm"]],
                [row["hall"] for row in stored["bm25_llm"]],
            ),
        }
    )
    return {"name": name, "n": len(questions), "passages": len(docs), "summary": summary, "tests": tests}


def main():
    started = time.time()
    smoke = "--smoke" in sys.argv
    device = reader.device_of()
    print("device", device, flush=True)
    embedder = reader.Embedder(device)
    llm = PublicReader(device)
    jobs = [("hoh_first", "hoh", load_hoh("first")), ("hoh_next", "hoh", load_hoh("next")), ("streamingqa", "news", load_streaming())]
    payload = {
        "protocol": {
            "embedding": reader.EMB_NAME,
            "reader": reader.LLM_NAME,
            "new_tokens": NEW_TOKENS,
            "read": READ,
            "passage_cap": PASSAGE_CAP,
            "context_cap": 8192,
            "hoh": "same first 1000 and next 1000 rows as the lexical check",
            "streamingqa": f"first {STREAM_N} validation rows of {STREAM_DATASET} with source streamingqa and hard-negative false",
            "clearance": "not applied; these corpora do not name issuers or organizations",
        },
        "slices": {},
    }
    if OUT.exists() and not smoke:
        previous = json.loads(OUT.read_text(encoding="utf-8"))
        payload["slices"] = previous.get("slices") or {}
    for name, kind, (questions, docs) in jobs:
        if name in payload["slices"] and not smoke:
            print("skip", name, flush=True)
            continue
        if smoke:
            questions = questions[:2]
            docs = [doc for doc in docs if doc["qid"] < 2]
            cursor = 0
            for qid, row in enumerate(questions):
                row["gold"] = cursor
                cursor += sum(doc["qid"] == qid for doc in docs)
        print("slice", name, "n", len(questions), "docs", len(docs), flush=True)
        payload["slices"][name] = run_slice(name, kind, questions, docs, embedder, llm)
        print(json.dumps(payload["slices"][name]["summary"], indent=2), flush=True)
        if smoke:
            break
        OUT.parent.mkdir(parents=True, exist_ok=True)
        payload["seconds"] = time.time() - started
        OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not smoke:
        payload["seconds"] = time.time() - started
        OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("wrote", OUT, "seconds", round(payload["seconds"], 1), flush=True)


if __name__ == "__main__":
    main()
