"""Embedding retrieval and an LLM reader on the evolving society corpus.

The corpus grows by epoch. Issuers are not labeled. A question is the role
of a named entity. The retriever does not receive the slot id. The reader
sees only the passages it was given and must answer with one value token.

Locked before seeds 716-731 are scored:
  embedding: BAAI/bge-small-en-v1.5, CLS, L2 norm, query instruction
  reader: Qwen/Qwen2.5-3B-Instruct, greedy, 8 new tokens
  pool 20, reader window 5, clearance rule the already selected margin-1 record
  primary epoch is the last one
  paraphrase is a four-seed stress, not a selection input

QA accuracy is an exact match to the epoch's truth. Freshness is that match
on mutable keys whose truth has left the initial value. A stale answer is an
earlier truth. Evidence hallucination is a generated value that is not the
value of any passage shown to the reader. A carried belief is not a generation.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as rx
import run_fair as fair
import run_rag as rag

ROOT = rx.ROOT
OUT = ROOT / "results" / "reader.json"
EMB_NAME = "BAAI/bge-small-en-v1.5"
LLM_NAME = "Qwen/Qwen2.5-3B-Instruct"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
POOL = 20
READ = 5
SEEDS = list(range(716, 732))
PARA_SEEDS = list(range(716, 720))
REGIMES = ("stationary", "topic", "open_rumor")
SCORE_EPOCHS = {4, 8}
VALUE_RE = re.compile(r"via-v\d+|v\d+|q\d+|EXPIRED", re.IGNORECASE)
CFG = dict(fair.RULES[4])


def device_of():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    best, free = 0, -1
    for i in range(torch.cuda.device_count()):
        unused = torch.cuda.mem_get_info(i)[0]
        if unused > free:
            best, free = i, unused
    return torch.device(f"cuda:{best}")


class Embedder:
    def __init__(self, device):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(EMB_NAME, local_files_only=True)
        self.model = AutoModel.from_pretrained(EMB_NAME, local_files_only=True).to(device)
        self.model.eval()

    def encode(self, texts, query=False):
        if not texts:
            return torch.empty(0, 0)
        if query:
            texts = [QUERY_PREFIX + text for text in texts]
        pieces = []
        for start in range(0, len(texts), 64):
            batch = self.tok(
                texts[start : start + 64],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                hidden = self.model(**batch).last_hidden_state[:, 0]
                hidden = torch.nn.functional.normalize(hidden, p=2, dim=1)
            pieces.append(hidden)
        return torch.cat(pieces, dim=0)


class LLMReader:
    def __init__(self, device):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(LLM_NAME, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_NAME, torch_dtype=torch.float16, local_files_only=True
        ).to(device)
        self.model.eval()
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"

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
            batch = self.tok(chunk, return_tensors="pt", padding=True).to(self.device)
            with torch.inference_mode():
                gen = self.model.generate(
                    **batch,
                    max_new_tokens=8,
                    do_sample=False,
                    pad_token_id=self.tok.pad_token_id,
                )
            new = gen[:, batch["input_ids"].shape[1] :]
            decoded = self.tok.batch_decode(new, skip_special_tokens=True)
            for offset, text in enumerate(decoded):
                outs[start + offset] = text
        return outs


def parse_value(text):
    match = VALUE_RE.search(text or "")
    if match is None:
        return None
    token = match.group(0)
    if token.upper() == "EXPIRED":
        return "EXPIRED"
    return token.lower()


def evidence_prompt(question, passages):
    lines = [f"[{i}] time {doc['time']}: {doc['text']}" for i, doc in enumerate(passages, 1)]
    return (
        "Answer the question using only the passages. "
        "If they disagree about the named entity, use the passage with the later time. "
        "Reply with one value token and no other words.\n\n"
        f"Question: {question}\n\nPassages:\n" + "\n".join(lines) + "\n\nValue:"
    )


def closed_prompt(question):
    return (
        "Reply with one value token for the question. "
        "Value tokens look like v2, q3, EXPIRED, or via-v2.\n\n"
        f"Question: {question}\n\nValue:"
    )


def bm25_topk(index, query, k):
    q = Counter(rag.tokens(query))
    seen = set()
    ranked = []
    for term in q:
        for i in index.postings.get(term, ()):
            if i in seen:
                continue
            seen.add(i)
            score = index._score(q, i)
            if score > 0:
                ranked.append((score, i))
    ranked.sort(key=lambda item: (-item[0], -index.docs[item[1]]["time"], item[1]))
    return [index.docs[i] for _score, i in ranked[:k]]


def dense_topk(q_vec, mat, docs, k):
    if not docs:
        return []
    scores = (mat @ q_vec).tolist()
    order = sorted(range(len(docs)), key=lambda i: (-scores[i], -docs[i]["time"], i))
    return [docs[i] for i in order[:k]]


def recalls(passages, slot, truth, names):
    return int(
        any(fair.entity_of_text(doc["text"], names) == slot and doc["value"] == truth for doc in passages)
    )


def latest(passages):
    if not passages:
        return None
    return max(passages, key=lambda doc: (doc["time"], doc.get("seq", 0)))["value"]


def eligible(passages, trust, cutoff):
    kept = []
    for doc in passages:
        weight = fair.weight_of(trust, doc["issuer"])
        if weight is not None and weight + 1e-9 >= cutoff:
            kept.append(doc)
    kept.sort(key=lambda doc: (doc["time"], doc.get("seq", 0)), reverse=True)
    return kept


def earlier_values(world, slot, epoch_i):
    seen = set()
    for prev in range(epoch_i):
        seen.add(world.timeline[prev][slot])
    return seen


def blank():
    return {
        "qa": 0,
        "fresh": 0,
        "fresh_n": 0,
        "stale": 0,
        "hall": 0,
        "hall_n": 0,
        "calls": 0,
        "n": 0,
        "recall20": 0,
        "recall_n": 0,
        "entity5": 0.0,
        "entity5_n": 0,
    }


def add_case(bucket, answer, truth, fresh, stale, generated, shown, recall=None):
    bucket["n"] += 1
    bucket["qa"] += int(answer == truth)
    if fresh:
        bucket["fresh_n"] += 1
        bucket["fresh"] += int(answer == truth)
        bucket["stale"] += int(stale and answer != truth)
    if generated:
        bucket["calls"] += 1
        bucket["hall_n"] += 1
        parsed_ok = answer in shown
        bucket["hall"] += int(not parsed_ok)
    if recall is not None:
        bucket["recall_n"] += 1
        bucket["recall20"] += int(recall)


def add_entity(bucket, passages, slot, names):
    if not passages:
        return
    bucket["entity5_n"] += 1
    bucket["entity5"] += sum(fair.entity_of_text(doc["text"], names) == slot for doc in passages) / float(len(passages))


def pack(bucket):
    def rate(num, den):
        return None if den == 0 else num / float(den)

    return {
        "qa": rate(bucket["qa"], bucket["n"]),
        "fresh": rate(bucket["fresh"], bucket["fresh_n"]),
        "stale": rate(bucket["stale"], bucket["fresh_n"]),
        "hall": rate(bucket["hall"], bucket["hall_n"]),
        "calls": bucket["calls"],
        "n": bucket["n"],
        "fresh_n": bucket["fresh_n"],
        "recall20": rate(bucket["recall20"], bucket["recall_n"]),
        "entity5": rate(bucket["entity5"], bucket["entity5_n"]),
    }


def run_world(world, embedder, reader, score_epochs):
    names = world.names
    order = list(world.mutable_ids) + list(world.stable_ids) + list(world.derived_ids)
    book = fair.TrustBook(CFG)
    belief = list(world.init_belief[0])
    mail = []
    scored = []
    names_of = (
        "closed_llm",
        "bm25_llm",
        "dense_llm",
        "bm25_newest",
        "dense_newest",
        "dense_routed_llm",
        "dense_routed_extract",
        "bm25_routed_llm",
        "bm25_routed_extract",
        "oracle_routed",
    )
    for epoch_i, epoch_docs in enumerate(world.visible):
        epoch = epoch_i + 1
        fresh_docs = list(epoch_docs)
        mail.extend(fresh_docs)
        find = fair.copy_find(mail, names) if CFG.get("copy") else None
        usable = fair.collapsed(mail, find)
        buckets = fair.index_hits(usable, names)
        if epoch in score_epochs:
            questions = [rx.query_text(world, slot) for slot in order]
            for slot, question in zip(order, questions):
                if fair.entity_of_text(question, names) != slot:
                    raise RuntimeError(f"question did not name slot {slot}")
            index = rag.Index(usable)
            mat = embedder.encode([doc["text"] for doc in usable])
            qmat = embedder.encode(questions, query=True)
            jobs = []
            dense_row = list(belief)
            lex_row = list(belief)
            oracle_row = list(belief)
            for row_i, slot in enumerate(order):
                question = questions[row_i]
                truth = world.timeline[epoch_i][slot]
                trust = book.for_slot(slot)
                dense20 = dense_topk(qmat[row_i], mat, usable, POOL)
                lexical20 = bm25_topk(index, question, POOL)
                kept = eligible(dense20, trust, CFG["cutoff"])[:READ]
                lexical_kept = eligible(lexical20, trust, CFG["cutoff"])[:READ]
                schema_dense = fair.schema_value(world, dense_row, slot) if CFG.get("schema") else None
                schema_lex = fair.schema_value(world, lex_row, slot) if CFG.get("schema") else None
                if kept:
                    extract = kept[0]["value"]
                    call = True
                elif schema_dense is not None:
                    extract = schema_dense
                    call = False
                else:
                    extract = dense_row[slot]
                    call = False
                if lexical_kept:
                    lexical_extract = lexical_kept[0]["value"]
                    lexical_call = True
                elif schema_lex is not None:
                    lexical_extract = schema_lex
                    lexical_call = False
                else:
                    lexical_extract = lex_row[slot]
                    lexical_call = False
                oracle = fair.answer_slot(
                    CFG,
                    buckets.get(slot, []),
                    oracle_row[slot],
                    trust,
                    fair.schema_value(world, oracle_row, slot) if CFG.get("schema") else None,
                )
                dense_row[slot] = extract
                lex_row[slot] = lexical_extract
                oracle_row[slot] = oracle
                jobs.append(
                    {
                        "slot": slot,
                        "question": question,
                        "truth": truth,
                        "fresh": slot in world.mutable_ids and truth != world.truth0[slot],
                        "earlier": earlier_values(world, slot, epoch_i),
                        "dense5": dense20[:READ],
                        "lexical5": lexical20[:READ],
                        "kept": kept,
                        "extract": extract,
                        "call": call,
                        "lexical_kept": lexical_kept,
                        "lexical_extract": lexical_extract,
                        "lexical_call": lexical_call,
                        "dense_recall": recalls(dense20, slot, truth, names),
                        "bm25_recall": recalls(lexical20, slot, truth, names),
                        "oracle": oracle,
                    }
                )
            prompts = []
            owners = []
            for job_i, job in enumerate(jobs):
                prompts.append(closed_prompt(job["question"]))
                owners.append((job_i, "closed"))
                if job["dense5"]:
                    prompts.append(evidence_prompt(job["question"], job["dense5"]))
                    owners.append((job_i, "dense"))
                if job["lexical5"]:
                    prompts.append(evidence_prompt(job["question"], job["lexical5"]))
                    owners.append((job_i, "bm25"))
                if job["call"]:
                    prompts.append(evidence_prompt(job["question"], job["kept"]))
                    owners.append((job_i, "routed"))
                if job["lexical_call"]:
                    prompts.append(evidence_prompt(job["question"], job["lexical_kept"]))
                    owners.append((job_i, "bm25_routed"))
            texts = reader.complete(prompts)
            parsed = {(job_i, kind): parse_value(text) for (job_i, kind), text in zip(owners, texts)}
            systems = {name: blank() for name in names_of}
            for job_i, job in enumerate(jobs):
                truth = job["truth"]
                fresh = job["fresh"]
                earlier = job["earlier"]
                closed = parsed.get((job_i, "closed"))
                add_case(systems["closed_llm"], closed, truth, fresh, closed in earlier, False, set())
                dense_ans = parsed.get((job_i, "dense"))
                lexical_ans = parsed.get((job_i, "bm25"))
                add_case(systems["bm25_llm"], lexical_ans, truth, fresh, lexical_ans in earlier, True, {doc["value"] for doc in job["lexical5"]}, job["bm25_recall"])
                add_case(systems["dense_llm"], dense_ans, truth, fresh, dense_ans in earlier, True, {doc["value"] for doc in job["dense5"]}, job["dense_recall"])
                lexical_new = latest(job["lexical5"])
                dense_new = latest(job["dense5"])
                add_case(systems["bm25_newest"], lexical_new, truth, fresh, lexical_new in earlier, False, set(), job["bm25_recall"])
                add_case(systems["dense_newest"], dense_new, truth, fresh, dense_new in earlier, False, set(), job["dense_recall"])
                routed_answer = parsed.get((job_i, "routed")) if job["call"] else job["extract"]
                add_case(
                    systems["dense_routed_llm"],
                    routed_answer,
                    truth,
                    fresh,
                    routed_answer in earlier,
                    job["call"],
                    {doc["value"] for doc in job["kept"]},
                    job["dense_recall"],
                )
                add_case(systems["dense_routed_extract"], job["extract"], truth, fresh, job["extract"] in earlier, False, set(), job["dense_recall"])
                lexical_routed = parsed.get((job_i, "bm25_routed")) if job["lexical_call"] else job["lexical_extract"]
                add_case(
                    systems["bm25_routed_llm"],
                    lexical_routed,
                    truth,
                    fresh,
                    lexical_routed in earlier,
                    job["lexical_call"],
                    {doc["value"] for doc in job["lexical_kept"]},
                    job["bm25_recall"],
                )
                add_case(systems["bm25_routed_extract"], job["lexical_extract"], truth, fresh, job["lexical_extract"] in earlier, False, set(), job["bm25_recall"])
                add_case(systems["oracle_routed"], job["oracle"], truth, fresh, job["oracle"] in earlier, False, set(), None)
                add_entity(systems["dense_llm"], job["dense5"], job["slot"], names)
                add_entity(systems["bm25_llm"], job["lexical5"], job["slot"], names)
            scored.append({"epoch": epoch, "systems": {name: pack(bucket) for name, bucket in systems.items()}})
        row = list(belief)
        for slot in order:
            schema_v = fair.schema_value(world, row, slot) if CFG.get("schema") else None
            row[slot] = fair.answer_slot(CFG, buckets.get(slot, []), row[slot], book.for_slot(slot), schema_v)
        belief = row
        book.update(fair.collapsed(fresh_docs, find), usable, names)
    return scored


def mean_rows(rows):
    systems = rows[0]["systems"].keys()
    out = {}
    for name in systems:
        acc = {key: [] for key in ("qa", "fresh", "stale", "hall", "recall20", "entity5")}
        for row in rows:
            cell = row["systems"][name]
            for key in acc:
                if cell[key] is not None:
                    acc[key].append(cell[key])
        packed = {}
        for key, values in acc.items():
            mu, sd = rx.mean_std(values) if values else (None, None)
            packed[key] = mu
            packed[key + "_std"] = sd
            packed[key + "_n"] = len(values)
        out[name] = packed
    return out


def paired(rows, left, right, key):
    diffs = []
    for row in rows:
        a = row["systems"][left][key]
        b = row["systems"][right][key]
        if a is None or b is None:
            continue
        diffs.append(a - b)
    if not diffs:
        return None
    mu, sd = rx.mean_std(diffs)
    p = rx.signflip_p(diffs) if len(diffs) <= 16 else None
    return {"mean": mu, "std": sd, "p": p, "n": len(diffs)}


def evaluate(embedder, reader, jobs, label):
    stored = []
    for regime, seed, paraphrase, epochs in jobs:
        world = fair.load_world(seed, "stationary" if paraphrase else regime, paraphrase=paraphrase)
        scored = run_world(world, embedder, reader, epochs)
        for item in scored:
            stored.append({"regime": regime, "seed": seed, "epoch": item["epoch"], "systems": item["systems"]})
            OUT.write_text(json.dumps({"partial": stored}, indent=2), encoding="utf-8")
            cell = item["systems"]["dense_routed_llm"]
            print(
                label,
                regime,
                seed,
                "e" + str(item["epoch"]),
                "qa",
                round(cell["qa"], 3),
                "fresh",
                None if cell["fresh"] is None else round(cell["fresh"], 3),
                "hall",
                None if cell["hall"] is None else round(cell["hall"], 3),
                "recall",
                None if cell["recall20"] is None else round(cell["recall20"], 3),
                flush=True,
            )
    return stored


def summarize(stored):
    summary = {}
    tests = []
    groups = {}
    for row in stored:
        groups.setdefault((row["regime"], row["epoch"]), []).append(row)
    for (regime, epoch), rows in groups.items():
        summary[f"{regime}-e{epoch}"] = mean_rows(rows)
        if epoch != 8:
            continue
        for key in ("qa", "fresh", "hall"):
            for other in ("dense_llm", "bm25_llm", "bm25_routed_llm", "dense_newest", "dense_routed_extract", "bm25_routed_extract", "oracle_routed", "closed_llm"):
                if key == "hall" and other in ("dense_newest", "dense_routed_extract", "oracle_routed", "closed_llm"):
                    continue
                stat = paired(rows, "dense_routed_llm", other, key)
                if stat:
                    tests.append({"regime": regime, "epoch": epoch, "metric": key, "other": other, **stat})
    return summary, tests


def main():
    started = time.time()
    device = device_of()
    print("device", device, flush=True)
    embedder = Embedder(device)
    reader = LLMReader(device)
    if "--smoke" in sys.argv:
        world = fair.load_world(399, "stationary")
        scored = run_world(world, embedder, reader, {8})
        print(json.dumps(scored[-1]["systems"], indent=2), flush=True)
        return
    jobs = [(regime, seed, False, SCORE_EPOCHS) for regime in REGIMES for seed in SEEDS]
    jobs += [("paraphrase", seed, True, {8}) for seed in PARA_SEEDS]
    stored = evaluate(embedder, reader, jobs, "reader")
    summary, tests = summarize(stored)
    payload = {
        "protocol": {
            "seeds": SEEDS,
            "paraphrase_seeds": PARA_SEEDS,
            "regimes": list(REGIMES) + ["paraphrase"],
            "epochs": sorted(SCORE_EPOCHS),
            "embedding": EMB_NAME,
            "reader": LLM_NAME,
            "pool": POOL,
            "read": READ,
            "clearance": CFG,
            "primary": "epoch 8 QA accuracy, freshness, evidence hallucination",
            "qa": "exact match to the epoch truth",
            "fresh": "exact match on mutable keys whose truth left the initial value",
            "stale": "among those keys, the answer is an earlier truth",
            "hall": "generated value is not the value of a passage shown to the reader",
            "recall20": "top 20 contains a passage of this entity with the current value",
        },
        "summary": summary,
        "tests": tests,
        "rows": stored,
        "seconds": time.time() - started,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
