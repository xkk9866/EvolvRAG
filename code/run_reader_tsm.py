"""Retrieval-augmented answering over the evolving corpus with a frozen reader.

Stage 1 (CPU):  python run_reader_tsm.py prepare <seeds> <regimes> <epochs> <out.jsonl>
  For each world and scored epoch, retrieve 20 passages per question with BM25
  and with BAAI/bge-base-en-v1.5, and build the reader windows:
    top5     the five best-scored passages
    newest5  the five most recent of the twenty
    clear5   EvolvRAG v1: passages from issuers cleared by the count-switch
             record, newest first (fallback: schema, then stored belief)
    tssm5    the twenty re-ranked by the testimony model's filtered posterior
             probability of the value each passage states
  The testimony model's posterior mode is also stored (no reader).
Stage 2 (GPU):  python run_reader_tsm.py read <in.jsonl> <model_dir> <out.json>
  Greedy decoding with vLLM, eight new tokens, one prompt for every window.
Scores: QA exact match over all keys; Fresh on changed mutable keys; Stale is
an earlier truth on those keys; Hall is a generated value not in the window.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

POOL = 20
READ = 5
VALUE_RE = re.compile(r"via-v\d+|ov\d+-\d+|v\d+|q\d+|EXPIRED", re.IGNORECASE)
EMB = "BAAI/bge-base-en-v1.5"
QPREFIX = "Represent this sentence for searching relevant passages: "


def parse_value(text):
    m = VALUE_RE.search(text or "")
    if m is None:
        return None
    tok = m.group(0)
    return "EXPIRED" if tok.upper() == "EXPIRED" else tok.lower()


def prompt_of(question, passages):
    lines = [f"[{i}] time {p['time']}: {p['text']}" for i, p in enumerate(passages, 1)]
    return ("Answer the question using only the passages. "
            "If they disagree about the named entity, use the passage with the later time. "
            "Reply with one value token and no other words.\n\n"
            f"Question: {question}\n\nPassages:\n" + "\n".join(lines) + "\n\nValue:")


def prepare(seeds, regimes, epochs, out_path):
    import torch
    from transformers import AutoModel, AutoTokenizer
    import run_experiment as rx
    import run_fair as fair
    import run_rag as rag
    import run_reader as old
    import tsm
    import run_tsm
    torch.set_num_threads(int(os.environ.get("THREADS", "16")))
    tok = AutoTokenizer.from_pretrained(EMB, local_files_only=True)
    enc = AutoModel.from_pretrained(EMB, local_files_only=True).eval()

    def embed(texts, query=False):
        if query:
            texts = [QPREFIX + t for t in texts]
        out = []
        for i in range(0, len(texts), 128):
            b = tok(texts[i:i + 128], padding=True, truncation=True, max_length=128, return_tensors="pt")
            with torch.inference_mode():
                h = enc(**b).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(h, p=2, dim=1))
        return torch.cat(out)

    hier = next(c for c in fair.RULES if c["name"] == "hier")
    with open(out_path, "w", encoding="utf-8") as fh:
        for regime in regimes:
            for seed in seeds:
                t0 = time.time()
                world = fair.load_world(seed, regime)
                names = world.names
                order = list(world.mutable_ids) + list(world.stable_ids) + list(world.derived_ids)
                book = fair.TrustBook(hier)
                belief_v1 = list(world.init_belief[0])
                model = tsm.Model(tsm.Spec(), run_tsm.N_CTX)
                claims, mail = [], []
                for epoch, docs in enumerate(world.visible, start=1):
                    mail.extend(docs)
                    claims.extend(run_tsm.world_claims(docs, names, epoch))
                    prob = tsm.Problem(claims, epoch, list(world.init_belief[0]), world.parent, run_tsm.schema_fn,
                                       run_tsm.ctx_of, world.n_slots, True, "arrival", "graph")
                    th = model.fit(prob, first=epoch == 1)
                    find = fair.copy_find(mail, names)
                    usable = fair.collapsed(mail, find)
                    buckets = fair.index_hits(usable, names)
                    if epoch in epochs:
                        post = model.posterior(prob, th)
                        mode, _ = model.decide(prob, th)
                        questions = [rx.query_text(world, k) for k in order]
                        index = rag.Index(mail)
                        mat = embed([d["text"] for d in mail])
                        qmat = embed(questions, query=True)
                        row = list(belief_v1)
                        for qi, slot in enumerate(order):
                            truth = world.timeline[epoch - 1][slot]
                            dense20 = old.dense_topk(qmat[qi], mat, mail, POOL)
                            bm20 = old.bm25_topk(index, questions[qi], POOL)
                            trust = book.for_slot(slot)
                            schema_v = fair.schema_value(world, row, slot)
                            rec = {"seed": seed, "regime": regime, "epoch": epoch, "slot": slot,
                                   "question": questions[qi], "truth": truth,
                                   "fresh": slot in world.mutable_ids and truth != world.truth0[slot],
                                   "earlier": sorted({world.timeline[e][slot] for e in range(epoch - 1)} | {world.truth0[slot]}),
                                   "tssm_mode": mode[slot], "windows": {}, "recall": {}}
                            dist = post.get(slot, {})
                            for rname, pool in (("dense", dense20), ("bm25", bm20)):
                                pas = [{"text": d["text"], "time": d["time"], "value": d["value"], "seq": d.get("seq", 0)} for d in pool]
                                rec["recall"][rname] = int(any(fair.entity_of_text(d["text"], names) == slot and d["value"] == truth for d in pool))
                                rec["windows"][f"{rname}_top5"] = pas[:READ]
                                rec["windows"][f"{rname}_newest5"] = sorted(pas, key=lambda p: (p["time"], p["seq"]), reverse=True)[:READ]
                                kept = old.eligible(pool, trust, hier["cutoff"])[:READ]
                                rec["windows"][f"{rname}_clear5"] = [{"text": d["text"], "time": d["time"], "value": d["value"], "seq": d.get("seq", 0)} for d in kept]
                                rec[f"{rname}_clear_fallback"] = None if kept else (schema_v if schema_v is not None else row[slot])
                                ranked = sorted(pas, key=lambda p: (-(dist.get(p["value"], 0.0) if fair.entity_of_text(p["text"], names) == slot else -1.0), -p["time"], -p["seq"]))
                                rec["windows"][f"{rname}_tssm5"] = ranked[:READ]
                            # the v1 belief row advances with its own clearance answer
                            row[slot] = fair.answer_slot(hier, buckets.get(slot, []), row[slot], trust, schema_v)
                            fh.write(json.dumps(rec) + "\n")
                    # v1 belief and record updates follow run_fair.run_cfg
                    row2 = list(belief_v1)
                    for slot in order:
                        sv = fair.schema_value(world, row2, slot)
                        row2[slot] = fair.answer_slot(hier, buckets.get(slot, []), row2[slot], book.for_slot(slot), sv)
                    belief_v1 = row2
                    book.update(fair.collapsed(docs, find), usable, names)
                print("prepared", regime, seed, round(time.time() - t0, 1), flush=True)


def read(in_path, model_dir, out_path):
    from vllm import LLM, SamplingParams
    recs = [json.loads(line) for line in open(in_path, encoding="utf-8")]
    llm = LLM(model=model_dir, quantization="awq", dtype="float16", gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.45")),
              max_model_len=2048, enforce_eager=False, seed=0)
    tok = llm.get_tokenizer()
    prompts, owners = [], []
    for i, r in enumerate(recs):
        for name, win in r["windows"].items():
            if not win:
                continue
            msg = [{"role": "user", "content": prompt_of(r["question"], win)}]
            text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True,
                                           **({"enable_thinking": False} if "qwen3" in model_dir.lower() else {}))
            prompts.append(text)
            owners.append((i, name))
    t0 = time.time()
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=8))
    print("generated", len(prompts), "in", round(time.time() - t0), "s", flush=True)
    gen = {}
    for (i, name), o in zip(owners, outs):
        gen[(i, name)] = (parse_value(o.outputs[0].text), o.outputs[0].text)
    rows = []
    for i, r in enumerate(recs):
        row = {k: r[k] for k in ("seed", "regime", "epoch", "slot", "truth", "fresh", "earlier", "tssm_mode", "recall")}
        row["answers"] = {}
        for name, win in r["windows"].items():
            shown = sorted({p["value"] for p in win})
            if win:
                ans, raw = gen[(i, name)]
                row["answers"][name] = {"value": ans, "generated": True, "shown": shown}
            else:
                fb = r.get(name.replace("_clear5", "_clear_fallback")) if name.endswith("clear5") else None
                row["answers"][name] = {"value": fb, "generated": False, "shown": shown}
        rows.append(row)
    Path(out_path).write_text(json.dumps({"model": model_dir, "rows": rows}), encoding="utf-8")
    print("wrote", out_path, len(rows), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        a, b = sys.argv[2].split("-")
        prepare(list(range(int(a), int(b) + 1)), sys.argv[3].split(","), {int(x) for x in sys.argv[4].split(",")}, sys.argv[5])
    else:
        read(sys.argv[2], sys.argv[3], sys.argv[4])
