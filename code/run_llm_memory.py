"""LLM memory managers on WikiCoach (frozen Qwen2.5-7B-Instruct, vLLM, greedy).

  python run_llm_memory.py <tssm_posterior.json> <model_dir> <out.json> [max_clubs]

Evidence lines are the clubs' infobox entries in each language edition, with
the date on which the edition last changed that entry. Three managers:
  consolidate  every edition's current line; the model picks the line naming
               the coach in office (conflict resolution over all evidence)
  memory       Mem0-style update: the model sees its own previous answer and
               the lines that changed since, and keeps or replaces it
  tssm3        the three lines whose values have the highest TSSM posterior
The answer is a line number; it is scored by the coach that line names.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "wikicoach" / "claims.json"
NAMES = {"en": "English", "de": "German", "fr": "French", "es": "Spanish", "it": "Italian",
         "pt": "Portuguese", "nl": "Dutch", "ru": "Russian"}


def evidence(d):
    """state[(k, t)] -> list of (lang, value, shown, since) for quarter t."""
    rows = sorted(d["claims"], key=lambda r: r[3])
    last = {}
    since = {}
    state = defaultdict(list)
    for k, v, lang, t, _rev, shown in rows:
        if last.get((k, lang)) != v:
            since[(k, lang)] = d["boundaries"][t - 1]
            last[(k, lang)] = v
        state[(k, t)].append((lang, v, shown, since[(k, lang)]))
    return state


def prompt(club, date, lines, memory=None):
    head = (f"Several language editions of Wikipedia name the head coach of {club}. Editions update at "
            f"different speeds, so some lines may be outdated or wrong.\n")
    body = []
    if memory is not None:
        body.append(f"[0] your memory (since {memory[1]}): {memory[0]}")
    for i, (lang, _v, shown, since) in enumerate(lines, 1):
        body.append(f"[{i}] {NAMES.get(lang, lang)} Wikipedia (entry unchanged since {since}): {shown}")
    ask = (f"Who was the head coach of {club} on {date}? "
           + ("Answer 0 to keep your memory, or the number of the line that should replace it. "
              if memory is not None else "")
           + "Reply with the line number only.")
    return head + "\n".join(body) + "\n\n" + ask


def parse(text, n):
    m = re.search(r"\d+", text or "")
    if not m:
        return None
    i = int(m.group(0))
    return i if 0 <= i <= n else None


def main():
    from vllm import LLM, SamplingParams
    d = json.loads(DATA.read_text(encoding="utf-8"))
    post = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["post"]   # post[t-1][k] = {value: p}
    max_clubs = int(sys.argv[4]) if len(sys.argv) > 4 else len(d["facts"])
    facts = d["facts"][:max_clubs]
    T = len(d["boundaries"])
    state = evidence(d)
    llm = LLM(model=sys.argv[2], quantization="awq", dtype="float16",
              gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.36")), max_model_len=3072, seed=0)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=4)

    def run(prompts):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
                 for p in prompts]
        return [o.outputs[0].text for o in llm.generate(texts, sp)] if texts else []

    answers = {m: [[None] * len(facts) for _ in range(T)] for m in ("consolidate", "memory", "tssm3")}
    # consolidate and tssm3 are independent per quarter
    jobs = []
    for t in range(1, T + 1):
        for k, f in enumerate(facts):
            lines = state.get((k, t), [])
            if not lines:
                continue
            jobs.append(("consolidate", t, k, lines))
            p = post[t - 1][k] if post[t - 1][k] else {}
            ranked = sorted(lines, key=lambda x: (-p.get(x[1], 0.0), x[3]), reverse=False)[:3]
            jobs.append(("tssm3", t, k, ranked))
    outs = run([prompt(facts[k]["label"], d["boundaries"][t - 1], lines) for _m, t, k, lines in jobs])
    for (m, t, k, lines), o in zip(jobs, outs):
        i = parse(o, len(lines))
        answers[m][t - 1][k] = lines[i - 1][1] if i else None
    # memory runs quarter by quarter
    mem = {}
    for t in range(1, T + 1):
        batch = []
        for k, f in enumerate(facts):
            lines = state.get((k, t), [])
            if k not in mem:
                if lines:
                    batch.append((k, lines, None))
                continue
            changed = [x for x in lines if x[3] == d["boundaries"][t - 1]]
            if changed:
                batch.append((k, changed, mem[k]))
        outs = run([prompt(facts[k]["label"], d["boundaries"][t - 1], lines, memory=(m[2], m[3]) if m else None)
                    for k, lines, m in batch])
        for (k, lines, m), o in zip(batch, outs):
            i = parse(o, len(lines))
            if m is not None and (i == 0 or i is None):
                continue
            if i:
                lang, v, shown, since = lines[i - 1]
                mem[k] = (lang, v, shown, d["boundaries"][t - 1])
        for k in range(len(facts)):
            answers["memory"][t - 1][k] = mem[k][1] if k in mem else None
        print("quarter", t, "memory updates", len(batch), flush=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_wikicoach as wc
    sub = dict(d)
    sub["facts"] = facts
    res = {m: wc.score(sub, wc.carry(g)) for m, g in answers.items()}
    tssm_grid = [[max(post[t][k], key=post[t][k].get) if post[t][k] else None for k in range(len(facts))] for t in range(T)]
    res["tssm_mode"] = wc.score(sub, tssm_grid)
    for m, r in res.items():
        print(m, {k: round(r[k], 4) for k in ("all", "changed", "stable", "delay", "balanced")}, flush=True)
    Path(sys.argv[3]).write_text(json.dumps(res), encoding="utf-8")


if __name__ == "__main__":
    main()
