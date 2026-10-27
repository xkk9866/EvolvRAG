"""Development comparison: reader on the margin passage versus the top five.

Official HoH slices are not used. N=100 development rows.
"""

from collections import Counter
import math

import run_hoh as hoh
import run_reader as reader
from probe_copy import DEV, build, top_passages, later_of_pair
from run_public_rag import PublicReader, hoh_prompt, norm_answer

N = 100


def main():
    rows = DEV[:N]
    docs, tfs, idf, avg = build(rows)
    packs = {"margin": [], "pair": [], "top5": []}
    golds = []
    for qid, row in enumerate(rows):
        cands = top_passages(row, qid, docs, tfs, idf, avg)
        margin = hoh.choose(cands, "margin", 0.85)
        revised = later_of_pair(cands)
        earlier = min(cands, key=lambda item: item["time"])
        packs["margin"].append(hoh_prompt(row["question"], [{"time": margin["time"], "text": margin["text"]}]))
        packs["pair"].append(
            hoh_prompt(
                row["question"],
                [
                    {"time": earlier["time"], "text": earlier["text"]},
                    {"time": revised["time"], "text": revised["text"]},
                ],
            )
        )
        packs["top5"].append(
            hoh_prompt(row["question"], [{"time": item["time"], "text": item["text"]} for item in cands])
        )
        golds.append(norm_answer(row["answer"]))
    device = reader.device_of()
    print("device", device, flush=True)
    llm = PublicReader(device)
    for name, prompts in packs.items():
        outs = llm.complete(prompts)
        hits = sum(int(norm_answer(text) == gold) for text, gold in zip(outs, golds))
        print(name, "em", round(hits / N, 3), flush=True)


if __name__ == "__main__":
    main()
