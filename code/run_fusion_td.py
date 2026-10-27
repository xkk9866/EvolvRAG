"""Classical truth discovery on the Stock and Flight source files.

Every method reads exactly the claims the testimony model reads: each
(item, day) is a fact, only items with gold on that day are kept, and the
estimator is refit on all days up to the scored day before that day is
answered. Source quality is shared across days, as in the original papers.
Gold is read only for scoring. Constants are the published defaults.

  majority     one vote per source
  truthfinder  Yin et al. 2008, gamma 0.3, initial trust 0.9, no implication
  accu         Dong et al. 2009 (Accu without copy detection), n false values
               per fact = distinct values - 1, initial accuracy 0.8
  dawid        one-coin Dawid-Skene by EM with a uniform prior over the
               values on record for the fact
  crh          Li et al. 2014, 0-1 loss, weight -log(loss share)
  invest       Pasternack and Roth 2010, growth 1.2
  two          Galland et al. 2010, 2-Estimates
  sums         Kleinberg-style hubs and authorities over claims
  accucopy     Dong et al. 2009, copy prior 0.2, copy rate 0.8
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_fusion as fu

ROOT = Path(__file__).resolve().parents[1]
ITERS = 20


class Batch:
    """Deduplicated claims up to a day as index arrays."""

    def __init__(self, days, upto, src_id):
        fact_id, cell_id = {}, {}
        f_of_cell, src, cell = [], [], []
        values = []
        seen = set()
        for d in range(upto + 1):
            gold = days[d]["gold"]
            for slot, value, source, _org in days[d]["claims"]:
                if slot not in gold:
                    continue
                key = (slot, d)
                if key not in fact_id:
                    fact_id[key] = len(fact_id)
                    values.append({})
                f = fact_id[key]
                s = src_id.setdefault(source, len(src_id))
                if (f, s) in seen:
                    continue
                seen.add((f, s))
                if value not in values[f]:
                    values[f][value] = len(cell_id)
                    cell_id[(f, value)] = len(cell_id)
                    f_of_cell.append(f)
                src.append(s)
                cell.append(values[f][value])
        self.fact_id = fact_id
        self.values = values
        self.n_src = len(src_id)
        self.src = np.array(src, dtype=np.int64)
        self.cell = np.array(cell, dtype=np.int64)
        self.f_of_cell = np.array(f_of_cell, dtype=np.int64)
        self.n_cell = len(f_of_cell)
        self.n_fact = len(fact_id)
        self.nv = np.bincount(self.f_of_cell, minlength=self.n_fact).astype(float)
        self.claims_per_src = np.bincount(self.src, minlength=self.n_src).astype(float)
        self.claims_per_cell = np.bincount(self.cell, minlength=self.n_cell).astype(float)

    def cell_sum(self, per_claim):
        return np.bincount(self.cell, weights=per_claim, minlength=self.n_cell)

    def src_sum(self, per_claim):
        return np.bincount(self.src, weights=per_claim, minlength=self.n_src)

    def fact_max(self, per_cell):
        out = np.full(self.n_fact, -np.inf)
        np.maximum.at(out, self.f_of_cell, per_cell)
        return out

    def softmax(self, score):
        m = self.fact_max(score)
        e = np.exp(score - m[self.f_of_cell])
        z = np.bincount(self.f_of_cell, weights=e, minlength=self.n_fact)
        return e / z[self.f_of_cell]

    def winners(self, score):
        """Cell with the highest score per fact (lowest index breaks ties)."""
        order = np.lexsort((np.arange(self.n_cell), -score, self.f_of_cell))
        first = np.ones(self.n_cell, dtype=bool)
        first[1:] = self.f_of_cell[order][1:] != self.f_of_cell[order][:-1]
        win = np.zeros(self.n_fact, dtype=np.int64)
        win[self.f_of_cell[order][first]] = order[first]
        return win

    def is_win(self, score):
        mark = np.zeros(self.n_cell)
        mark[self.winners(score)] = 1.0
        return mark


def majority(b):
    return b.claims_per_cell


def truthfinder(b, gamma=0.3, t0=0.9):
    t = np.full(b.n_src, t0)
    for _ in range(ITERS):
        tau = -np.log(np.clip(1 - t, 1e-6, 1))
        sigma = b.cell_sum(tau[b.src])
        conf = 1.0 / (1.0 + np.exp(-gamma * sigma))
        t = b.src_sum(conf[b.cell]) / np.maximum(b.claims_per_src, 1)
    return sigma


def accu(b, a0=0.8):
    a = np.full(b.n_src, a0)
    n = np.maximum(b.nv[b.f_of_cell], 2.0) - 1.0
    for _ in range(ITERS):
        aa = np.clip(a, 1e-4, 1 - 1e-4)
        vote = np.log(aa[b.src] / (1 - aa[b.src]))
        score = b.cell_sum(vote) + np.log(n) * b.claims_per_cell
        p = b.softmax(score)
        a = b.src_sum(p[b.cell]) / np.maximum(b.claims_per_src, 1)
    return score


def dawid(b, a0=0.8):
    a = np.full(b.n_src, a0)
    for _ in range(ITERS):
        aa = np.clip(a, 1e-4, 1 - 1e-4)
        # one-coin: a source says the truth w.p. a, each other value on record w.p. (1-a)/(nv-1)
        off = np.log((1 - aa[b.src]) / np.maximum(b.nv[b.f_of_cell[b.cell]] - 1, 1))
        on = np.log(aa[b.src])
        claim_f = b.f_of_cell[b.cell]
        base = np.bincount(claim_f, weights=off, minlength=b.n_fact)
        score = base[b.f_of_cell] + b.cell_sum(on - off)
        p = b.softmax(score)
        a = (b.src_sum(p[b.cell]) + 1.0) / (b.claims_per_src + 2.0)
    return score


def crh(b):
    w = np.ones(b.n_src)
    for _ in range(ITERS):
        score = b.cell_sum(w[b.src])
        win = b.is_win(score)
        loss = b.src_sum(1.0 - win[b.cell]) + 1e-6
        w = -np.log(loss / loss.sum())
    return b.cell_sum(w[b.src])


def invest(b, g=1.2):
    t = np.ones(b.n_src)
    for _ in range(ITERS):
        share = t / np.maximum(b.claims_per_src, 1)
        base = b.cell_sum(share[b.src])
        belief = base ** g
        t = b.src_sum((belief / np.maximum(base, 1e-12))[b.cell] * share[b.src])
        t = t / max(t.max(), 1e-12)
    return belief


def two_estimates(b):
    t = np.full(b.n_src, 0.8)
    for _ in range(ITERS):
        # a claim for v is a vote against every other value of the fact
        pos = b.cell_sum(t[b.src])
        neg_f = np.bincount(b.f_of_cell[b.cell], weights=1 - t[b.src], minlength=b.n_fact)
        n_f = np.bincount(b.f_of_cell[b.cell], minlength=b.n_fact).astype(float)
        neg = neg_f[b.f_of_cell] - b.cell_sum(1 - t[b.src])
        conf = (pos + neg) / np.maximum(n_f[b.f_of_cell], 1)
        conf = (conf - conf.min()) / max(conf.max() - conf.min(), 1e-12)
        # source trust: agreement on own claims, disagreement elsewhere on its facts
        t = b.src_sum(conf[b.cell]) / np.maximum(b.claims_per_src, 1)
        t = (t - t.min()) / max(t.max() - t.min(), 1e-12)
    return conf


def sums(b):
    t = np.ones(b.n_src)
    for _ in range(ITERS):
        belief = b.cell_sum(t[b.src])
        belief = belief / max(belief.max(), 1e-12)
        t = b.src_sum(belief[b.cell])
        t = t / max(t.max(), 1e-12)
    return belief


def accucopy(b, a0=0.8, alpha=0.2, c=0.8, outer=5):
    """AccuCopy (Dong, Berti-Equille and Srivastava 2009). Pairwise copy
    probability from shared true, shared false and differing values; a vote
    is discounted by the chance that its source copied a more accurate
    provider of the same value."""
    from scipy import sparse
    S = b.n_src
    a = np.full(S, a0)
    n_claim = len(b.src)
    M = sparse.csr_matrix((np.ones(n_claim), (b.src, b.cell)), shape=(S, b.n_cell))
    claim_f = b.f_of_cell[b.cell]
    Mf = sparse.csr_matrix((np.ones(n_claim), (b.src, claim_f)), shape=(S, b.n_fact))
    shared = (Mf @ Mf.T).toarray()
    n_false = np.maximum(b.nv - 1, 1)
    order = np.argsort(b.cell, kind="stable")
    bounds = np.flatnonzero(np.r_[True, b.cell[order][1:] != b.cell[order][:-1], True])
    groups = [order[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
    dep = np.zeros((S, S))
    n = np.maximum(b.nv[b.f_of_cell], 2.0) - 1.0
    score = None
    for it in range(outer * 4):
        aa = np.clip(a, 1e-3, 1 - 1e-3)
        vote = np.log(n[b.cell] * aa[b.src] / (1 - aa[b.src]))
        if it >= 1:
            # discount: product over more accurate co-providers of 1 - c P(copy)
            disc = np.zeros(n_claim)
            for g in groups:
                if len(g) < 2:
                    continue
                s = b.src[g]
                # providers in decreasing accuracy, ties by source id
                rank = np.lexsort((s, -aa[s]))
                pos = np.empty(len(s), dtype=np.int64)
                pos[rank] = np.arange(len(s))
                earlier = pos[None, :] < pos[:, None]
                disc[g] = np.log(np.clip(1 - c * dep[np.ix_(s, s)] * earlier, 1e-6, 1)).sum(1)
            vote = vote * np.exp(disc)
        score = b.cell_sum(vote)
        p = b.softmax(score)
        a = b.src_sum(p[b.cell]) / np.maximum(b.claims_per_src, 1)
        if True:
            aa = np.clip(a, 1e-3, 1 - 1e-3)
            kt = (M @ sparse.diags(p) @ M.T).toarray()
            kf = (M @ sparse.diags(1 - p) @ M.T).toarray()
            kd = np.maximum(shared - kt - kf, 0)
            nf = float(np.median(n_false))
            pt = aa[:, None] * aa[None, :]
            pf = (1 - aa[:, None]) * (1 - aa[None, :]) / nf
            pd = np.clip(1 - pt - pf, 1e-9, 1)
            ind = kt * np.log(pt) + kf * np.log(pf) + kd * np.log(pd)
            # s1 copies s2: a shared value is copied w.p. c, else independent
            cp = kt * np.log(aa[None, :] * c + pt * (1 - c)) + kf * np.log((1 - aa[None, :]) * c + pf * (1 - c)) \
                + kd * np.log(pd * (1 - c))
            cp2 = cp.T
            m = np.maximum(np.maximum(ind, cp), cp2)
            num1 = alpha / 2 * np.exp(cp - m)
            num2 = alpha / 2 * np.exp(cp2 - m)
            den = (1 - alpha) * np.exp(ind - m) + num1 + num2
            dep = (num1 + num2) / den
            np.fill_diagonal(dep, 0.0)
    return score


METHODS = {"majority": majority, "truthfinder": truthfinder, "accu": accu, "dawid": dawid,
           "crh": crh, "invest": invest, "two": two_estimates, "sums": sums, "accucopy": accucopy}


def dynatd_answers(days, decay):
    """DynaTD (Li et al. 2015): truths of each day by a weighted vote whose
    weights come from the exponentially decayed 0-1 loss of the earlier days;
    the day's own loss enters after the vote (online, no hindsight)."""
    loss = {}
    count = {}
    answers = []
    for day in days:
        gold_items = day["gold"]
        votes = {}
        seen = set()
        for slot, value, source, _org in day["claims"]:
            if slot not in gold_items or (slot, source) in seen:
                continue
            seen.add((slot, source))
            votes.setdefault(slot, []).append((source, value))
        total = sum(loss.values()) + 1e-6
        def weight(s):
            return math.log((total + 1.0) / (loss.get(s, 0.0) + 1e-3 * (count.get(s, 0.0) + 1.0) + 1e-6))
        chosen = {}
        for slot, rows in votes.items():
            score = {}
            for s, v in rows:
                score[v] = score.get(v, 0.0) + max(weight(s), 0.0)
            chosen[slot] = max(sorted(score), key=lambda v: score[v])
        for s in list(loss):
            loss[s] *= decay
            count[s] *= decay
        for slot, rows in votes.items():
            for s, v in rows:
                loss[s] = loss.get(s, 0.0) + (v != chosen[slot])
                count[s] = count.get(s, 0.0) + 1.0
        answers.append(chosen)
    return answers


def run(days, names):
    src_id = {}
    items = {m: [] for m in names}
    changed = {m: [] for m in names}
    t0 = time.time()
    for d in range(1, len(days)):
        b = Batch(days, d, src_id)
        inv = [None] * b.n_cell
        for f, vals in enumerate(b.values):
            for v, c in vals.items():
                inv[c] = v
        prev = days[d - 1]["gold"]
        for m in names:
            score = METHODS[m](b)
            win = b.winners(score)
            for slot, gold in days[d]["gold"].items():
                f = b.fact_id.get((slot, d))
                ok = int(f is not None and inv[win[f]] == gold)
                items[m].append(ok)
                if slot in prev and prev[slot] != gold:
                    changed[m].append(ok)
        print(days[d]["day"], {m: round(float(np.mean(items[m][-len(days[d]["gold"]):])), 3) for m in names},
              "sec", round(time.time() - t0), flush=True)
    return {m: {"em": float(np.mean(items[m])), "changed": float(np.mean(changed[m])),
                "items": items[m], "changed_items": changed[m]} for m in names}


def main():
    which = sys.argv[1]
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results" / f"fusion_td_{which}.json"
    days = fu.stock_days() if which == "stock" else fu.flight_days()
    if len(sys.argv) > 4 and sys.argv[4] == "tol":
        import fusion_tol
        days = fusion_tol.canonical(days)
    names = sys.argv[3].split(",") if len(sys.argv) > 3 else list(METHODS)
    res = run(days, names)
    for m, row in res.items():
        print(which, m, round(row["em"], 4), round(row["changed"], 4), flush=True)
    out.write_text(json.dumps(res), encoding="utf-8")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
