"""Refresh Testimony Model (RTM).

A source does not report the world. It reports its own record of the world,
and it touches that record only now and then. One latent event, the refresh,
explains three behaviors that earlier work treats with three mechanisms:

  stale restatement   no refresh since the last claim: the source repeats
                      its own record, whatever the world did
  copying             a refresh that reads another source's record
  testimony           a refresh that reads the world, correctly with
                      probability e, otherwise names some other value

World. Fact k has a hidden value z_{k,t} that persists with probability 1-h.
Source. Let a be the value of the source's previous claim about k, g epochs
earlier. Over the gap the source refreshes with probability F = 1-(1-f)^g.

  p(v | z) = (1-F) [v = a]
           + F { (1-kappa) ( e [v = z] + (1-e)/(n-1) [v != z] ) + kappa A(v) }

with A(v) = sum_u pi(u) [u's record about k equals v] the probability that a
copy reproduces v. A first claim has F = 1. Accuracy and refresh rate are
functions of two identities only, the source and the context of the fact:
  logit e = b_s + u_s w_c,   logit f = d_s + r_s q_c.
Because the previous claim is observed, the emission of a claim depends on z_t
alone and every fact stays a hidden Markov chain over its value set; the
marginal likelihood is exact and the fit is MAP by L-BFGS with restarts, as in
TSSM. Nothing in the decision is a threshold: a stored answer changes when the
filtered posterior of another value is larger.
"""

from __future__ import annotations

import math
import os
from collections import defaultdict

import numpy as np
import torch

DTYPE = torch.float64
LOG0 = -1e30
OTHER = "<other>"
DEVICE = torch.device(os.environ.get("RTM_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))


def logit(p):
    return math.log(p / (1.0 - p))


def sig(x):
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


class Spec:
    """Each switch removes one assumption of the model (ablations)."""

    def __init__(self, refresh=True, copy=True, rank=1, refresh_ctx=True, hazard=None, hindsight=True,
                 sincere=True, restarts=4, steps_first=60, steps=30, prior_scale=2.5, max_hazard=0.5,
                 copy_window=1, seed=0):
        self.refresh = refresh          # False: every claim is fresh testimony (F = 1)
        self.copy = copy                # False: kappa = 0
        self.rank = rank                # 0: accuracy and refresh do not depend on the context
        self.refresh_ctx = refresh_ctx  # False: refresh rate depends on the source only
        self.hazard = hazard            # None: learned; a float freezes h
        self.hindsight = hindsight      # False: the filter state is detached (credit at arrival)
        self.sincere = sincere          # population accuracy >= 1/2 (removes the mirror solution)
        self.restarts = restarts
        self.steps_first = steps_first
        self.steps = steps
        self.prior_scale = prior_scale
        self.max_hazard = max_hazard
        self.copy_window = copy_window  # a copy reads a record made at most this many epochs earlier
        self.seed = seed


SCALARS = ("beta0", "delta0", "gamma0", "lh")


class Params:
    def __init__(self, n_ctx, seed=0):
        self.rng = np.random.RandomState(seed)
        self.n_ctx = n_ctx
        self.b, self.u, self.d, self.r, self.k = {}, {}, {}, {}, {}
        self.up = {}
        self.w = self.rng.normal(0, 0.5, n_ctx)
        self.q = self.rng.normal(0, 0.5, n_ctx)
        self.beta0, self.delta0, self.gamma0, self.lh = 1.0, 0.0, -1.5, logit(0.2)

    def ensure(self, s, sincere):
        if s not in self.b:
            self.b[s] = float(np.log1p(np.exp(self.beta0))) if sincere else self.beta0
            self.u[s] = self.rng.normal(0, 0.5)
            self.d[s] = self.delta0
            self.r[s] = self.rng.normal(0, 0.5)
            self.k[s] = self.gamma0

    def fresh(self, seed):
        out = Params(self.n_ctx, seed)
        out.beta0 = out.rng.normal(1.0, 1.0)
        out.delta0 = out.rng.normal(0.0, 1.5)
        out.gamma0 = out.rng.normal(-1.5, 1.0)
        out.lh = logit(out.rng.uniform(0.05, 0.8))
        return out


class Problem:
    """All claims received up to epoch T. claims: iterable of (fact, value, source, epoch)."""

    def __init__(self, claims, T, ctx_of, spec):
        self.T = T
        seen, rows = set(), []
        for k, v, s, t in claims:
            if t > T or (k, s, t) in seen:
                continue
            seen.add((k, s, t))
            rows.append((k, v, s, t))
        rows.sort(key=lambda r: (r[0], r[2], r[3]))
        facts = sorted({r[0] for r in rows})
        self.facts = facts
        self.f_pos = {k: i for i, k in enumerate(facts)}
        vals = defaultdict(set)
        for k, v, s, t in rows:
            vals[k].add(v)
        self.values = {k: sorted(vals[k]) + [OTHER] for k in facts}
        self.index = {k: {v: i for i, v in enumerate(vs)} for k, vs in self.values.items()}
        self.sources = sorted({r[2] for r in rows})
        self.s_pos = {s: i for i, s in enumerate(self.sources)}
        # previous claim of the same source about the same fact
        prev_same, has_prev, gap = [], [], []
        last = {}
        for k, v, s, t in rows:
            p = last.get((k, s))
            if p is None:
                has_prev.append(0.0); prev_same.append(0.0); gap.append(1.0)
            else:
                has_prev.append(1.0); prev_same.append(1.0 if p[0] == v else 0.0); gap.append(float(t - p[1]))
            last[(k, s)] = (v, t)
        # copy candidates: another source whose record about k, made at most
        # copy_window epochs earlier (or now), names the same value
        rec = defaultdict(list)
        for k, v, s, t in rows:
            rec[(k, v)].append((s, t))
        pair_id, ent_row, ent_pair = {}, [], []
        for g, (k, v, s, t) in enumerate(rows):
            ups = sorted({u for u, tu in rec[(k, v)] if u != s and t - spec.copy_window <= tu <= t})
            for u in ups:
                key = (s, u)
                if key not in pair_id:
                    pair_id[key] = len(pair_id)
                ent_row.append(g); ent_pair.append(pair_id[key])
        self.pairs = sorted(pair_id, key=pair_id.get)
        D = DEVICE
        L = lambda a, dt=torch.int64: torch.tensor(a, dtype=dt, device=D)
        self.n = len(rows)
        self.rows = rows
        self.cf = L([self.f_pos[r[0]] for r in rows])
        self.ct = L([r[3] for r in rows])
        self.cv = L([self.index[r[0]][r[1]] for r in rows])
        self.cs = L([self.s_pos[r[2]] for r in rows])
        self.cc = L([ctx_of(r[0]) for r in rows])
        self.cnv = L([len(self.values[r[0]]) for r in rows], DTYPE)
        self.has_prev = L(has_prev, DTYPE)
        self.prev_same = L(prev_same, DTYPE)
        self.gap = L(gap, DTYPE)
        self.pair_src = L([self.s_pos[a] for a, _ in self.pairs] or [0])
        self.ent_row = L(ent_row or [0])
        self.ent_pair = L(ent_pair or [0])
        self.n_ent = len(ent_row)
        F = len(facts)
        self.V = max(len(v) for v in self.values.values()) if facts else 2
        self.nv = L([len(self.values[k]) for k in facts], DTYPE).view(F, 1)
        self.mask = torch.zeros(F, self.V, dtype=DTYPE, device=D)
        for i, k in enumerate(facts):
            self.mask[i, : len(self.values[k])] = 1.0


class Model:
    def __init__(self, spec, n_ctx):
        self.spec = spec
        self.n_ctx = n_ctx
        self.params = Params(n_ctx, spec.seed)
        self.history = []

    # ------------------------------------------------------------------ tensors
    def _tensors(self, prob, p):
        for s in prob.sources:
            p.ensure(s, self.spec.sincere)
        D = DEVICE
        mk = lambda a: torch.tensor(np.asarray(a, dtype=float), dtype=DTYPE, device=D, requires_grad=True)
        th = {name: mk([getattr(p, name)[s] for s in prob.sources]) for name in ("b", "u", "d", "r", "k")}
        th["up"] = mk([p.up.get(pr, 0.0) for pr in prob.pairs] or [0.0])
        th["w"], th["q"] = mk(p.w), mk(p.q)
        for name in SCALARS:
            th[name] = mk(getattr(p, name))
        if self.spec.hazard is not None:
            th["lh"] = torch.tensor(logit(self.spec.hazard / self.spec.max_hazard), dtype=DTYPE, device=D)
        return th

    def _store(self, prob, th, p):
        for name in ("b", "u", "d", "r", "k"):
            arr = th[name].detach().cpu().numpy()
            store = getattr(p, name)
            for i, s in enumerate(prob.sources):
                store[s] = float(arr[i])
        up = th["up"].detach().cpu().numpy()
        for i, pr in enumerate(prob.pairs):
            p.up[pr] = float(up[i])
        p.w, p.q = th["w"].detach().cpu().numpy().copy(), th["q"].detach().cpu().numpy().copy()
        for name in SCALARS:
            if name == "lh" and self.spec.hazard is not None:
                continue
            setattr(p, name, float(th[name].detach()))

    def _free(self, th):
        names = ["b", "d", "beta0", "delta0"]
        if self.spec.rank > 0:
            names += ["u", "w"]
            if self.spec.refresh_ctx:
                names += ["r", "q"]
        if self.spec.copy:
            names += ["k", "gamma0", "up"]
        if self.spec.hazard is None:
            names.append("lh")
        if not self.spec.refresh:
            names = [n for n in names if n not in ("d", "delta0", "r", "q")]
        return [th[n] for n in names]

    # ------------------------------------------------------------------ model
    def _logits(self, th, s, c):
        ze = th["b"][s]
        zf = th["d"][s]
        if self.spec.rank > 0:
            u = th["u"] - th["u"].mean() if self.spec.sincere else th["u"]
            ze = ze + u[s] * th["w"][c]
            if self.spec.refresh_ctx:
                zf = zf + th["r"][s] * th["q"][c]
        return ze, zf

    def _claim_logs(self, th, prob):
        ze, zf = self._logits(th, prob.cs, prob.cc)
        le = torch.nn.functional.logsigmoid(ze)
        lne = torch.nn.functional.logsigmoid(-ze) - torch.log(prob.cnv - 1.0)
        if self.spec.copy:
            kk = th["k"][prob.cs]
            lk, lnk = torch.nn.functional.logsigmoid(kk), torch.nn.functional.logsigmoid(-kk)
            la = self._copy_avail(th, prob)
            on = torch.logaddexp(lnk + le, lk + la)
            off = torch.logaddexp(lnk + lne, lk + la)
        else:
            on, off = le, lne
        if not self.spec.refresh:
            return on, off
        # probability of no refresh over the gap, and of at least one
        l1F = prob.gap * torch.nn.functional.logsigmoid(-zf)
        lF = torch.log(-torch.expm1(l1F).clamp(max=-1e-300))
        stale = torch.where(prob.prev_same > 0, l1F, torch.full_like(l1F, LOG0))
        on_p = torch.logaddexp(stale, lF + on)
        off_p = torch.logaddexp(stale, lF + off)
        hp = prob.has_prev > 0
        return torch.where(hp, on_p, on), torch.where(hp, off_p, off)

    def _copy_avail(self, th, prob):
        out = torch.full((prob.n,), LOG0, dtype=DTYPE, device=DEVICE)
        if prob.n_ent == 0:
            return out
        e = torch.exp(th["up"].clamp(-30, 30))
        den = torch.zeros(len(prob.sources), dtype=DTYPE, device=DEVICE).index_add(0, prob.pair_src, e)
        share = e / den[prob.pair_src]
        avail = torch.zeros(prob.n, dtype=DTYPE, device=DEVICE).index_add(0, prob.ent_row, share[prob.ent_pair])
        return torch.where(avail > 0, torch.log(avail.clamp_min(1e-300)), out)

    def _forward(self, th, prob, keep=False):
        T, F, V = prob.T, len(prob.facts), prob.V
        h = self.spec.max_hazard * torch.sigmoid(th["lh"])
        on, off = self._claim_logs(th, prob)
        cell = prob.cf * (T + 1) + prob.ct
        base = torch.zeros(F * (T + 1), dtype=DTYPE, device=DEVICE).index_add(0, cell, off)
        E = torch.zeros(F * (T + 1) * V, dtype=DTYPE, device=DEVICE).index_add(0, cell * V + prob.cv, on - off)
        E = E.view(F, T + 1, V) + base.view(F, T + 1, 1)
        mask, nv = prob.mask, prob.nv
        alpha = mask / mask.sum(-1, keepdim=True)
        total = torch.zeros((), dtype=DTYPE, device=DEVICE)
        seq = []
        for t in range(1, T + 1):
            if not self.spec.hindsight:
                alpha = alpha.detach()
            alpha = ((1 - h) * alpha + h * (1 - alpha) / (nv - 1)) * mask
            e = E[:, t, :].masked_fill(mask == 0, LOG0)
            m = e.max(-1, keepdim=True).values
            wgt = alpha * torch.exp(e - m)
            s = wgt.sum(-1, keepdim=True)
            alpha = wgt / s
            total = total + (torch.log(s) + m).sum()
            if keep:
                seq.append(alpha)
        return total, alpha, seq

    def _log_prior(self, th):
        lam = self.spec.prior_scale

        def gauss(x, center=0.0, scale=lam):
            return -0.5 * (((x - center) / scale) ** 2).sum()

        center = torch.nn.functional.softplus(th["beta0"]) if self.spec.sincere else th["beta0"]
        lp = gauss(th["b"], center) + gauss(th["d"], th["delta0"])
        if self.spec.rank > 0:
            lp = lp + gauss(th["u"]) + gauss(th["w"], 0.0, 1.0)
            if self.spec.refresh_ctx:
                lp = lp + gauss(th["r"]) + gauss(th["q"], 0.0, 1.0)
        if self.spec.copy:
            lp = lp + gauss(th["k"], th["gamma0"]) + gauss(th["up"])
        return lp

    def _objective(self, th, prob):
        return self._forward(th, prob)[0] + self._log_prior(th)

    def _optimize(self, prob, p, steps):
        th = self._tensors(prob, p)
        free = self._free(th)
        opt = torch.optim.LBFGS(free, lr=1.0, max_iter=steps, history_size=20, tolerance_grad=1e-6,
                                tolerance_change=1e-9, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = -self._objective(th, prob)
            loss.backward()
            return loss

        for _ in range(3):
            opt.step(closure)
        with torch.no_grad():
            value = float(self._objective(th, prob))
        return th, value if math.isfinite(value) else -float("inf")

    def fit(self, prob, first):
        cands = [self.params] + [self.params.fresh(1000 * prob.T + 17 * self.spec.seed + r)
                                 for r in range(self.spec.restarts)]
        best = None
        for i, p in enumerate(cands):
            th, value = self._optimize(prob, p, self.spec.steps_first if (first or i > 0) else self.spec.steps)
            if best is None or value > best[1]:
                best = (th, value, p)
        th, value, p = best
        self._store(prob, th, p)
        self.params = p
        self.history.append({"T": prob.T, "objective": value, **self.scalars()})
        return th

    def decide(self, prob, th):
        with torch.no_grad():
            _, alpha, _ = self._forward(th, prob)
        a = alpha.cpu().numpy()
        answers, post = {}, {}
        for i, k in enumerate(prob.facts):
            vals = prob.values[k]
            dist = {v: float(a[i, j]) for j, v in enumerate(vals) if v != OTHER}
            post[k] = dist
            answers[k] = max(dist, key=dist.get) if dist else None
        return answers, post

    # ------------------------------------------------------------------ read-outs
    def source_table(self, sources, n_ctx):
        p = self.params
        mu = np.mean(list(p.u.values())) if (self.spec.sincere and p.u) else 0.0
        out = {}
        for s in sources:
            p.ensure(s, self.spec.sincere)
            e = [sig(p.b[s] + (p.u[s] - mu) * p.w[c] * (self.spec.rank > 0)) for c in range(n_ctx)]
            f = [sig(p.d[s] + p.r[s] * p.q[c] * (self.spec.rank > 0 and self.spec.refresh_ctx)) for c in range(n_ctx)]
            out[s] = {"e": e, "f": f, "kappa": sig(p.k[s]) if self.spec.copy else 0.0}
        return out

    def scalars(self):
        p = self.params
        h = self.spec.hazard if self.spec.hazard is not None else self.spec.max_hazard * sig(p.lh)
        return {"h": h, "e0": sig(float(np.log1p(np.exp(p.beta0))) if self.spec.sincere else p.beta0),
                "f0": sig(p.delta0), "kappa0": sig(p.gamma0)}
