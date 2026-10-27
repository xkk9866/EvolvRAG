"""Testimony state-space model.

Memory revision is posterior inference in one generative model of testimony.

  world     each fact k has a value z_{k,t} at every epoch t. The value is kept
            with probability 1-h and jumps to another value with probability h.
            The stored memory is a noisy reading of z_{k,0} (accuracy mu).
            A derived fact follows its parent through a known schema f, except
            that a parent change is overridden with probability omega.
  witness   a source s either repeats a claim that is already on record for the
            same fact and reported time (probability kappa_s), or testifies.
            Testimony about fact k at reported time tau says z_{k,tau} with
            probability rho_s(c_k) and any other value otherwise.
            rho_s(c) = sigmoid(a_s + u_s . w_c) is a low-rank credibility map
            over sources and contexts (the organization that owns the fact).
            A source never seen before sits at the population means.

Every quantity the decision uses (h, mu, omega, a, u, w, kappa and the two
population means) is fitted by maximum a posteriori on the marginal likelihood
of the testimony that has arrived so far. No truth label, threshold, or
recipient list is read. The answer is the filtered posterior mode at the
current epoch. The gradient of the marginal likelihood is an expectation under
the smoothed posterior, so a source is credited in hindsight: each claim is
judged against the trajectory that all testimony on record supports.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch

OTHER = "<other>"
DTYPE = torch.float64
LOG0 = -1e30


def softplus(x):
    return math.log1p(math.exp(-abs(x))) + max(x, 0.0)


def logit(p):
    return math.log(p / (1.0 - p))


def sig(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


class Spec:
    """Model switches. Each ablation removes one assumption of the model."""

    def __init__(self, rank=1, hazard=None, hindsight=True, memory="none", schema=True,
                 context=True, full_context=False, copy=True, restarts=4,
                 steps_first=60, steps=30, lr=0.05, prior_scale=2.5, oracle=None,
                 clock="arrival", learn_scale=False, optimizer="lbfgs", rate_prior=0.0,
                 init_spread=2.0, max_hazard=0.5, echo="graph", drift=0.0, sincere=False, span="time"):
        self.rank = rank if context else 0
        self.hazard = hazard            # None: learned. A float freezes h.
        self.hindsight = hindsight      # False: credit fixed at arrival time
        # "decision": witnesses are judged without the agent's own memory; the
        # memory enters only as the prior of the decision (no self-certification).
        # "joint": memory is evidence while credibility is learned. "none": unused.
        self.memory = {True: "joint", False: "none"}.get(memory, memory)
        self.schema = schema            # False: derived facts are modeled alone
        self.full_context = full_context  # independent rho per (source, context)
        self.copy = copy                # False: every claim is independent testimony
        self.restarts = restarts        # fresh starts per epoch besides the warm start
        self.steps_first = steps_first
        self.steps = steps
        self.lr = lr
        self.prior_scale = prior_scale
        self.oracle = oracle            # callable (issuer, ctx) -> rho, for an upper bound
        self.clock = clock              # "arrival": a claim describes the epoch it arrived in
        self.learn_scale = learn_scale  # population spread of credibility is fitted
        self.optimizer = optimizer
        self.rate_prior = rate_prior    # Beta(1+r, 1+r) pseudo-counts on h, mu, omega
        self.init_spread = init_spread  # spread of fresh multi-start initializations
        self.max_hazard = max_hazard    # persistence is more likely than change
        self.echo = echo                # "graph": an echo reproduces the claim of an upstream
                                        # witness drawn from a learned per-source distribution;
                                        # "exact": any identical claim on record may be the source
        # True: the population's typical source is more often right than wrong,
        # sigma(beta0) >= 1/2 (the witness counterpart of h < 1/2)
        self.sincere = sincere
        # "time": an echo repeats a claim about the same fact and reported time by
        # another source; "fact": any earlier identical claim about the fact,
        # the source's own included (a re-statement is an echo of oneself)
        self.span = span
        self.drift = drift              # > 0: credibility logit follows a Gaussian random walk
                                        # over epochs with this step scale; 0: constant


SCALARS = ("beta0", "gamma0", "lh", "lmu", "lom", "lsb", "lsu", "lsk")


class Params:
    """Parameters that persist across epochs (warm start)."""

    def __init__(self, n_ctx, rank, seed=0, sincere=False):
        self.sincere = sincere
        self.rank = rank
        self.n_ctx = n_ctx
        self.rng = np.random.RandomState(seed)
        self.bias, self.vec, self.full, self.cop = {}, {}, {}, {}
        self.up = {}   # (source, upstream) -> logit of choosing that upstream when echoing
        self.walk = {}  # source -> per-epoch credibility offset (random walk)
        self.ctx = self.rng.normal(0.0, 0.5, size=(n_ctx, max(rank, 1)))
        self.beta0 = 0.0
        self.gamma0 = logit(0.2)
        self.lh = logit(0.2)
        self.lmu = logit(0.8)
        self.lom = logit(0.2)
        self.lsb = 0.0   # log spread of source credibility
        self.lsu = 0.0   # log spread of context-specific credibility
        self.lsk = 0.0   # log spread of copy propensity

    def ensure(self, issuer):
        if issuer not in self.bias:
            self.bias[issuer] = self.center()
            self.vec[issuer] = self.rng.normal(0.0, 0.5, size=max(self.rank, 1))
            self.full[issuer] = np.zeros(self.n_ctx)
            self.cop[issuer] = self.gamma0
            self.walk[issuer] = np.zeros(0)

    def center(self):
        """Population mean of the credibility logit."""
        return softplus(self.beta0) if self.sincere else self.beta0

    def fresh(self, seed, spread=1.0):
        out = Params(self.n_ctx, self.rank, seed, self.sincere)
        out.beta0 = out.rng.normal(0.0, spread)
        out.gamma0 = out.rng.normal(-1.0, spread)
        out.lh = logit(out.rng.uniform(0.06, 0.8))
        out.lmu = logit(out.rng.uniform(0.55, 0.95))
        return out


class Problem:
    """Tensors for one inference call: all claims delivered up to t_now.

    claims is a list of (slot, value, issuer, reported_time, arrival_epoch) in
    arrival order. With clock="arrival" a claim describes the epoch it arrived
    in; with clock="reported" it describes its reported time (capped at now).
    """

    def __init__(self, claims, t_now, memory, parent, schema_fn, ctx_of, n_slots, use_schema, clock="arrival", echo="exact", span="time"):
        self.T = t_now
        self.ctx_of = ctx_of
        children = defaultdict(list)
        if use_schema:
            for child, par in parent.items():
                children[par].append(child)
        pairs, paired = [], set()
        for par, kids in children.items():
            if len(kids) != 1:
                raise ValueError("one child per parent is assumed")
            pairs.append((par, kids[0]))
            paired.update((par, kids[0]))
        self.singles = [k for k in range(n_slots) if k not in paired]
        self.pairs = pairs
        # A source repeating its own statement adds nothing. Copy availability q
        # is the share of claims already on record for the same fact and
        # reported time, from other sources, that carry the same value.
        rows, seen = [], set()
        record = defaultdict(list)
        by_fact = defaultdict(list)
        raw = []
        for n_claim, claim in enumerate(claims):
            slot, value, issuer, tau, arrived = claim[:5]
            # arrival stamp: a claim can echo only what was on record before it
            # arrived; claims sharing a stamp arrived together
            stamp = claim[5] if len(claim) > 5 else n_claim
            given = claim[6] if len(claim) > 6 else None
            tt = min(max(int(arrived if clock == "arrival" else tau), 1), t_now)
            key = (issuer, slot, value, int(tau))
            if key in seen:
                continue
            seen.add(key)
            prior = [(who, v) for who, v, st in record[(slot, int(tau))] if who != issuer and st < stamp]
            record[(slot, int(tau))].append((issuer, value, stamp))
            by_fact[slot].append((issuer, value, stamp))
            raw.append((slot, value, issuer, tt, int(tau), prior, given, stamp))
        self.cands = []
        for slot, value, issuer, tt, tau, prior, given, own_stamp in raw:
            if echo == "graph" and span == "fact":
                cand = sorted({who for who, v, st in by_fact[slot]
                               if v == value and (st < own_stamp or (st == own_stamp and who < issuer))})
                self.cands.append(cand)
                rows.append((slot, value, issuer, tt, 1.0 if cand else 0.0))
                continue
            if echo == "graph":
                # upstream witnesses whose identical claim was on record when this
                # one arrived; simultaneous claims are ordered by source id so
                # the echo graph stays acyclic
                own = next(st for who, v, st in record[(slot, tau)] if who == issuer and v == value)
                cand = sorted({who for who, v, st in record[(slot, tau)]
                               if who != issuer and v == value and (st < own or (st == own and who < issuer))})
                self.cands.append(cand)
                rows.append((slot, value, issuer, tt, 1.0 if cand else 0.0))
                continue
            if given is not None:
                # echo availability supplied by the data adapter (report containment)
                q = float(given)
            elif echo == "pool":
                others = [(who, v) for who, v, _st in record[(slot, tau)] if who != issuer]
                q = sum(v == value for _w, v in others) / len(others) if others else 0.0
            elif echo == "exact":
                q = 1.0 if any(v == value for _w, v in prior) else 0.0
            else:
                q = sum(v == value for _w, v in prior) / len(prior) if prior else 0.0
            rows.append((slot, value, issuer, tt, q))
        self.rows = rows
        self.echo_graph = echo == "graph"
        claimed = defaultdict(set)
        for slot, value, _i, _t, _q in rows:
            claimed[slot].add(value)
        values = {}
        for k in range(n_slots):
            vals = [memory[k]]
            for v in sorted(claimed[k]):
                if v not in vals:
                    vals.append(v)
            values[k] = vals
        for par, child in pairs:
            for v in list(values[par]):
                img = schema_fn(v)
                if img not in values[child]:
                    values[child].append(img)
        for k in range(n_slots):
            values[k].append(OTHER)
        self.values = values
        self.index = {k: {v: i for i, v in enumerate(vals)} for k, vals in values.items()}
        self.issuers = sorted({r[2] for r in rows})
        self.src_index = {issuer: i for i, issuer in enumerate(self.issuers)}
        if self.echo_graph:
            pair_id = {}
            ent_row, ent_pair = [], []
            for g, ((slot, value, issuer, tt, q), cand) in enumerate(zip(rows, self.cands)):
                for up in cand:
                    key = (issuer, up)
                    if key not in pair_id:
                        pair_id[key] = len(pair_id)
                    ent_row.append(g)
                    ent_pair.append(pair_id[key])
            self.pairs_echo = sorted(pair_id, key=pair_id.get)
            self.pair_src = torch.tensor([self.src_index[a] for a, _b in self.pairs_echo], dtype=torch.int64)
            self.ent_row = torch.tensor(ent_row, dtype=torch.int64)
            self.ent_pair = torch.tensor(ent_pair, dtype=torch.int64)
            self.n_rows = len(rows)
        # singles
        S = len(self.singles)
        self.s_pos = {k: i for i, k in enumerate(self.singles)}
        self.V1 = max([len(values[k]) for k in self.singles] + [2])
        self.s_nv = torch.tensor([len(values[k]) for k in self.singles], dtype=DTYPE)
        self.s_mask = torch.zeros(S, self.V1, dtype=DTYPE)
        for i, k in enumerate(self.singles):
            self.s_mask[i, : len(values[k])] = 1.0
        # pairs
        P = len(pairs)
        self.p_pos = {}
        for i, (par, child) in enumerate(pairs):
            self.p_pos[par] = (i, 0)
            self.p_pos[child] = (i, 1)
        self.Vp = max([len(values[p]) for p, _c in pairs] + [2])
        self.Vc = max([len(values[c]) for _p, c in pairs] + [2])
        self.p_nv = torch.tensor([len(values[p]) for p, _c in pairs], dtype=DTYPE)
        self.c_nv = torch.tensor([len(values[c]) for _p, c in pairs], dtype=DTYPE)
        self.p_mask = torch.zeros(P, self.Vp, dtype=DTYPE)
        self.c_mask = torch.zeros(P, self.Vc, dtype=DTYPE)
        self.fmap = torch.zeros(P, self.Vp, self.Vc, dtype=DTYPE)
        for i, (par, child) in enumerate(pairs):
            self.p_mask[i, : len(values[par])] = 1.0
            self.c_mask[i, : len(values[child])] = 1.0
            for j, v in enumerate(values[par]):
                img = OTHER if v == OTHER else schema_fn(v)
                self.fmap[i, j, self.index[child][img]] = 1.0
        s_rows, pp_rows, pc_rows = [], [], []
        for g, (slot, value, issuer, tt, q) in enumerate(rows):
            item = (None, tt, self.index[slot][value], self.src_index[issuer], ctx_of(slot), len(values[slot]), g, q)
            if slot in self.s_pos:
                s_rows.append((self.s_pos[slot],) + item[1:])
            else:
                pi, role = self.p_pos[slot]
                (pp_rows if role == 0 else pc_rows).append((pi,) + item[1:])
        self.s_claims = self._pack(s_rows)
        self.pp_claims = self._pack(pp_rows)
        self.pc_claims = self._pack(pc_rows)

    @staticmethod
    def _pack(rows):
        if not rows:
            return None
        ints = np.array([r[:7] for r in rows], dtype=np.int64)
        q = np.array([r[7] for r in rows], dtype=np.float64)
        return {
            "g": torch.from_numpy(ints[:, 6]),
            "f": torch.from_numpy(ints[:, 0]),
            "t": torch.from_numpy(ints[:, 1]),
            "v": torch.from_numpy(ints[:, 2]),
            "s": torch.from_numpy(ints[:, 3]),
            "c": torch.from_numpy(ints[:, 4]),
            "nv": torch.from_numpy(ints[:, 5]).to(DTYPE),
            "logq": torch.from_numpy(np.where(q > 0, np.log(np.maximum(q, 1e-300)), LOG0)),
        }


def emissions(claims, n_fact, T, V, log_on, log_off):
    """Sum of per-claim log-likelihoods as a [n_fact, T+1, V] tensor."""
    if claims is None:
        return torch.zeros(n_fact, T + 1, V, dtype=DTYPE)
    cell = claims["f"] * (T + 1) + claims["t"]
    base = torch.zeros(n_fact * (T + 1), dtype=DTYPE).index_add(0, cell, log_off)
    E = torch.zeros(n_fact * (T + 1) * V, dtype=DTYPE).index_add(0, cell * V + claims["v"], log_on - log_off)
    return E.view(n_fact, T + 1, V) + base.view(n_fact, T + 1, 1)


class Model:
    def __init__(self, spec, n_ctx, seed=0):
        self.spec = spec
        self.n_ctx = n_ctx
        self.params = Params(n_ctx, spec.rank, seed, spec.sincere)
        self.arrival = {}   # claim -> correctness frozen at arrival (no-hindsight ablation)
        self.history = []

    # ----- parameters ----------------------------------------------------------
    def _tensors(self, prob, p):
        for issuer in prob.issuers:
            p.ensure(issuer)
        n = len(prob.issuers)
        mk = lambda a, grad=True: torch.tensor(a, dtype=DTYPE, requires_grad=grad)
        up = [p.up.get(pair, 0.0) for pair in getattr(prob, "pairs_echo", [])]
        th = {
            "up": mk(up if up else [0.0]),
            "bias": mk([p.bias[i] for i in prob.issuers]),
            "vec": mk(np.array([p.vec[i] for i in prob.issuers]).reshape(n, -1)),
            "full": mk(np.array([p.full[i] for i in prob.issuers]).reshape(n, -1)),
            "cop": mk([p.cop[i] for i in prob.issuers]),
            "ctx": mk(p.ctx),
            "walk": mk(np.array([_pad(p.walk[i], prob.T) for i in prob.issuers]).reshape(n, prob.T)),
        }
        for name in SCALARS:
            th[name] = mk(getattr(p, name))
        if self.spec.hazard is not None:
            th["lh"] = mk(logit(self.spec.hazard / self.spec.max_hazard), grad=False)
        return th

    def _store(self, prob, th, p):
        for key in ("bias", "cop"):
            arr = th[key].detach().numpy()
            store = p.bias if key == "bias" else p.cop
            for i, issuer in enumerate(prob.issuers):
                store[issuer] = float(arr[i])
        vec, full = th["vec"].detach().numpy(), th["full"].detach().numpy()
        for i, issuer in enumerate(prob.issuers):
            p.vec[issuer] = vec[i].copy()
            p.full[issuer] = full[i].copy()
        p.ctx = th["ctx"].detach().numpy().copy()
        walk = th["walk"].detach().numpy()
        for i, issuer in enumerate(prob.issuers):
            p.walk[issuer] = walk[i].copy()
        if getattr(prob, "echo_graph", False):
            arr = th["up"].detach().numpy()
            for i, pair in enumerate(prob.pairs_echo):
                p.up[pair] = float(arr[i])
        for name in SCALARS:
            if name == "lh" and self.spec.hazard is not None:
                continue
            setattr(p, name, float(th[name].detach()))

    def _free(self, th):
        if self.spec.oracle is not None:
            names = ["lmu", "lom", "cop", "gamma0"]
        else:
            names = ["bias", "ctx", "vec", "beta0", "lmu", "lom", "cop", "gamma0"]
            if self.spec.full_context:
                names.append("full")
        if self.spec.copy and self.spec.echo == "graph":
            names.append("up")
        if self.spec.drift > 0 and self.spec.oracle is None:
            names.append("walk")
        if not self.spec.copy:
            names = [n for n in names if n not in ("cop", "gamma0")]
        if self.spec.rank == 0:
            names = [n for n in names if n not in ("ctx", "vec")]
        if self.spec.learn_scale and self.spec.oracle is None:
            names.append("lsb")
            if self.spec.rank > 0:
                names.append("lsu")
            if self.spec.copy:
                names.append("lsk")
        if self.spec.memory != "joint":
            names = [n for n in names if n != "lmu"]
        if self.spec.hazard is None:
            names.append("lh")
        return [th[n] for n in names]

    def _rho_logit(self, th, s, c, t=None):
        z = th["bias"][s]
        if self.spec.drift > 0 and t is not None:
            z = z + th["walk"][s, t - 1]
        if self.spec.rank > 0:
            vec = th["vec"]
            if self.spec.sincere:
                # context re-ranks sources inside an organization; it does not
                # move the organization's mean credibility off the population's
                vec = vec - vec.mean(0, keepdim=True)
            z = z + (vec[s] * th["ctx"][c]).sum(-1)
        if self.spec.full_context:
            z = z + th["full"][s, c]
        return z

    def _claim_logs(self, th, claims, prob):
        if claims is None:
            return None, None
        if self.spec.oracle is not None:
            rho = torch.tensor([self.spec.oracle(prob.issuers[int(s)], int(c)) for s, c in zip(claims["s"], claims["c"])], dtype=DTYPE)
            lr_on, lr_off = torch.log(rho), torch.log1p(-rho)
        else:
            z = self._rho_logit(th, claims["s"], claims["c"], claims["t"])
            lr_on = torch.nn.functional.logsigmoid(z)
            lr_off = torch.nn.functional.logsigmoid(-z)
        lr_off = lr_off - torch.log(claims["nv"] - 1.0)
        if not self.spec.copy:
            return lr_on, lr_off
        k = th["cop"][claims["s"]]
        log_k = torch.nn.functional.logsigmoid(k)
        log_nk = torch.nn.functional.logsigmoid(-k)
        if getattr(prob, "echo_graph", False):
            log_a = self._echo_avail(th, prob)[claims["g"]]
            copy_term = log_k + log_a
        else:
            copy_term = log_k + claims["logq"]
        on = torch.logaddexp(copy_term, log_nk + lr_on)
        off = torch.logaddexp(copy_term, log_nk + lr_off)
        return on, off

    def _echo_avail(self, th, prob):
        """log of the probability that an echo by this source reproduces this
        claim: the learned share of its upstream witnesses who had made it."""
        cache = getattr(self, "_avail_cache", None)
        if cache is not None and cache[0] is th["up"] and cache[1] == th["up"]._version:
            return cache[2]
        n_src = len(prob.issuers)
        if len(prob.pairs_echo) == 0:
            out = torch.full((prob.n_rows,), LOG0, dtype=DTYPE)
        else:
            e = torch.exp(th["up"].clamp(-30, 30))
            den = torch.zeros(n_src, dtype=DTYPE).index_add(0, prob.pair_src, e)
            share = e / den[prob.pair_src]
            avail = torch.zeros(prob.n_rows, dtype=DTYPE).index_add(0, prob.ent_row, share[prob.ent_pair])
            out = torch.where(avail > 0, torch.log(avail.clamp_min(1e-300)), torch.full_like(avail, LOG0))
        self._avail_cache = (th["up"], th["up"]._version, out)
        return out

    # ----- forward filter ------------------------------------------------------
    def _prior(self, mu, mask, nv):
        if mu is None:
            return mask / mask.sum(-1, keepdim=True)
        first = torch.zeros_like(mask)
        first[:, 0] = 1.0
        return (mu * first + (1 - mu) * (1 - first) / (nv - 1)) * mask

    def _forward(self, th, prob, keep=False, online=False, use_memory=None):
        """Scaled forward filter. online=True detaches the state before the
        current epoch and scores only the current epoch's testimony."""
        if use_memory is None:
            use_memory = self.spec.memory == "joint"
        self._avail_cache = None
        T = prob.T
        h = self.spec.max_hazard * torch.sigmoid(th["lh"])
        mu = torch.sigmoid(th["lmu"]) if use_memory else None
        om = torch.sigmoid(th["lom"])
        total = torch.zeros((), dtype=DTYPE)
        out = {}
        S = len(prob.singles)
        if S:
            on, off = self._claim_logs(th, prob.s_claims, prob)
            E = emissions(prob.s_claims, S, T, prob.V1, on, off)
            mask, nv = prob.s_mask, prob.s_nv.view(S, 1)
            alpha = self._prior(mu, mask, nv)
            seq = [alpha]
            for t in range(1, T + 1):
                if online and t < T:
                    alpha = alpha.detach()
                alpha = ((1 - h) * alpha + h * (1 - alpha) / (nv - 1)) * mask
                e = E[:, t, :].masked_fill(mask == 0, LOG0)
                m = e.max(-1, keepdim=True).values
                w = alpha * torch.exp(e - m)
                s = w.sum(-1, keepdim=True)
                alpha = w / s
                if not online or t == T:
                    total = total + (torch.log(s) + m).sum()
                    if keep:
                        out["single_ll"] = out.get("single_ll", 0) + (torch.log(s) + m).view(-1)
                if keep:
                    seq.append(alpha)
            out["single"] = alpha
            if keep:
                out["single_seq"] = seq
        P = len(prob.pairs)
        if P:
            on_p, off_p = self._claim_logs(th, prob.pp_claims, prob)
            on_c, off_c = self._claim_logs(th, prob.pc_claims, prob)
            Ep = emissions(prob.pp_claims, P, T, prob.Vp, on_p, off_p)
            Ec = emissions(prob.pc_claims, P, T, prob.Vc, on_c, off_c)
            pm, cm = prob.p_mask, prob.c_mask
            nvp, nvc = prob.p_nv.view(P, 1), prob.c_nv.view(P, 1, 1)
            joint = pm[:, :, None] * cm[:, None, :]
            alpha = self._prior(mu, pm, nvp)[:, :, None] * self._prior(mu, cm, prob.c_nv.view(P, 1))[:, None, :]
            # after a parent change the child follows the schema unless overridden
            G = om / (nvc - 1) * cm[:, None, :] + prob.fmap * ((1 - om) - om / (nvc - 1))
            seq = [alpha]
            for t in range(1, T + 1):
                if online and t < T:
                    alpha = alpha.detach()
                Pp = alpha.sum(-1)
                child = ((1 - h) * alpha + h * (Pp[:, :, None] - alpha) / (nvc - 1)) * joint
                M = h * (1 - Pp) / (nvp - 1) * pm
                alpha = ((1 - h) * child + M[:, :, None] * G) * joint
                e = Ep[:, t, :].masked_fill(pm == 0, LOG0)[:, :, None] + Ec[:, t, :].masked_fill(cm == 0, LOG0)[:, None, :]
                m = e.amax(dim=(1, 2), keepdim=True)
                w = alpha * torch.exp(e - m)
                s = w.sum(dim=(1, 2), keepdim=True)
                alpha = w / s
                if not online or t == T:
                    total = total + (torch.log(s) + m).sum()
                    if keep:
                        out["pair_ll"] = out.get("pair_ll", 0) + (torch.log(s) + m).view(-1)
                if keep:
                    seq.append(alpha)
            out["pair"] = alpha
            if keep:
                out["pair_seq"] = seq
        out["loglik"] = total
        return out

    def _log_prior(self, th):
        """Hierarchical Gaussian priors. The spread of each population is a
        fitted parameter, so a society of near-perfect and near-useless sources
        is not shrunk toward a middling credibility."""
        def gauss(x, center, log_scale):
            if not self.spec.learn_scale:
                log_scale = torch.tensor(math.log(self.spec.prior_scale), dtype=DTYPE)
            return (-0.5 * ((x - center) ** 2) * torch.exp(-2 * log_scale) - log_scale).sum()
        center = torch.nn.functional.softplus(th["beta0"]) if self.spec.sincere else th["beta0"]
        lp = gauss(th["bias"], center, th["lsb"])
        if self.spec.copy:
            lp = lp + gauss(th["cop"], th["gamma0"], th["lsk"])
            if self.spec.echo == "graph":
                lp = lp + gauss(th["up"], 0.0, th["lsk"])
        if self.spec.rank > 0:
            lp = lp + gauss(th["vec"], 0.0, th["lsu"]) - 0.5 * (th["ctx"] ** 2).sum()
        if self.spec.full_context:
            lp = lp + gauss(th["full"], 0.0, th["lsu"])
        if self.spec.drift > 0:
            w = th["walk"]
            steps = torch.cat([w[:, :1], w[:, 1:] - w[:, :-1]], dim=1)
            lp = lp - 0.5 * (steps ** 2).sum() / self.spec.drift ** 2
        r = self.spec.rate_prior
        if r > 0:
            for name in ("lh", "lmu", "lom"):
                lp = lp + r * (torch.nn.functional.logsigmoid(th[name]) + torch.nn.functional.logsigmoid(-th[name]))
        return lp

    # ----- no-hindsight ablation ---------------------------------------------
    def _arrival_objective(self, th, prob):
        """Each past claim keeps the correctness it was given on arrival."""
        rows, r = [], []
        for (issuer, slot, value, tt), resp in self.arrival.items():
            if issuer in prob.src_index:
                rows.append((prob.src_index[issuer], prob.ctx_of(slot)))
                r.append(resp)
        if not rows:
            return torch.zeros((), dtype=DTYPE)
        arr = torch.from_numpy(np.array(rows, dtype=np.int64))
        z = self._rho_logit(th, arr[:, 0], arr[:, 1])
        r = torch.tensor(r, dtype=DTYPE)
        return (r * torch.nn.functional.logsigmoid(z) + (1 - r) * torch.nn.functional.logsigmoid(-z)).sum()

    def _objective(self, th, prob):
        if self.spec.hindsight or self.spec.oracle is not None:
            return self._forward(th, prob)["loglik"] + self._log_prior(th)
        return self._arrival_objective(th, prob) + self._forward(th, prob, online=True)["loglik"] + self._log_prior(th)

    def _optimize(self, prob, p, steps):
        th = self._tensors(prob, p)
        free = self._free(th)
        if self.spec.optimizer == "lbfgs":
            opt = torch.optim.LBFGS(free, lr=1.0, max_iter=steps, history_size=20,
                                    tolerance_grad=1e-6, tolerance_change=1e-9,
                                    line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                loss = -self._objective(th, prob)
                loss.backward()
                return loss

            for _ in range(3):
                opt.step(closure)
        else:
            opt = torch.optim.Adam(free, lr=self.spec.lr)
            for _ in range(steps):
                opt.zero_grad()
                loss = -self._objective(th, prob)
                loss.backward()
                opt.step()
        with torch.no_grad():
            value = float(self._objective(th, prob))
        if not math.isfinite(value):
            value = -float("inf")
        return th, value

    def fit(self, prob, first):
        """MAP by multi-start: the warm start from the last epoch and fresh
        starts compete on the same objective; the highest one is kept."""
        steps = self.spec.steps_first if first else self.spec.steps
        best = None
        candidates = [self.params]
        if self.spec.oracle is None:
            candidates += [self.params.fresh(1000 * prob.T + r, self.spec.init_spread) for r in range(self.spec.restarts)]
        for i, p in enumerate(candidates):
            th, value = self._optimize(prob, p, steps if i == 0 or first else self.spec.steps_first)
            if best is None or value > best[1]:
                best = (th, value, p)
        th, value, p = best
        if self.spec.memory == "decision":
            self._fit_memory(th, prob)
        self._store(prob, th, p)
        self.params = p
        self.history.append({"T": prob.T, "objective": value, **self.scalars()})
        return th

    def _fit_memory(self, th, prob):
        """Accuracy of the stored memory, with everything else held fixed."""
        lmu = th["lmu"].detach().clone().requires_grad_(True)
        frozen = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in th.items()}
        frozen["lmu"] = lmu
        opt = torch.optim.LBFGS([lmu], lr=1.0, max_iter=30, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = -self._forward(frozen, prob, use_memory=True)["loglik"]
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            th["lmu"].copy_(lmu.detach())

    # ----- decision --------------------------------------------------------------
    def decide(self, prob, th):
        use_memory = self.spec.memory != "none"
        with torch.no_grad():
            out = self._forward(th, prob, keep=not self.spec.hindsight, use_memory=use_memory)
        answers, post = {}, {}
        if prob.singles:
            a = out["single"].numpy()
            for i, k in enumerate(prob.singles):
                vals = prob.values[k]
                answers[k], post[k] = _mode(vals, a[i, : len(vals)])
        if prob.pairs:
            a = out["pair"].numpy()
            for i, (par, child) in enumerate(prob.pairs):
                vp, vc = prob.values[par], prob.values[child]
                answers[par], post[par] = _mode(vp, a[i].sum(-1)[: len(vp)])
                answers[child], post[child] = _mode(vc, a[i].sum(-2)[: len(vc)])
        if not self.spec.hindsight:
            self._freeze_arrivals(prob, out)
        return answers, post

    def posterior(self, prob, th):
        """Filtered marginal distribution over named values for every fact."""
        with torch.no_grad():
            out = self._forward(th, prob, use_memory=self.spec.memory != "none")
        dist = {}
        if prob.singles:
            a = out["single"].numpy()
            for i, k in enumerate(prob.singles):
                dist[k] = dict(zip(prob.values[k], a[i, : len(prob.values[k])]))
        if prob.pairs:
            a = out["pair"].numpy()
            for i, (par, child) in enumerate(prob.pairs):
                dist[par] = dict(zip(prob.values[par], a[i].sum(-1)[: len(prob.values[par])]))
                dist[child] = dict(zip(prob.values[child], a[i].sum(-2)[: len(prob.values[child])]))
        return dist

    def _freeze_arrivals(self, prob, out):
        T = prob.T
        for slot, value, issuer, tt, _q in prob.rows:
            key = (issuer, slot, value, tt)
            if key in self.arrival:
                continue
            vi = prob.index[slot][value]
            if slot in prob.s_pos:
                dist = out["single_seq"][T][prob.s_pos[slot]].numpy()
            else:
                pi, role = prob.p_pos[slot]
                joint = out["pair_seq"][T][pi].numpy()
                dist = joint.sum(-1) if role == 0 else joint.sum(-2)
            self.arrival[key] = float(min(max(dist[vi], 1e-6), 1 - 1e-6))

    # ----- read-outs ------------------------------------------------------------
    def credibility(self, issuers, n_ctx):
        p = self.params
        out = np.zeros((len(issuers), n_ctx))
        mean_vec = np.mean([p.vec[i] for i in p.vec], axis=0) if (self.spec.sincere and p.vec) else 0.0
        for i, issuer in enumerate(issuers):
            p.ensure(issuer)
            for c in range(n_ctx):
                z = p.bias[issuer]
                if self.spec.drift > 0 and len(p.walk.get(issuer, [])):
                    z += float(p.walk[issuer][-1])
                if self.spec.rank > 0:
                    z += float(np.dot(p.vec[issuer] - mean_vec, p.ctx[c]))
                if self.spec.full_context:
                    z += p.full[issuer][c]
                out[i, c] = sig(z)
        return out

    def copy_propensity(self, issuers):
        p = self.params
        return [sig(p.cop.get(i, p.gamma0)) for i in issuers]

    def scalars(self):
        p = self.params
        h = self.spec.hazard if self.spec.hazard is not None else self.spec.max_hazard * sig(p.lh)
        return {"h": h, "mu": sig(p.lmu), "omega": sig(p.lom), "base": sig(p.center()), "copy": sig(p.gamma0),
                "spread": math.exp(p.lsb)}


def _pad(arr, T):
    """Per-epoch offsets carried to T epochs; a new epoch starts at the last value."""
    arr = np.asarray(arr, dtype=float)
    if len(arr) >= T:
        return arr[:T]
    last = arr[-1] if len(arr) else 0.0
    return np.concatenate([arr, np.full(T - len(arr), last)])


def _mode(values, probs):
    best, best_p = None, -1.0
    for v, q in zip(values, probs):
        if isinstance(v, str) and v.startswith("<"):
            continue
        if q > best_p:
            best, best_p = v, float(q)
    return best, best_p
