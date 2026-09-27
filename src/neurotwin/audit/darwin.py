"""Darwin-Cage search: evolve feature programs from noise, rewarded only by the structure they find in a residual.

Port of the team lead's Darwin-Cage search (s6e9-honest-ceiling, ``honest_ceiling/darwin.py``, Apache-2.0,
Copyright 2026 Francisco Angulo de Lafuente; see ``repo/NOTICE``). Engine, fitness and the evolutionary loop are
the original ones. A *program* is a composition of up to four atoms (a categorical column, or an integer transform
of a numeric column). Its *key* is the joint value of its atoms. Fitness is out-of-fold: the shrunk group mean of
the residual is fit on all but one search fold and correlated with the residual on the held-out search fold,
rotating over the search folds. The confirmation fold is never seen by the search.

Changes with respect to the original (all deliberate, all tested in ``repo/tests/test_audit.py``):

1. **The alphabet is a parameter** (:class:`Alphabet`) instead of the 13 S6E9 columns. For NeuroTrajectory the
   caller restricts it to information available at prediction time. Numeric atoms are ``raw``, ``div``
   (fixed-width bins of an integer-scaled column) and a new ``qbin`` (quantile bins). The original ``mod``,
   ``digit`` and ``pshift`` atoms, which hunted synthetic-generator artefacts in S6E9, are still implemented but
   are off unless the alphabet enables them.
2. **Cluster-aware confirmation.** The original rule counts a program as a discovery when its confirmation
   correlation exceeds ``z / sqrt(n_confirm_rows)``, which assumes independent rows. Residual rows of a forecast
   are clustered (all rows of one chemical share its errors), so that rule alone is anti-conservative. A program is
   confirmed only if it passes the original rule **and** a sign-flip randomisation test over clusters
   (``groups``, e.g. chemicals) at the same one-sided level ``P(Z > z)``. The report also carries a max-T
   family-wise p-value over the whole hall of fame, computed with the same sign flips.
3. :func:`plant_residual` adds a continuous structure of known size to a residual (the original :func:`plant`,
   which draws Bernoulli labels around a probability, is kept unchanged).
4. Atom codes are cached and the per-fold sums are computed in one pass; keys induce the same partition of the
   rows as the original mixed-radix construction and the fitness/confirmation statistics are the original ones.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

MODS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 19, 23, 25, 29, 31, 37, 41, 50, 64, 97, 100, 101, 128, 250,
        256, 500, 1000, 1024, 2500, 5000, 10000]
DIVS = [2, 3, 5, 7, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000, 25000]
DIGITS = [1, 10, 100, 1000, 10000, 100000]
QBINS = [2, 3, 4, 5, 6, 8, 10, 16, 20]


@dataclass(frozen=True)
class Alphabet:
    """What the search may look at. ``cats``: categorical columns; ``nums``: integer-scaled numeric columns."""

    cats: tuple[str, ...]
    nums: tuple[str, ...] = ()
    ops: tuple[str, ...] = ("raw", "div", "qbin")
    p_cat: float = 0.25                 # probability that a fresh atom is categorical (original: 0.25)
    divs: tuple[int, ...] = tuple(DIVS)
    qbins: tuple[int, ...] = tuple(QBINS)
    mods: tuple[int, ...] = tuple(MODS)
    digits: tuple[int, ...] = tuple(DIGITS)

    def params(self, op: str) -> tuple[int, ...]:
        return {"mod": self.mods, "div": self.divs, "digit": self.digits, "qbin": self.qbins}[op]


def integer_view(df: pd.DataFrame, cats: list[str], num_scale: dict[str, float]) -> dict[str, np.ndarray]:
    """Integer view of a table (the original ``integer_columns`` with the columns as arguments)."""
    D = {c: pd.factorize(df[c].astype(str))[0].astype(np.int64) for c in cats}
    for c, s in num_scale.items():
        D[c] = np.round(df[c].to_numpy(np.float64) * s).astype(np.int64)
    return D


def random_atom(rng: np.random.Generator, alphabet: Alphabet) -> tuple:
    if alphabet.cats and (not alphabet.nums or rng.random() < alphabet.p_cat):
        return ("cat", str(rng.choice(list(alphabet.cats))))
    col = str(rng.choice(list(alphabet.nums)))
    op = str(rng.choice(list(alphabet.ops)))
    if op in ("mod", "div", "digit", "qbin"):
        return (op, col, int(rng.choice(alphabet.params(op))))
    if op == "pshift":
        return ("pshift", col, int(rng.integers(1, 7)), int(rng.choice([2, 3, 4, 8])))
    return ("raw", col)


def atom_values(atom: tuple, D: dict[str, np.ndarray]) -> np.ndarray:
    kind, col = atom[0], atom[1]
    v = D[col]
    if kind in ("cat", "raw"):
        return v
    if kind == "mod":
        return v % atom[2]
    if kind == "div":
        return v // atom[2]
    if kind == "digit":
        return (v // atom[2]) % 10
    if kind == "pshift":
        return (v >> atom[2]) % atom[3]
    if kind == "qbin":
        q = int(atom[2])
        edges = np.unique(np.quantile(v, np.linspace(0.0, 1.0, q + 1)[1:-1]))
        return np.searchsorted(edges, v, side="right").astype(np.int64)
    raise ValueError(kind)


class AtomCache:
    """Factorised codes per atom (label-free, so caching them cannot leak anything). Bounded LRU, int32 codes."""

    def __init__(self, D: dict[str, np.ndarray], max_items: int = 64):
        self.D = D
        self.max_items = max_items
        self._c: dict[tuple, np.ndarray] = {}

    def codes(self, atom: tuple) -> np.ndarray:
        a = tuple(atom)
        if a in self._c:
            v = self._c.pop(a)
        else:
            v = pd.factorize(atom_values(a, self.D))[0].astype(np.int32)
            while len(self._c) >= self.max_items:
                self._c.pop(next(iter(self._c)))
        self._c[a] = v
        return v


def program_key(prog: list[tuple], D: dict[str, np.ndarray], cache: AtomCache | None = None) -> np.ndarray:
    """Joint key of the program's atoms (mixed radix over factorised values), compacted to 0..n_keys-1.

    Same partition of the rows as the original; when the radix product is small the compaction is a sorted remap
    instead of ``pd.factorize`` (codes may be numbered differently, which no statistic depends on)."""
    key = np.zeros(len(next(iter(D.values()))), dtype=np.int64)
    P = 1
    for atom in prog:
        codes = cache.codes(atom) if cache is not None else pd.factorize(atom_values(tuple(atom), D))[0]
        m = int(codes.max()) + 1
        key = key * m + codes.astype(np.int64)
        P *= m
        if P > 2 ** 62 or key.max() > 2 ** 62:
            key = pd.factorize(key)[0].astype(np.int64)
            P = int(key.max()) + 1
    if P <= 1 << 22:
        used = np.bincount(key, minlength=P) > 0
        return (np.cumsum(used) - 1)[key].astype(np.int64)
    return pd.factorize(key)[0].astype(np.int64)


def prog_id(prog) -> str:
    return hashlib.md5(json.dumps([list(a) for a in prog]).encode()).hexdigest()[:12]


def mutate(prog, rng, alphabet: Alphabet):
    prog = [tuple(a) for a in prog]
    t = rng.random()
    if t < 0.3 and len(prog) < 4:
        prog.append(random_atom(rng, alphabet))
    elif t < 0.55 and len(prog) > 1:
        prog.pop(int(rng.integers(len(prog))))
    else:
        i = int(rng.integers(len(prog)))
        a = list(prog[i])
        if a[0] in ("mod", "div", "digit", "qbin") and rng.random() < 0.6:
            a[2] = int(rng.choice(alphabet.params(a[0])))
            prog[i] = tuple(a)
        else:
            prog[i] = random_atom(rng, alphabet)
    return prog


def crossover(p1, p2, rng):
    child = list(p1[: int(rng.integers(1, len(p1) + 1))]) + list(p2[int(rng.integers(0, len(p2) + 1)):])
    return child[:4] if child else list(p1)


def uses_atoms(program: list, atoms: list) -> bool:
    """True if every atom of ``atoms`` (compared as lists) appears in ``program``."""
    have = {json.dumps(list(a)) for a in program}
    return all(json.dumps(list(a)) in have for a in atoms)


@dataclass
class DarwinSearch:
    D: dict[str, np.ndarray]
    r: np.ndarray
    fold: np.ndarray
    alphabet: Alphabet
    confirm_fold: int
    groups: np.ndarray | None = None      # cluster id per row (e.g. chemical); enables the sign-flip test
    z: float = 3.0
    threshold: float | None = None
    alpha: float = 5.0
    max_keys: int = 200_000
    hall_min_fit: float = 0.006
    n_perm: int = 4000
    perm_seed: int = 0
    cache: AtomCache | None = None       # may be shared between searches over the same D (label-free)
    seen: dict = field(default_factory=dict)
    hall: dict = field(default_factory=dict)

    def __post_init__(self):
        self.r = np.asarray(self.r, np.float64)
        self.fold = np.asarray(self.fold)
        if self.cache is None:
            self.cache = AtomCache(self.D)
        ho = self.fold == self.confirm_fold
        self.n_confirm = int(ho.sum())
        if self.threshold is None:
            self.threshold = self.z / float(np.sqrt(self.n_confirm))
        self.p_one = 0.5 * erfc(self.z / sqrt(2))
        folds = [int(k) for k in np.unique(self.fold)]
        self._F = len(folds)
        self._fidx = np.searchsorted(folds, self.fold).astype(np.int64)
        self._ci = folds.index(int(self.confirm_fold))
        self._si = [i for i, k in enumerate(folds) if k != self.confirm_fold]
        self._search_folds = [folds[i] for i in self._si]
        self._rows = [np.where(self._fidx == i)[0] for i in range(self._F)]
        self._rr = [self.r[ix] for ix in self._rows]
        self._r_conf = self.r[ho]
        self._maxT = None
        if self.groups is not None:
            g = pd.factorize(np.asarray(self.groups)[ho])[0]
            self._g_conf = g
            G = int(g.max()) + 1
            rng = np.random.default_rng(self.perm_seed)
            self._S = rng.choice([-1.0, 1.0], size=(self.n_perm, G))
            R = np.bincount(g, weights=self._r_conf, minlength=G)
            n = float(self.n_confirm)
            self._mr_b = (self._S @ R) / n                          # permuted residual means
            self._sd_r_b = np.sqrt(np.maximum((self._r_conf ** 2).sum() / n - self._mr_b ** 2, 1e-300))
            self._maxT = np.full(self.n_perm, -np.inf)
            self.n_groups_confirm = G

    # -------------------------------------------------------------- fitness (original statistic)
    def _stats(self, key):
        """Residual sums and counts per (key, fold); every fit set below is a sum of fold columns."""
        nk = int(key.max()) + 1
        kk = key * self._F + self._fidx
        s = np.bincount(kk, weights=self.r, minlength=nk * self._F).reshape(nk, self._F)
        c = np.bincount(kk, minlength=nk * self._F).reshape(nk, self._F).astype(np.float64)
        return s, c

    def fitness(self, key: np.ndarray, stats=None) -> tuple[float, float]:
        s, c = stats if stats is not None else self._stats(key)
        s_all, c_all = s[:, self._si].sum(1), c[:, self._si].sum(1)
        cors = []
        for i in self._si:     # fit on the other search folds, correlate on search fold i
            est = ((s_all - s[:, i]) / (c_all - c[:, i] + self.alpha))[key[self._rows[i]]]
            cors.append(0.0 if est.std() < 1e-12 else float(np.corrcoef(est, self._rr[i])[0, 1]))
        return float(np.mean(cors)), float(np.min(cors))

    # -------------------------------------------------------------- confirmation (original + cluster test)
    def confirm_detail(self, key: np.ndarray, stats=None) -> dict:
        s, c = stats if stats is not None else self._stats(key)
        est = (s[:, self._si].sum(1) / (c[:, self._si].sum(1) + self.alpha))[key[self._rows[self._ci]]]
        if est.std() < 1e-12:
            return {"confirm": 0.0, "p_signflip": 1.0, "est": est}
        corr = float(np.corrcoef(est, self._r_conf)[0, 1])
        out = {"confirm": corr, "p_signflip": None, "est": est}
        if self.groups is not None:
            n = float(self.n_confirm)
            A = np.bincount(self._g_conf, weights=est * self._r_conf, minlength=self._S.shape[1])
            cov_b = (self._S @ A) / n - est.mean() * self._mr_b
            corr_b = cov_b / (est.std() * self._sd_r_b)
            out["p_signflip"] = float((1 + np.sum(corr_b >= corr)) / (self.n_perm + 1))
            out["null_sd"] = float(corr_b.std())
            out["corr_b"] = corr_b
        return out

    def confirm(self, key: np.ndarray) -> float:
        return self.confirm_detail(key)["confirm"]

    def is_confirmed(self, h: dict) -> bool:
        ok = h["confirm"] > self.threshold
        if self.groups is not None:
            ok = ok and h["p_signflip"] is not None and h["p_signflip"] < self.p_one
        return bool(ok)

    def null(self, reps: int = 20, n_groups: int = 500, seed: int = 0) -> tuple[float, float]:
        rng = np.random.default_rng(seed)
        v = [self.fitness(pd.factorize(rng.integers(0, n_groups, len(self.r)))[0])[0] for _ in range(reps)]
        return float(np.mean(v)), float(np.std(v))

    def evaluate(self, prog) -> tuple[float, float, int]:
        pid = prog_id(prog)
        if pid not in self.seen:
            key = program_key(prog, self.D, self.cache)
            nk = int(key.max()) + 1
            st = None if nk > self.max_keys else self._stats(key)
            f = (-1.0, -1.0) if st is None else self.fitness(key, st)
            self.seen[pid] = (f[0], f[1], nk)
            if f[0] > self.hall_min_fit and f[1] > 0:
                det = self.confirm_detail(key, st)
                h = dict(program=[list(a) for a in prog], fit_mean=f[0], fit_min=f[1], n_keys=nk,
                         confirm=det["confirm"], p_signflip=det["p_signflip"], null_sd=det.get("null_sd"))
                if self._maxT is not None and "corr_b" in det:
                    np.maximum(self._maxT, det["corr_b"], out=self._maxT)
                self.hall[pid] = h
        return self.seen[pid]

    def run(self, seconds: float | None = None, max_evals: int | None = None, pop: int = 80, seed: int = 2026,
            log_every: int = 25, log_path: str | Path | None = None, stop_on_confirm: bool = False,
            init: list | None = None, stop_when=None) -> dict:
        """Evolve until a budget is spent. ``stop_when(confirmed)`` may end the run early (e.g. once a planted
        structure is recovered: recovery within the budget is unchanged, only the remaining budget is saved)."""
        rng = np.random.default_rng(seed)
        population = [[random_atom(rng, self.alphabet)] for _ in range(pop)]
        if init:
            population[: len(init)] = [list(map(tuple, p)) for p in init]
        t0, gen = time.time(), 0
        history = []
        while True:
            scored = sorted(((*self.evaluate(p), p) for p in population), key=lambda t: t[0], reverse=True)
            gen += 1
            confirmed = [h for h in self.hall.values() if self.is_confirmed(h)]
            if gen % log_every == 0 or confirmed:
                rec = dict(gen=gen, evaluated=len(self.seen), sec=round(time.time() - t0, 1), best_fit=scored[0][0],
                           best_prog=[list(a) for a in scored[0][3]], hall=len(self.hall), confirmed=len(confirmed))
                history.append(rec)
                if log_path:
                    with open(log_path, "a") as f:
                        f.write(json.dumps(rec) + "\n")
            done = (seconds is not None and time.time() - t0 > seconds) or \
                   (max_evals is not None and len(self.seen) >= max_evals) or (stop_on_confirm and bool(confirmed)) or \
                   (stop_when is not None and bool(stop_when(confirmed)))
            if done:
                break
            elite = [s[3] for s in scored[: max(6, pop // 5)]]
            nxt = list(elite)
            while len(nxt) < pop:
                u = rng.random()
                if u < 0.45:
                    nxt.append(mutate(elite[int(rng.integers(len(elite)))], rng, self.alphabet))
                elif u < 0.8:
                    nxt.append(crossover(elite[int(rng.integers(len(elite)))], elite[int(rng.integers(len(elite)))], rng))
                else:
                    nxt.append([random_atom(rng, self.alphabet) for _ in range(int(rng.integers(1, 4)))])
            population = nxt
        if self._maxT is not None:
            for h in self.hall.values():
                h["p_fwer_maxT"] = float((1 + np.sum(self._maxT >= h["confirm"])) / (self.n_perm + 1))
        for h in self.hall.values():
            h["confirmed"] = self.is_confirmed(h)
            h["passes_original_rule"] = bool(h["confirm"] > self.threshold)
        hall = sorted(self.hall.values(), key=lambda h: -h["confirm"])
        return dict(evaluated=len(self.seen), generations=gen, seconds=round(time.time() - t0, 1),
                    threshold=self.threshold, z=self.z, p_one_sided=self.p_one, n_confirm_rows=self.n_confirm,
                    n_confirm_groups=getattr(self, "n_groups_confirm", None), n_hall=len(self.hall),
                    familywise_fp_expected=len(self.hall) * self.p_one,
                    n_pass_original_rule=int(sum(h["passes_original_rule"] for h in hall)),
                    confirmed=[h for h in hall if h["confirmed"]], hall=hall[:100], history=history)


def plant(p: np.ndarray, D: dict[str, np.ndarray], program: list[tuple], effect: float, seed: int = 0):
    """Original S6E9 plant: synthetic labels y' ~ Bernoulli(p + effect * s(key)), with s = +-1 drawn per key value.

    Returns (y_planted, shift) so that r' = y' - p contains a structure of known size living on `program`.
    """
    rng = np.random.default_rng(seed)
    key = program_key(program, D)
    sign = rng.choice([-1.0, 1.0], size=int(key.max()) + 1)
    shift = effect * sign[key]
    return rng.binomial(1, np.clip(p + shift, 1e-4, 1 - 1e-4)).astype(np.int8), shift


def plant_residual(r: np.ndarray, D: dict[str, np.ndarray], program: list[tuple], effect: float, seed: int = 0,
                   cache: AtomCache | None = None):
    """Continuous analogue of :func:`plant`: r' = r + effect * s(key), s = +-1 drawn per key value.

    ``effect`` is in the units of ``r`` (for NeuroTrajectory: vehicle robust SD). Returns (r_planted, shift).
    """
    rng = np.random.default_rng(seed)
    key = program_key(program, D, cache)
    sign = rng.choice([-1.0, 1.0], size=int(key.max()) + 1)
    shift = effect * sign[key]
    return np.asarray(r, np.float64) + shift, shift
