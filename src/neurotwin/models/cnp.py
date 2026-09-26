"""NeuroTrajectory: an amortised few-shot model of dose x DIV network-formation trajectories.

A conditional neural process with an attentive (dose-local) path:

    context  C = {(log c_i, Y_i, M_i)}   wells of one chemical at the measured concentrations
    query    log c*                        any concentration (measured or not)
    output   mu(log c*), sigma(log c*)    in R^{4 x 17}: the whole DIV 5-12 trajectory

    h_i      = phi([log c_i, Y_i * M_i, M_i])                  per-well encoding
    r_glob   = mean_i h_i  (learned 'empty' vector if |C| = 0)  chemical-level summary
    r_loc    = MHA(query = e(log c*), keys = e(log c_i), values = h_i)   dose-local summary
    (mu, s)  = psi([r_glob, r_loc, e(log c*)]),  sigma = 0.02 + softplus(s)

It is meta-trained across chemicals: every episode samples a chemical, a random
number k of its concentration levels as context and scores the Gaussian NLL of
all its wells (unmasked entries only). At test time the same network answers
'what happens at the concentrations I have not measured?' for a chemical it has
never seen, and its predictive variance drives next-dose selection.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .trajectory import ND, NF, ChemTask, interp_rows, level_means

D_OUT = ND * NF


def _mlp(i, h, o, n=2):
    layers, d = [], i
    for _ in range(n):
        layers += [nn.Linear(d, h), nn.GELU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class DoseEmbed(nn.Module):
    """Fourier features of standardised log10 concentration."""

    def __init__(self, n=8, out=64):
        super().__init__()
        self.register_buffer("freq", 2.0 ** torch.arange(n, dtype=torch.float32) * 0.5)
        self.lin = nn.Linear(2 * n + 1, out)

    def forward(self, x):                   # x: (..., 1) in log10 µM
        z = (x + 0.5) / 1.5
        ang = z * self.freq
        return self.lin(torch.cat([z, torch.sin(ang), torch.cos(ang)], -1))


class NeuroTrajectoryCNP(nn.Module):
    """use_interp: the decoder also receives the piecewise-linear interpolation (in log dose) of the
    context level means at the query, plus the distance to the nearest context level, and predicts
    mu = sigmoid(gate) * interp + delta. With no context interp = 0 and the model is a pure prior."""

    def __init__(self, d=128, heads=4, use_interp=False, n_freq=8, use_attention=True):
        super().__init__()
        self.use_interp = use_interp
        self.use_attention = use_attention
        self.ip = nn.Linear(D_OUT + 1, 64) if use_interp else None
        self.emb = DoseEmbed(n=n_freq, out=64)   # n_freq Fourier frequencies 0.5*2^i: fewer -> smoother in dose
        self.enc = _mlp(64 + 2 * D_OUT, 256, d, n=2)
        self.empty_g = nn.Parameter(torch.zeros(d))
        self.empty_l = nn.Parameter(torch.zeros(d))
        self.q = nn.Linear(64, d)
        self.k = nn.Linear(64, d)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.dec = _mlp(2 * d + 64 + (64 if use_interp else 0), 256, (3 if use_interp else 2) * D_OUT, n=3)

    def forward(self, cx, cy, cm, cmask, qx, qi=None):
        """cx (B,N,1) cy/cm (B,N,68) cmask (B,N) bool valid context; qx (B,Q,1)."""
        B = qx.shape[0]
        ex = self.emb(cx)
        h = self.enc(torch.cat([ex, cy * cm, cm], -1))                 # (B,N,d)
        w = cmask.float().unsqueeze(-1)
        n = w.sum(1)
        has = (n > 0).float()
        r_g = (h * w).sum(1) / n.clamp(min=1)
        r_g = has * r_g + (1 - has) * self.empty_g                    # (B,d)
        eq = self.emb(qx)
        if cmask.any() and self.use_attention:
            kpm = ~cmask
            kpm_safe = kpm.clone()
            kpm_safe[kpm.all(1)] = False                                 # avoid all-masked rows
            r_l, _ = self.att(self.q(eq), self.k(ex), h, key_padding_mask=kpm_safe)
            r_l = has.unsqueeze(1) * r_l + (1 - has.unsqueeze(1)) * self.empty_l
        else:
            r_l = self.empty_l.expand(B, qx.shape[1], -1)
        parts = [r_g.unsqueeze(1).expand(-1, qx.shape[1], -1), r_l, eq]
        if self.use_interp:
            parts.append(self.ip(qi))
        out = self.dec(torch.cat(parts, -1))
        if self.use_interp:
            delta, s, gate = out.chunk(3, -1)
            mu = torch.sigmoid(gate) * qi[..., :D_OUT] + delta
        else:
            mu, s = out.chunk(2, -1)
        return mu, 0.02 + nn.functional.softplus(s)


# ------------------------------------------------------------------ batching
def _pad(arrs, n, fill=0.0):
    out = np.full((len(arrs), n) + arrs[0].shape[1:], fill, np.float32)
    for i, a in enumerate(arrs):
        out[i, : len(a)] = a
    return out


def interp_features(t: ChemTask, c: np.ndarray, q_logc: np.ndarray) -> np.ndarray:
    """(Q, 69): context level means interpolated at the queries (flat outside) + distance to the
    nearest context level (5.0 when there is no context)."""
    out = np.zeros((len(q_logc), D_OUT + 1), np.float32)
    if not c.any():
        out[:, -1] = 5.0
        return out
    lv, mu, ok = level_means(t, c)
    out[:, :D_OUT] = interp_rows(lv, mu, ok, q_logc).reshape(len(q_logc), -1)
    out[:, -1] = np.abs(q_logc[:, None] - lv[None, :]).min(1)
    return out


def make_batch(tasks: list[ChemTask], ctx_sel: list[np.ndarray], tgt_sel: list[np.ndarray], device):
    cxs, cys, cms, qxs, qys, qms, qis = [], [], [], [], [], [], []
    for t, c, q in zip(tasks, ctx_sel, tgt_sel):
        qis.append(interp_features(t, c, t.logc[q]))
        cxs.append(t.logc[c][:, None]); cys.append(t.y[c].reshape(-1, D_OUT)); cms.append(t.m[c].reshape(-1, D_OUT))
        qxs.append(t.logc[q][:, None]); qys.append(t.y[q].reshape(-1, D_OUT)); qms.append(t.m[q].reshape(-1, D_OUT))
    N = max(1, max(len(a) for a in cxs))
    Q = max(len(a) for a in qxs)
    cmask = np.zeros((len(tasks), N), bool)
    qmask = np.zeros((len(tasks), Q), bool)
    for i, (a, b) in enumerate(zip(cxs, qxs)):
        cmask[i, : len(a)] = True
        qmask[i, : len(b)] = True
    cx = _pad([a if len(a) else np.zeros((0, 1), np.float32) for a in cxs], N)
    cy = _pad([a if len(a) else np.zeros((0, D_OUT), np.float32) for a in cys], N)
    cm = _pad([a.astype(np.float32) if len(a) else np.zeros((0, D_OUT), np.float32) for a in cms], N)
    T = lambda a: torch.from_numpy(a).to(device)
    return (T(cx), T(cy), T(cm), T(cmask), T(_pad(qxs, Q)), T(_pad(qys, Q)),
            T(_pad([m.astype(np.float32) for m in qms], Q)), T(qmask), T(_pad(qis, Q)))


def gaussian_nll(mu, sig, y, m, qmask):
    w = m * qmask.unsqueeze(-1).float()
    nll = 0.5 * ((y - mu) / sig) ** 2 + torch.log(sig) + 0.5 * math.log(2 * math.pi)
    return (nll * w).sum() / w.sum().clamp(min=1)


def laplace_nll(mu, b, y, m, qmask):
    """Laplace likelihood: robust to the heavy tails of MEA responses (scale b)."""
    w = m * qmask.unsqueeze(-1).float()
    nll = (y - mu).abs() / b + torch.log(2 * b)
    return (nll * w).sum() / w.sum().clamp(min=1)


LOSSES = {"gaussian": gaussian_nll, "laplace": laplace_nll}
# central 90 % interval half-width in units of the predicted scale
Z90 = {"gaussian": 1.6449, "laplace": math.log(10.0)}


@dataclass
class TrainConfig:
    steps: int = 4000
    batch: int = 32
    lr: float = 1e-3
    wd: float = 1e-4
    max_ctx_levels: int = 5
    seed: int = 0
    likelihood: str = "gaussian"
    use_interp: bool = False
    n_freq: int = 8
    use_attention: bool = True


def sample_episode(t: ChemTask, rng: np.random.Generator, max_k: int):
    lv = t.levels
    k = int(rng.integers(0, min(max_k, len(lv) - 1) + 1))
    ctx_lv = rng.choice(lv, size=k, replace=False) if k else np.array([], np.float32)
    is_ctx = np.isin(t.logc, ctx_lv)
    return is_ctx, np.ones(len(t.logc), bool)       # score every well (context + held-out)


def train_cnp(train: list[ChemTask], cfg: TrainConfig, device="cuda", log_every=0):
    torch.set_num_threads(int(__import__("os").environ.get("NT_THREADS", "3")))
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = NeuroTrajectoryCNP(use_interp=cfg.use_interp, n_freq=cfg.n_freq, use_attention=cfg.use_attention).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.steps, pct_start=0.1)
    model.train()
    for step in range(cfg.steps):
        idx = rng.integers(0, len(train), cfg.batch)
        bt = [train[i] for i in idx]
        eps = [sample_episode(t, rng, cfg.max_ctx_levels) for t in bt]
        cx, cy, cm, cmask, qx, qy, qm, qmask, qi = make_batch(bt, [e[0] for e in eps], [e[1] for e in eps], device)
        mu, sig = model(cx, cy, cm, cmask, qx, qi)
        loss = LOSSES[cfg.likelihood](mu, sig, qy, qm, qmask)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if log_every and step % log_every == 0:
            print(f"step {step} nll {loss.item():.3f}", flush=True)
    model.eval()
    return model


@torch.no_grad()
def predict(model, task: ChemTask, is_ctx: np.ndarray, q_logc: np.ndarray, device="cuda"):
    """Return (mu, sigma) arrays of shape (Q, ND, NF) for query log10 concentrations."""
    fake = ChemTask(task.chem, task.fold, np.concatenate([task.logc, q_logc]).astype(np.float32),
                    np.concatenate([task.y, np.zeros((len(q_logc), ND, NF), np.float32)]),
                    np.concatenate([task.m, np.zeros((len(q_logc), ND, NF), bool)]),
                    np.concatenate([task.plate, np.array([""] * len(q_logc))]))
    ctx = np.concatenate([is_ctx, np.zeros(len(q_logc), bool)])
    tgt = np.concatenate([np.zeros(len(task.logc), bool), np.ones(len(q_logc), bool)])
    cx, cy, cm, cmask, qx, _, _, _, qi = make_batch([fake], [ctx], [tgt], device)
    mu, sig = model(cx, cy, cm, cmask, qx, qi)
    return (mu[0].cpu().numpy().reshape(-1, ND, NF), sig[0].cpu().numpy().reshape(-1, ND, NF))


def models_dir():
    """Directory of the cross-validated models (override with env NT_MODELS_DIR)."""
    import os
    from pathlib import Path
    return Path(os.environ.get("NT_MODELS_DIR", str(Path(__file__).resolve().parents[4] / "data/processed/models")))


def load_model(path, device="cpu"):
    """Build a NeuroTrajectoryCNP from a checkpoint {'state', 'cfg'} (cfg decides the architecture)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    m = NeuroTrajectoryCNP(use_interp=cfg.get("use_interp", False), n_freq=cfg.get("n_freq", 8),
                           use_attention=cfg.get("use_attention", True)).to(device)
    m.load_state_dict(ck["state"])
    m.eval()
    return m, cfg


def load_fold_models(fold, device="cpu", directory=None):
    from pathlib import Path
    d = Path(directory) if directory else models_dir()
    out = [load_model(p, device) for p in sorted(d.glob(f"cnp_fold{fold}_seed*.pt"))]
    return [m for m, _ in out], (out[0][1] if out else None)
