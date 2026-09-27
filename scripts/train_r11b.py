"""R11b: retrain the NeuroTrajectory twin with DIV 9/12-masked context episodes (preregistered:
docs/prereg/r11b_early_exit_retrained.md). The only change from v1 is the masking: with p = 0.5 the context wells lose DIV 9
and DIV 12, and the targets are always full. Everything else matches v1 (per-fold config from the v1
checkpoint, seeds 0/1/2, TrainConfig defaults). Output: data/processed/models_r11b/cnp_fold{f}_seed{s}.pt

Run through the shared GPU queue:
  python D:/PROJECTS/.cognition/gpu_queue/gpuq.py run --name "ai4s:r11b-train" --vram 4 --ram 4 -- python repo/scripts/train_r11b.py
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repo/src"))
from neurotwin.models.cnp import LOSSES, NeuroTrajectoryCNP, TrainConfig, make_batch, sample_episode  # noqa: E402
from neurotwin.models.trajectory import ChemTask  # noqa: E402

PROTOCOL = ROOT / "repo/docs/prereg/r11b_early_exit_retrained.md"
PROTOCOL_SHA = "3ed9b54de673ee23f0a7f5afdc1d88ed87069e76ca052875a634eb48d6eb1a51"
MASK_P = 0.5
LATE = [2, 3]                       # DIV 9, DIV 12
V1_DIR = ROOT / "data/processed/models"
OUT_DIR = ROOT / "data/processed/models_r11b"
FOLDS = [1, 2, 3, 4]
SEEDS = [0, 1, 2]


def composite(t: ChemTask, is_ctx: np.ndarray, hide_late: bool):
    """Task whose first block is the context wells (optionally without DIV 9/12) and whose second
    block is every well in full, so context and targets can differ in visible DIVs."""
    cm = t.m[is_ctx].copy()
    cy = t.y[is_ctx].copy()
    if hide_late:
        cm[:, LATE] = False
        cy[:, LATE] = 0.0
    n = int(is_ctx.sum())
    c = ChemTask(t.chem, t.fold, np.concatenate([t.logc[is_ctx], t.logc]).astype(np.float32),
                 np.concatenate([cy, t.y]).astype(np.float32), np.concatenate([cm, t.m]),
                 np.concatenate([t.plate[is_ctx], t.plate]), t.label)
    ctx = np.zeros(len(c.logc), bool); ctx[:n] = True
    return c, ctx, ~ctx


def train(train_tasks, cfg: TrainConfig, device):
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = NeuroTrajectoryCNP(use_interp=cfg.use_interp, n_freq=cfg.n_freq, use_attention=cfg.use_attention).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.steps, pct_start=0.1)
    model.train()
    for step in range(cfg.steps):
        idx = rng.integers(0, len(train_tasks), cfg.batch)
        comps = []
        for i in idx:
            t = train_tasks[i]
            is_ctx, _ = sample_episode(t, rng, cfg.max_ctx_levels)
            comps.append(composite(t, is_ctx, bool(rng.random() < MASK_P)))
        cx, cy, cm, cmask, qx, qy, qm, qmask, qi = make_batch([c[0] for c in comps], [c[1] for c in comps],
                                                              [c[2] for c in comps], device)
        mu, sig = model(cx, cy, cm, cmask, qx, qi)
        loss = LOSSES[cfg.likelihood](mu, sig, qy, qm, qmask)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0:
            print(f"  step {step} nll {loss.item():.3f}", flush=True)
    model.eval()
    return model


def main():
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == PROTOCOL_SHA, "protocol changed after registration"
    torch.set_num_threads(int(os.environ.get("NT_THREADS", "3")))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks, _ = pickle.load(open(ROOT / "data/processed/epa_nfa/tasks_cache.pkl", "rb"))
    log = {"protocol_sha256": PROTOCOL_SHA, "mask_p": MASK_P, "device": dev, "runs": []}
    t0 = time.time()
    for f in FOLDS:
        v1cfg = torch.load(V1_DIR / f"cnp_fold{f}_seed0.pt", map_location="cpu", weights_only=False)["cfg"]
        tr = [t for t in tasks if t.fold != f]
        for s in SEEDS:
            out = OUT_DIR / f"cnp_fold{f}_seed{s}.pt"
            if out.exists():
                print(f"fold {f} seed {s}: already trained, skipped", flush=True)
                continue
            cfg = TrainConfig(steps=v1cfg["steps"], likelihood=v1cfg["likelihood"], use_interp=v1cfg["use_interp"],
                              n_freq=v1cfg.get("n_freq", 8), seed=s)
            print(f"fold {f} seed {s}: cfg {v1cfg} on {dev} ({len(tr)} training chemicals)", flush=True)
            m = train(tr, cfg, dev)
            torch.save({"state": m.state_dict(), "cfg": {**v1cfg, "r11b_mask_p": MASK_P}}, out)
            log["runs"].append({"fold": f, "seed": s, "elapsed_s": round(time.time() - t0)})
            print(f"fold {f} seed {s}: saved ({time.time() - t0:.0f}s)", flush=True)
    (OUT_DIR / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print("done", flush=True)


if __name__ == "__main__":
    main()
