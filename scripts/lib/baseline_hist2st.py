"""Hist2ST baseline (Zeng et al., Brief Bioinform 2022) under the project's participant-held-out protocol.

The network is the official implementation (scripts/third_party/hist2st, commit 7480e5d): ConvMixer on the
spot image, Transformer over all spots of a section, GraphSAGE-style GNN over the spot lattice, LSTM
jumping-knowledge, MSE gene head + ZINB head (weight 0.25) + self-distillation over 5 augmented views
(weight 0.5), Adam lr 1e-5 with StepLR(50, 0.9), one whole section per step. Adaptations:
  * input: 112 um spot-centred crop resized to 112 x 112 px (crop() of lib.imagefeat), raw 0-255 values as
    in the original data loader;
  * positions: Visium array_col / array_row embedded with n_pos=128 (the original 64 covers ST arrays only);
  * graph: the six immediate hex-lattice neighbours (the Visium analogue of the original 'Grid' prune,
    which keeps the immediate square-lattice neighbours of ST); spots without neighbours get a self loop
    so that mean aggregation is defined;
  * attention: identical computation through torch scaled_dot_product_attention (memory-efficient kernel)
    so that whole sections fit on one GPU without tiling;
  * target (final, mode="scores"): the three programme scores, standardised on training spots, regressed
    directly with the original MSE + self-distillation losses (ZINB needs counts and is dropped).
    The gene-panel variant (mode="genes": programme genes + training-fold HVGs to 300 genes,
    log1p(counts / library size * 1e4), MSE + ZINB on raw counts, programme scores from predicted
    expression with lib.visium.programme_scores) was run first on all five folds; it collapsed to
    near-constant within-section predictions (validation injury rho 0.00-0.04, predicted-score SD ~1%
    of measured, self-distillation loss ~0, i.e. the image was ignored) at lr 1e-5 and at 1e-4, so the
    direct-score variant (validation rho 0.057 vs 0.019 on fold 0) is used;
  * predicted scores are mapped to the target scale by a monotone linear fit on the validation
    participants (rank-preserving within section);
  * training: max 60 epochs, validation every 2 epochs, early stopping after epoch 20 with patience 8
    validations (the original trains 350 epochs on 31 sections, about 11k steps; 60 epochs here are
    about 8k steps).
"""
from __future__ import annotations

import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import structure as st  # noqa: E402
from lib.imagefeat import crop  # noqa: E402
from lib.repro import load_config, p, set_global_seed  # noqa: E402

H2DIR = ROOT / "scripts" / "third_party" / "hist2st"
PROGS = ["pt_injury", "inflammation", "pt_transport"]
FIG = 112          # Hist2ST fig_size (2 * 224 // 4)
CROP_UM = 112.0    # physical crop size, as scale 1 of the project's image features
N_GENES = 300
N_POS = 128
# official defaults (HIST2ST_train.py: tag 5-7-2-8-4-16-32, lr 1e-5, dropout 0.2, zinb 0.25, bake 5, lamb 0.5)
HP = dict(learning_rate=1e-5, fig_size=FIG, dropout=0.2, n_pos=N_POS, kernel_size=5, patch_size=7,
          n_genes=N_GENES, depth1=2, depth2=8, depth3=4, heads=16, channel=32, zinb=0.25, nb=False,
          bake=5, lamb=0.5, policy="mean")

cfg = load_config()
CACHE = p(cfg["paths"]["cache"]) / "hist2st"


# ----------------------------------------------------------------------------- official network
def official():
    """Import the official Hist2ST module; attention is computed with the fused kernel (same maths)."""
    if "easydl" not in sys.modules:          # imported with * by transformer.py but unused
        sys.modules["easydl"] = types.ModuleType("easydl")
    sys.path.insert(0, str(H2DIR))
    try:
        import transformer as h2t
        import HIST2ST as h2
        import NB_module as h2nb
    finally:
        sys.path.remove(str(H2DIR))
    from einops import rearrange

    def fwd(self, x):
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in self.to_qkv(x).chunk(3, dim=-1))
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.to_out(rearrange(out, "b h n d -> b n (h d)"))

    h2t.Attention.forward = fwd
    return h2, h2nb


# ----------------------------------------------------------------------------- data preparation
def sections_kpmp() -> pd.DataFrame:
    idx = pd.read_csv(p("results/10_kpmp_index/sections.tsv"), sep="\t")
    enc = set(pd.read_csv(p("results/12_image_features_kpmp/encoded_sections.tsv"), sep="\t").section)
    sm = pd.read_csv(p("results/11b_kpmp_targets/spot_summary.tsv"), sep="\t")
    keep = set(sm.section[sm.status == "ok"]) & enc
    return idx[idx.section.isin(keep)].reset_index(drop=True)


def sections_ext() -> list[str]:
    return pd.read_csv(p("results/13_gse211785_prep/sections.tsv"), sep="\t").section.tolist()


def _crops_one(cohort: str, sid: str) -> str:
    import cv2
    import json
    import tifffile
    cv2.setNumThreads(1)
    out = CACHE / f"crops_{cohort}" / f"{sid}.npy"
    if out.exists():
        return sid
    t = pd.read_parquet(p(cfg["paths"]["interim"]) / f"{cohort}_spots" / f"{sid}.parquet")
    sec = p(cfg["paths"]["interim"]) / f"{cohort}_sections" / sid
    if (sec / "um_per_px.txt").exists():
        upp = float((sec / "um_per_px.txt").read_text())
    else:
        upp = 55.0 / json.loads((sec / "scalefactors_json.json").read_text())["spot_diameter_fullres"]
    img = tifffile.imread((sec / "image_path.txt").read_text().strip())
    if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
        img = np.moveaxis(img, 0, -1)
    if img.dtype != np.uint8:
        img = (img / img.max() * 255).astype(np.uint8)
    h = CROP_UM / 2 / upp
    c = np.stack([crop(img, a, b, h, out=FIG) for a, b in zip(t.pxl_row_in_fullres.values,
                                                                 t.pxl_col_in_fullres.values)])
    tmp = out.with_suffix(".tmp.npy")
    np.save(tmp, np.ascontiguousarray(c.transpose(0, 3, 1, 2)))  # N, 3, 112, 112 uint8
    os.replace(tmp, out)
    return sid


def _counts_one(sid: str) -> str:
    import scanpy as sc
    out = CACHE / "counts_kpmp" / f"{sid}.npz"
    if out.exists():
        return sid
    t = pd.read_parquet(p(cfg["paths"]["interim"]) / "kpmp_spots" / f"{sid}.parquet")
    ad = sc.read_10x_h5(p(cfg["paths"]["interim"]) / "kpmp_sections" / sid / "filtered_feature_bc_matrix.h5")
    ad.var_names_make_unique()
    ad = ad[t.index.values]
    X = sp.csr_matrix(ad.X, dtype=np.float32)
    tmp = CACHE / "counts_kpmp" / f"{sid}.tmp.npz"
    sp.save_npz(tmp, X)
    os.replace(tmp, out)
    gf = CACHE / "counts_kpmp" / "genes.txt"
    if not gf.exists():
        gf.write_text("\n".join(ad.var_names))
    else:
        assert gf.read_text().split("\n") == list(ad.var_names), sid
    return sid


def prepare(cohort: str, sections: list[str], n_jobs: int = 24) -> None:
    from joblib import Parallel, delayed
    (CACHE / f"crops_{cohort}").mkdir(parents=True, exist_ok=True)
    Parallel(n_jobs=n_jobs, max_nbytes=None)(delayed(_crops_one)(cohort, s) for s in sections)
    if cohort == "kpmp":
        (CACHE / "counts_kpmp").mkdir(parents=True, exist_ok=True)
        _counts_one(sections[0])  # writes genes.txt first
        Parallel(n_jobs=n_jobs)(delayed(_counts_one)(s) for s in sections[1:])


# ----------------------------------------------------------------------------- per-section inputs
def load_section(cohort: str, sid: str, with_counts: bool) -> dict:
    t = pd.read_parquet(p(cfg["paths"]["interim"]) / f"{cohort}_spots" / f"{sid}.parquet")
    nb = st.hex_neighbours(t.array_row.values, t.array_col.values)
    n = len(t)
    adj = np.zeros((n, n), np.float32)
    for i, j in enumerate(nb):
        adj[i, j] = 1.0
        if len(j) == 0:
            adj[i, i] = 1.0
    d = dict(table=t, crops=torch.from_numpy(np.load(CACHE / f"crops_{cohort}" / f"{sid}.npy")),
             centers=torch.as_tensor(np.stack([t.array_col.values, t.array_row.values], 1), dtype=torch.long),
             adj=torch.from_numpy(adj))
    assert d["centers"].max() < N_POS
    if with_counts:
        d["counts"] = sp.load_npz(CACHE / "counts_kpmp" / f"{sid}.npz").tocsc()
        d["lib"] = np.asarray(d["counts"].sum(1)).ravel()
    return d


def genes_kpmp() -> list[str]:
    return (CACHE / "counts_kpmp" / "genes.txt").read_text().split("\n")


def select_panel(data: dict, train: list[str], seed: int, n_total: int = N_GENES, per_section: int = 300) -> list[str]:
    """Programme genes plus highly variable genes (Seurat dispersion, per section as batch) of the
    training participants, up to n_total genes."""
    import anndata as an
    import scanpy as sc
    genes = genes_kpmp()
    rng = np.random.default_rng(seed)
    Xs, bat = [], []
    for s in train:
        X = data[s]["counts"].tocsr()
        k = rng.choice(X.shape[0], min(per_section, X.shape[0]), replace=False)
        Xs.append(X[k])
        bat += [s] * len(k)
    X = sp.vstack(Xs).tocsr()
    ad = an.AnnData(X)
    ad.var_names = genes
    ad.obs["section"] = bat
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    det = np.asarray((X > 0).mean(0)).ravel()
    ad = ad[:, det >= 0.01].copy()
    hv = sc.pp.highly_variable_genes(ad, flavor="seurat", batch_key="section", n_top_genes=2000, inplace=False)
    hv.index = ad.var_names
    hv = hv.sort_values(["highly_variable_nbatches", "dispersions_norm"], ascending=False)
    prog = [g for k in PROGS for g in cfg["programmes"][k] if g in genes]
    prog = list(dict.fromkeys(prog))
    rest = [g for g in hv.index if g not in prog and not g.startswith("MT-")]
    return prog + rest[:n_total - len(prog)]


def section_targets(d: dict, gidx: np.ndarray) -> dict:
    X = d["counts"][:, gidx].toarray().astype(np.float32)
    lib = d["lib"].astype(np.float32)
    return dict(exp=torch.from_numpy(np.log1p(X / lib[:, None] * 1e4)), oris=torch.from_numpy(X),
                sfs=torch.from_numpy(lib / np.median(lib)))


# ----------------------------------------------------------------------------- scoring
def predicted_scores(pred: np.ndarray, panel: list[str], index) -> pd.DataFrame:
    """Programme scores of predicted expression with the target-defining function."""
    import anndata as an
    from lib.visium import programme_scores
    ad = an.AnnData(pred.astype(np.float32))
    ad.var_names = panel
    ad.obs_names = [str(i) for i in range(len(index))]
    sc_ = programme_scores(ad, {k: cfg["programmes"][k] for k in PROGS}, cfg["visium"]["ctrl_size"], cfg["seed"])
    out = pd.DataFrame({f"s_{k}": sc_[f"y_{k}"].values for k in PROGS}, index=index)
    return out


def within_rho(a, b) -> float:
    return spearmanr(a, b)[0] if np.std(a) > 0 and np.std(b) > 0 else np.nan


def calibrate(scores: pd.DataFrame, tables: pd.DataFrame) -> dict:
    """Linear map from predicted score to the target scale; slope floored at a small positive value so
    the within-section ordering of the predictions is never reversed."""
    cal = {}
    for k in PROGS:
        x, y = scores[f"s_{k}"].values, tables[f"y_{k}"].values
        b = np.cov(x, y)[0, 1] / max(np.var(x, ddof=1), 1e-12)
        b = max(b, 1e-3 * np.std(y) / max(np.std(x), 1e-12))
        cal[k] = (float(y.mean() - b * x.mean()), float(b))
    return cal


def apply_cal(scores: pd.DataFrame, cal: dict) -> pd.DataFrame:
    return pd.DataFrame({f"pred_{k}": cal[k][0] + cal[k][1] * scores[f"s_{k}"].values for k in PROGS},
                        index=scores.index)


# ----------------------------------------------------------------------------- training
class Trainer:
    """Re-implements Hist2ST.training_step (same losses and weights) outside Lightning, with periodic
    validation, early stopping on within-section Spearman of the injury score and checkpoint resume."""

    def __init__(self, data: dict, tr: list[str], va: list[str], panel: list[str], seed: int, workdir: Path,
                 device: str, max_epochs: int, val_every: int, patience: int, log=print,
                 mode: str = "genes", min_epochs: int = 0, lr: float | None = None):
        self.h2, self.h2nb = official()
        self.mode, self.min_epochs, self.lr = mode, min_epochs, lr
        self.data, self.tr, self.va, self.panel, self.seed = data, tr, va, panel, seed
        self.dev, self.max_epochs, self.val_every, self.patience = device, max_epochs, val_every, patience
        self.wd = workdir
        self.wd.mkdir(parents=True, exist_ok=True)
        self.log = log
        gi = {g: i for i, g in enumerate(genes_kpmp())}
        gidx = np.array([gi[g] for g in panel])
        if mode == "genes":
            self.tg = {s: section_targets(data[s], gidx) for s in tr}
        else:  # direct regression of the three programme scores (standardised on training spots)
            yy = np.concatenate([data[s]["table"][[f"y_{k}" for k in PROGS]].values for s in tr])
            self.ymu, self.ysd = yy.mean(0), yy.std(0) + 1e-6
            self.tg = {s: dict(exp=torch.from_numpy(((data[s]["table"][[f"y_{k}" for k in PROGS]].values
                                                      - self.ymu) / self.ysd).astype(np.float32))) for s in tr}

    def _batch(self, s):
        d = self.data[s]
        return (d["crops"].to(self.dev, non_blocking=True).float().unsqueeze(0),
                d["centers"].to(self.dev).unsqueeze(0), d["adj"].to(self.dev))

    def _step(self, model, opt, s):
        patch, center, adj = self._batch(s)
        t = self.tg[s]
        exp = t["exp"].to(self.dev)
        pred, extra, h = model(patch, center, adj)
        mse_loss = F.mse_loss(pred, exp)
        bake_loss = 0
        if model.bake > 0:
            bake_x = model.aug(patch, center, adj)
            new_pred = model.distillation(bake_x)
            bake_loss = F.mse_loss(new_pred, pred)
        zinb_loss = 0.0
        if model.zinb > 0:
            m, dsp, pi = extra
            zinb_loss = self.h2nb.ZINB_loss(t["oris"].to(self.dev), m, dsp, pi, t["sfs"].to(self.dev))
        loss = mse_loss + model.zinb * zinb_loss + model.lamb * bake_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return float(loss), float(mse_loss), float(zinb_loss), float(bake_loss)

    def _retry(self, fn, *a):
        for k in range(60):
            try:
                return fn(*a)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                self.log(f"OOM ({k}); waiting")
                time.sleep(30)
        raise RuntimeError("persistent OOM")

    @torch.no_grad()
    def predict_expr(self, model, s) -> np.ndarray:
        model.eval()
        patch, center, adj = self._batch(s)
        return model(patch, center, adj)[0].float().cpu().numpy()

    def predict_scores(self, model, s) -> pd.DataFrame:
        pr = self._retry(self.predict_expr, model, s)
        if self.mode != "genes":
            return pd.DataFrame({f"s_{k}": pr[:, j] for j, k in enumerate(PROGS)}, index=self.data[s]["table"].index)
        return predicted_scores(pr, self.panel, self.data[s]["table"].index)

    def validate(self, model):
        """Mean within-section Spearman of the injury score; also the mean within-section SD of the
        predicted injury score relative to the measured one (collapse check)."""
        r, sd = [], []
        for s in self.va:
            sc_ = self.predict_scores(model, s)
            y = self.data[s]["table"]["y_pt_injury"].values
            r.append(within_rho(sc_["s_pt_injury"].values, y))
            sd.append(np.std(sc_["s_pt_injury"].values) * (self.ysd[0] if self.mode != "genes" else 1) / np.std(y))
        return float(np.nanmean(r)), float(np.mean(sd))

    def fit(self):
        set_global_seed(self.seed)
        hp = dict(HP)
        if self.mode != "genes":
            hp.update(n_genes=len(PROGS), zinb=0)
        if self.lr is not None:
            hp["learning_rate"] = self.lr
        model = self.h2.Hist2ST(**hp).to(self.dev)
        opt = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.9)
        ck = self.wd / "ckpt.pt"
        state = dict(epoch=0, best=-np.inf, bad=0, hist=[])
        best_state = None
        if ck.exists():
            z = torch.load(ck, map_location=self.dev)
            model.load_state_dict(z["model"]); opt.load_state_dict(z["opt"]); sched.load_state_dict(z["sched"])
            state, best_state = z["state"], z["best_state"]
            torch.set_rng_state(z["rng"]); torch.cuda.set_rng_state(z["cuda_rng"], torch.device(self.dev))
            self.log(f"resumed at epoch {state['epoch']} best {state['best']:.4f}")
        t0 = time.time()
        while state["epoch"] < self.max_epochs and state["bad"] < self.patience:
            ep = state["epoch"]
            model.train()
            ls = []
            for i in np.random.default_rng([self.seed, ep]).permutation(len(self.tr)):
                model.train()
                ls.append(self._retry(self._step, model, opt, self.tr[i]))
            sched.step()
            state["epoch"] = ep + 1
            if state["epoch"] % self.val_every == 0 or state["epoch"] == self.max_epochs:
                v, psd = self.validate(model)
                l = np.mean(ls, 0)
                state["hist"].append(dict(epoch=state["epoch"], val_rho=v, loss=l[0], mse=l[1], zinb=l[2],
                                          bake=l[3], minutes=(time.time() - t0) / 60))
                if v > state["best"] + 1e-4:
                    state["best"], state["bad"] = v, 0
                    best_state = {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}
                elif state["epoch"] > self.min_epochs:
                    state["bad"] += 1
                self.log(f"ep {state['epoch']} loss {l[0]:.4f} mse {l[1]:.4f} zinb {l[2]:.4f} bake {l[3]:.5f} "
                         f"val_rho {v:.4f} pred_sd {psd:.4f} best {state['best']:.4f} ({(time.time() - t0) / 60:.1f} min)")
                tmp = self.wd / "ckpt.tmp"
                torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                                state=state, best_state=best_state, rng=torch.get_rng_state(),
                                cuda_rng=torch.cuda.get_rng_state(torch.device(self.dev))), tmp)
                os.replace(tmp, ck)
        model.load_state_dict(best_state)
        model.eval()
        pd.DataFrame(state["hist"]).to_csv(self.wd / "val_history.tsv", sep="\t", index=False)
        return model, state["best"], state["epoch"]
