"""What NephroSet attends to inside the spot: token attention in held-out participants.

Usage: python 16d_token_attention.py [--resume]   (--resume reuses the per-fold attention cache)

1  Retraining. For each outer fold of 14b_train_tokens.py (same folds, validation participants,
   spot roles, token-role heads and graph construction), tok_abmil_ctx is retrained with the first
   seed only and applied to the held-out participants through lib.attention.AttnTokenNet, which
   records the gated-attention weights over the 196 tokens of every spot without changing the
   computation. Held-out predictions are compared with the cached cross-validated predictions of
   the same seed (sanity check).
2  Deletion. The top 10/20/40% tissue tokens by attention, the bottom 20%, or random tissue tokens
   of the same number (5 draws) are removed from the attention pool, and the within-section
   Spearman correlation between predicted and measured programme scores is recomputed.
3  Analyses (participant = statistical unit; per-spot or per-section quantities are averaged
   within participant first; participant bootstrap CIs and Wilcoxon signed-rank tests):
   a  attention mass by token role (argmax class of the out-of-fold token-role head; tissue tokens)
      relative to the role's token share, per class, per compartment and by position: pale tokens
      (lowest-quartile total H+E optical density of the section's tissue tokens; lumen, space,
      oedematous stroma), lining tokens (non-pale tokens 4-adjacent to a pale token) and dense tokens
      (the rest). Almost all 8 um tokens pass the tissue threshold, so lumina are identified by
      staining density rather than by the tissue mask;
   b  per-spot Spearman correlation between attention and token tissue fraction, haematoxylin and
      eosin optical density (all tissue tokens, and PT tokens only);
   c  dependence on injury in PT-rich spots: within-section Spearman correlation between measured
      PT injury and excess attention on non-PT / interstitial / pale / lining tokens and the
      normalised attention entropy;
   d  deletion test as above.
4  Examples: real 112 um H&E crops with the attention map and token-role map, selected by a fixed
   rule (see EXAMPLE_RULE).
"""
from __future__ import annotations

import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import data as D  # noqa: E402
from lib import tokens as TK  # noqa: E402
from lib.attention import from_trained  # noqa: E402
from lib.evaluate import boot_ci  # noqa: E402
from lib.repro import load_config, p, stage_dir, write_provenance  # noqa: E402
from lib.structure import CLASSES, COMPARTMENTS  # noqa: E402


def _load(name, file, argv):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    mod = importlib.util.module_from_spec(spec)
    old = sys.argv
    sys.argv = argv
    spec.loader.exec_module(mod)
    sys.argv = old
    return mod


m14b = _load("m14b", "14b_train_tokens.py", [sys.argv[0]])
cfg = load_config()
T, E = cfg["train"], cfg["evaluate"]
DEV = T["device"]
MODEL = "tok_abmil_ctx"
SEED = cfg["seeds"][0]
S = stage_dir("16d_token_attention")
CACHE = p(cfg["paths"]["cache"]) / "tok_attention_16d"
CACHE.mkdir(parents=True, exist_ok=True)
SRC = p("figures/source_data")
OUTC = p(cfg["paths"]["cache"]) / "cv_predictions"
PROGS = ["pt_injury", "inflammation", "pt_transport"]
TOP_FRACS = [0.1, 0.2, 0.4]
N_RAND = 5
MIN_TIS = 20            # tissue tokens per spot for per-spot statistics
PT_RICH = E["pt_rich_min"]
COMPS = list(COMPARTMENTS)
EXAMPLE_RULE = ("Participants: the three held-out participants with the most candidate spots, at most one per "
                "outer fold. Candidate spots (representative tubular tissue): measured PT fraction >= 0.7, "
                ">= 95% tissue tokens, >= 50% of tissue tokens assigned to PT and <= 15% pale tokens. High injury: top decile of measured PT injury among the participant's "
                "candidate spots, choosing the spot with the highest predicted injury; low injury: bottom "
                "decile, choosing the spot with the lowest predicted injury.")


# ------------------------------------------------------------------------------------------------
# 1-2  retraining, attention and deletion predictions
# ------------------------------------------------------------------------------------------------
def keep_mask(att, tis, frac, how, rng):
    """(n,196) 0/1 mask removing `frac` of each spot's tissue tokens by rule `how`."""
    n = att.shape[0]
    keep = np.ones_like(tis, np.float32)
    k = np.floor(frac * tis.sum(1)).astype(int)
    if how == "random":
        key = rng.random(att.shape)
    elif how == "top":
        key = -att
    else:  # bottom
        key = att.copy()
    key = np.where(tis, key, np.inf)
    order = np.argsort(key, 1, kind="stable")
    rank = np.empty_like(order)
    rank[np.arange(n)[:, None], order] = np.arange(att.shape[1])[None]
    keep[rank < k[:, None]] = 0.0
    return keep


@torch.no_grad()
def predict(model, g, ysd, ymu, keep=None):
    model.keep = None if keep is None else torch.as_tensor(keep.ravel(), device=DEV)
    yp, _ = model(g)
    model.keep = None
    return (yp.cpu().numpy() * ysd + ymu).astype(np.float32), model.last_att.cpu().numpy()


def run_fold(fi, tr_all, te, sections, data, tok):
    rng = np.random.default_rng(cfg["seed"] + fi)
    tr, va = D.split_validation(sections, tr_all, T["val_frac"], rng)
    roles_tr = D.oof_roles(data, tr, T["role_alpha"], 3, rng)
    rm = D.role_model(data, tr, T["role_alpha"])
    roles = {**roles_tr, **{s: D.predict_roles(rm, data[s]) for s in va + te}}
    mm = np.concatenate([data[s]["morph"] for s in tr])
    mmu, msd = mm.mean(0), mm.std(0) + 1e-6
    yy = np.concatenate([data[s]["table"][D.PROGS].values for s in tr])
    ymu, ysd = yy.mean(0), yy.std(0) + 1e-6
    troles = m14b.role_heads(tok, data, tr, va + te, cfg["seed"] + fi)
    gr = np.random.default_rng(SEED)
    G = {}
    for s in tr + va + te:
        g = D.build_graph(data[s], roles[s], mmu, msd, ymu, ysd, T["unit_grid"][0], 1, gr, DEV)
        G[s] = m14b.add_tokens(g, tok[s], troles[s], gr, set())
    d_in = G[tr[0]]["x"].shape[1]
    for attempt in range(20):
        try:
            model, val, ep = m14b.cv14.train_one(MODEL, [G[s] for s in tr], [G[s] for s in va], SEED, d_in)
            break
        except torch.cuda.OutOfMemoryError:
            print("OOM, retrying in 120 s", flush=True)
            torch.cuda.empty_cache()
            time.sleep(120)
    else:
        raise RuntimeError("repeated OOM")
    model.eval()
    am = from_trained(model, MODEL, d_in, T["hidden"], T["dropout"])
    drng = np.random.default_rng(cfg["seed"] + 100 + fi)
    out = {}
    for s in te:
        with torch.no_grad():
            y0 = (model(G[s])[0].cpu().numpy() * ysd + ymu).astype(np.float32)
        y1, att = predict(am, G[s], ysd, ymu)
        tis = tok[s][1][..., 0].float().numpy() >= TK.TISSUE_MIN
        rec = dict(barcode=tok[s][2], att=att.astype(np.float32), cls=troles[s].argmax(-1).astype(np.int8),
                   pred_orig=y0, pred=y1, max_abs_diff=float(np.abs(y0 - y1).max()))
        for f in TOP_FRACS:
            rec[f"pred_top{int(f * 100)}"] = predict(am, G[s], ysd, ymu, keep_mask(att, tis, f, "top", drng))[0]
            rec[f"pred_rand{int(f * 100)}"] = np.stack(
                [predict(am, G[s], ysd, ymu, keep_mask(att, tis, f, "random", drng))[0] for _ in range(N_RAND)])
        rec["pred_bot20"] = predict(am, G[s], ysd, ymu, keep_mask(att, tis, 0.2, "bottom", drng))[0]
        out[s] = rec
    info = dict(fold=fi, n_train=len(tr), n_val=len(va), n_test=len(te), val_rho=val, epochs=ep)
    del G, model, am
    torch.cuda.empty_cache()
    return out, info


# ------------------------------------------------------------------------------------------------
# 3  analyses
# ------------------------------------------------------------------------------------------------
def lining_tokens(pale, tis):
    """Non-pale tissue tokens 4-adjacent to a pale token (lining of a lumen or space)."""
    m = pale.reshape(-1, TK.G, TK.G)
    nb = np.zeros_like(m)
    nb[:, 1:, :] |= m[:, :-1, :]
    nb[:, :-1, :] |= m[:, 1:, :]
    nb[:, :, 1:] |= m[:, :, :-1]
    nb[:, :, :-1] |= m[:, :, 1:]
    return (nb.reshape(tis.shape) & tis & ~pale)


def row_spearman(a, b, mask, min_n):
    out = np.full(a.shape[0], np.nan)
    for i in np.flatnonzero(mask.sum(1) >= min_n):
        x, y = a[i, mask[i]], b[i, mask[i]]
        if np.ptp(x) > 0 and np.ptp(y) > 0:
            rx, ry = rankdata(x), rankdata(y)
            out[i] = np.corrcoef(rx, ry)[0, 1]
    return out


def summarise(v: pd.Series, rng, null=0.0) -> dict:
    v = v.dropna()
    lo, hi = boot_ci(v.values, E["n_boot"], rng)
    try:
        pw = wilcoxon(v.values - null).pvalue if len(v) > 5 else np.nan
    except ValueError:
        pw = np.nan
    return dict(mean=float(v.mean()), lo=lo, hi=hi, p_wilcoxon=float(pw), n_participants=int(len(v)))


def spot_frame(s, rec, tok, part, fold):
    """Per-spot attention summaries joined with measured targets and composition."""
    t = pd.read_parquet(p(cfg["paths"]["interim"]) / "kpmp_spots" / f"{s}.parquet")
    assert (t.index.values == rec["barcode"]).all()
    col = tok[s][1].float().numpy()
    tis = col[..., 0] >= TK.TISSUE_MIN
    att = rec["att"] * tis
    att = att / np.maximum(att.sum(1, keepdims=True), 1e-12)
    cls = np.where(tis, rec["cls"], -1)
    comp = np.where(tis, TK.COMP_OF_CLASS[np.clip(rec["cls"], 0, None)], -1)
    od = col[..., 1] + col[..., 2]
    pale = tis & (od < np.quantile(od[tis], 0.25))
    lin = lining_tokens(pale, tis)
    dense = tis & ~pale & ~lin
    ntis = tis.sum(1)
    df = pd.DataFrame(index=t.index)
    df["section"], df["participant"], df["fold"], df["n_tissue"] = s, part, fold, ntis
    for k in PROGS:
        df[f"y_{k}"] = t[f"y_{k}"].values
    df["comp_PT"] = t["comp_PT"].values
    df["comp_inter"] = t[["comp_Endothelial", "comp_Fibroblast", "comp_Mural", "comp_Immune"]].sum(1).values
    for j, c in enumerate(CLASSES):
        df[f"mass_{c}"] = (att * (cls == j)).sum(1)
        df[f"share_{c}"] = (cls == j).sum(1) / np.maximum(ntis, 1)
    for j, c in enumerate(COMPS):
        df[f"mass_{c}"] = (att * (comp == j)).sum(1)
        df[f"share_{c}"] = (comp == j).sum(1) / np.maximum(ntis, 1)
    for nm, m in [("pale", pale), ("lining", lin), ("dense", dense), ("PT_pale", pale & (cls == 0)),
                  ("PT_lining", lin & (cls == 0)), ("PT_dense", dense & (cls == 0))]:
        df[f"mass_{nm}"] = (att * m).sum(1)
        df[f"share_{nm}"] = m.sum(1) / np.maximum(ntis, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -(np.where(att > 0, att * np.log(att), 0)).sum(1)
        df["entropy_norm"] = H / np.log(np.maximum(ntis, 2))
    df["att_max"] = att.max(1)
    # attention vs token morphology
    feats = {"tissue_fraction": col[..., 0], "haematoxylin_od": col[..., 1], "eosin_od": col[..., 2],
             "pale": pale.astype(float), "lining": lin.astype(float)}
    for nm, f in feats.items():
        df[f"rho_{nm}"] = row_spearman(rec["att"], f, tis, MIN_TIS)
        df[f"rhoPT_{nm}"] = row_spearman(rec["att"], f, tis & (cls == 0), 10)
    for k, j in zip(PROGS, range(3)):
        df[f"pred_{k}"] = rec["pred"][:, j]
    return df


def analyse(cache, tok, sections, ex_rng):
    part_of = dict(zip(sections.section, sections.participant))
    spots = pd.concat([spot_frame(s, rec, tok, part_of[s], fi) for fi, c in cache.items() for s, rec in c.items()])
    ok = spots[spots.n_tissue >= MIN_TIS]
    rng = np.random.default_rng(cfg["seed"])
    # a  enrichment ----------------------------------------------------------------------------
    enr = []
    subsets = {"all": ok, "pt_rich": ok[ok.comp_PT >= PT_RICH],
               "mixed": ok[(ok.comp_PT >= 0.2) & (ok.comp_inter >= 0.2)]}
    groups = [("class", c) for c in CLASSES] + [("compartment", c) for c in COMPS] + \
             [("position", c) for c in ["pale", "lining", "dense", "PT_pale", "PT_lining", "PT_dense"]]
    for sub, q in subsets.items():
        P = q.groupby("participant").sum(numeric_only=True)
        n_sp = q.groupby("participant").size()
        for kind, c in groups:
            share = P[f"share_{c}"] / n_sp
            mass = P[f"mass_{c}"] / n_sp
            valid = (share >= 0.01) & (n_sp >= 20)
            lr = np.log2(mass[valid] / share[valid])
            r = summarise(lr, rng)
            enr.append(dict(subset=sub, kind=kind, group=c, token_share=float(share[valid].mean()),
                            attention_mass=float(mass[valid].mean()), log2_ratio=r["mean"], lo=r["lo"], hi=r["hi"],
                            ratio=2 ** r["mean"], ratio_lo=2 ** r["lo"], ratio_hi=2 ** r["hi"],
                            p_wilcoxon=r["p_wilcoxon"], n_participants=r["n_participants"],
                            frac_participants_enriched=float((lr > 0).mean())))
    enr = pd.DataFrame(enr)
    # b  morphology -------------------------------------------------------------------------------
    mor = []
    for sub, q in [("all", ok), ("pt_rich", ok[ok.comp_PT >= PT_RICH])]:
        for scope, pre in [("tissue_tokens", "rho_"), ("PT_tokens", "rhoPT_")]:
            for f in ["tissue_fraction", "haematoxylin_od", "eosin_od", "pale", "lining"]:
                v = q.groupby("participant")[f"{pre}{f}"].mean()
                r = summarise(v, rng)
                mor.append(dict(subset=sub, tokens=scope, feature=f, mean_rho=r["mean"], lo=r["lo"], hi=r["hi"],
                                p_wilcoxon=r["p_wilcoxon"], n_participants=r["n_participants"],
                                n_spots=int(q[f"{pre}{f}"].notna().sum())))
    mor = pd.DataFrame(mor)
    # c  injury dependence in PT-rich spots -------------------------------------------------------
    q = ok[ok.comp_PT >= PT_RICH].copy()
    q["excess_nonPT"] = (1 - q.mass_PT) - (1 - q.share_PT)
    q["excess_interstitial"] = q.mass_interstitial - q.share_interstitial
    q["excess_pale"] = q.mass_pale - q.share_pale
    q["excess_lining"] = q.mass_lining - q.share_lining
    q["share_nonPT"] = 1 - q.share_PT
    q["mass_nonPT"] = 1 - q.mass_PT
    mets = ["excess_nonPT", "excess_interstitial", "excess_pale", "excess_lining", "entropy_norm", "att_max", "mass_nonPT",
            "share_nonPT", "share_interstitial"]
    secr = []
    for s, g in q.groupby("section"):
        if len(g) < 30:
            continue
        for m in mets:
            for tgt in ["y_pt_injury", "pred_pt_injury"]:
                a, b = g[tgt].values, g[m].values
                rr = spearmanr(a, b)[0] if np.ptp(a) > 0 and np.ptp(b) > 0 else np.nan
                secr.append(dict(section=s, participant=g.participant.iloc[0], metric=m, target=tgt, rho=rr,
                                 n_spots=len(g)))
    secr = pd.DataFrame(secr)
    inj = []
    for (m, tgt), g in secr.groupby(["metric", "target"]):
        r = summarise(g.groupby("participant").rho.mean(), rng)
        inj.append(dict(metric=m, target=tgt, mean_rho=r["mean"], lo=r["lo"], hi=r["hi"],
                        p_wilcoxon=r["p_wilcoxon"], n_participants=r["n_participants"]))
    inj = pd.DataFrame(inj)
    # also between participants: mean measured injury vs mean excess non-PT attention
    pm = q.groupby("participant")[["y_pt_injury", "excess_nonPT", "excess_interstitial", "entropy_norm"]].mean()
    for m in ["excess_nonPT", "excess_interstitial", "entropy_norm"]:
        inj.loc[len(inj)] = dict(metric=m, target="between_participant_y_pt_injury",
                                 mean_rho=spearmanr(pm.y_pt_injury, pm[m])[0], lo=np.nan, hi=np.nan,
                                 p_wilcoxon=spearmanr(pm.y_pt_injury, pm[m])[1], n_participants=len(pm))
    # d  deletion -----------------------------------------------------------------------------------
    dels = []
    conds = ["pred"] + [f"pred_top{int(f * 100)}" for f in TOP_FRACS] + \
            [f"pred_rand{int(f * 100)}" for f in TOP_FRACS] + ["pred_bot20"]
    for fi, c in cache.items():
        for s, rec in c.items():
            t = pd.read_parquet(p(cfg["paths"]["interim"]) / "kpmp_spots" / f"{s}.parquet")
            ptr = t.comp_PT.values >= PT_RICH
            for j, k in enumerate(PROGS):
                y = t[f"y_{k}"].values
                for cond in conds:
                    P_ = rec[cond]
                    P_ = P_[None] if P_.ndim == 2 else P_
                    rho = np.mean([spearmanr(y, x[:, j])[0] for x in P_])
                    rho_pt = np.mean([spearmanr(y[ptr], x[ptr, j])[0] for x in P_]) if ptr.sum() >= 30 else np.nan
                    dels.append(dict(section=s, participant=part_of[s], fold=fi, programme=k, condition=cond,
                                     rho=rho, rho_pt_rich=rho_pt))
    dels = pd.DataFrame(dels)
    dpart = dels.groupby(["participant", "programme", "condition"])[["rho", "rho_pt_rich"]].mean().reset_index()
    dsum = []
    for k in PROGS:
        for metric in ["rho", "rho_pt_rich"]:
            W = dpart[dpart.programme == k].pivot(index="participant", columns="condition", values=metric)
            for cond in conds:
                r = summarise(W[cond], rng)
                dsum.append(dict(programme=k, metric=metric, contrast=cond, kind="level", **r))
            for f in TOP_FRACS:
                tag = int(f * 100)
                for a, b in [(f"pred_top{tag}", "pred"), (f"pred_rand{tag}", "pred"),
                             (f"pred_top{tag}", f"pred_rand{tag}")]:
                    r = summarise(W[a] - W[b], rng)
                    dsum.append(dict(programme=k, metric=metric, contrast=f"{a} - {b}", kind="difference", **r))
            r = summarise(W["pred_bot20"] - W["pred_rand20"], rng)
            dsum.append(dict(programme=k, metric=metric, contrast="pred_bot20 - pred_rand20", kind="difference", **r))
    dsum = pd.DataFrame(dsum)
    return spots, enr, mor, secr, inj, dpart, dsum


# ------------------------------------------------------------------------------------------------
# 4  examples
# ------------------------------------------------------------------------------------------------
def choose_examples(spots):
    q = spots[(spots.comp_PT >= 0.7) & (spots.n_tissue >= 0.95 * TK.NT) & (spots.share_PT >= 0.5)
              & (spots.share_pale <= 0.15)]
    cnt = q.groupby(["participant", "fold"]).size().reset_index(name="n").sort_values("n", ascending=False)
    chosen, folds = [], set()
    for _, r in cnt.iterrows():
        if r.fold not in folds:
            chosen.append(r.participant)
            folds.add(r.fold)
        if len(chosen) == 3:
            break
    rows = []
    for part in chosen:
        g = q[q.participant == part]
        hi_t, lo_t = g.y_pt_injury.quantile(0.9), g.y_pt_injury.quantile(0.1)
        hi = g[g.y_pt_injury >= hi_t].sort_values("pred_pt_injury", ascending=False).iloc[0]
        lo = g[g.y_pt_injury <= lo_t].sort_values("pred_pt_injury").iloc[0]
        for lab, r in [("low", lo), ("high", hi)]:
            rows.append(dict(participant=part, fold=int(r.fold), section=r.section, barcode=r.name, injury_group=lab,
                             y_pt_injury=r.y_pt_injury, pred_pt_injury=r.pred_pt_injury,
                             y_pt_injury_pct=float((g.y_pt_injury <= r.y_pt_injury).mean() * 100),
                             pred_pt_injury_pct=float((g.pred_pt_injury <= r.pred_pt_injury).mean() * 100),
                             comp_PT=r.comp_PT, n_candidates=len(g), n_tissue=int(r.n_tissue),
                             mass_PT=r.mass_PT, share_PT=r.share_PT, entropy_norm=r.entropy_norm))
    return pd.DataFrame(rows)


def example_arrays(ex, cache):
    m12b = _load("m12b", "12b_token_features.py", [sys.argv[0], "kpmp"])
    imgs, atts, clss = [], [], []
    for s, g in ex.groupby("section", sort=False):
        sid, c1, bc = m12b.crops(s)
        pos = {b: i for i, b in enumerate(bc)}
        rec = next(c[s] for c in cache.values() if s in c)
        for _, r in g.iterrows():
            i = pos[r.barcode]
            imgs.append((r.name, c1[i]))
            atts.append((r.name, rec["att"][i].reshape(TK.G, TK.G)))
            clss.append((r.name, rec["cls"][i].reshape(TK.G, TK.G)))
    order = lambda L: np.stack([x for _, x in sorted(L, key=lambda z: z[0])])  # noqa: E731
    return order(imgs), order(atts), order(clss)


def main() -> None:
    sections = m14b.sections_kpmp()
    folds = D.participant_folds(sections, T["n_folds"], cfg["seed"])
    cache, infos = {}, []
    need = [fi for fi in range(len(folds)) if not ("--resume" in sys.argv and (CACHE / f"fold{fi}.pkl").exists())]
    if need:
        data = D.load_cohort(p(cfg["paths"]["interim"]) / "kpmp_spots",
                             p(cfg["paths"]["cache"]) / "img_kpmp", sections.section)
    for fi, (tr_all, te) in enumerate(folds):
        f = CACHE / f"fold{fi}.pkl"
        if fi in need:
            tok = m14b.load_tokens(sections.section, fold=fi)  # projection fitted on this fold's training data
            for s in sections.section:
                assert (tok[s][2] == data[s]["table"].index.values).all(), s
            out, info = run_fold(fi, tr_all, te, sections, data, tok)
            pickle.dump((out, info), open(f, "wb"))
            print("fold", fi, info, time.strftime("%H:%M"), flush=True)
        out, info = pickle.load(open(f, "rb"))
        cache[fi] = out
        infos.append(info)
    # sanity check against the cached cross-validated predictions (same seed)
    san = []
    for fi, c in cache.items():
        ref = pd.read_parquet(OUTC / f"{MODEL}_fold{fi}.parquet")
        ref = ref[ref.seed == SEED]
        for s, rec in c.items():
            r = ref[ref.section == s].loc[rec["barcode"]]
            for j, k in enumerate(PROGS):
                a, b = rec["pred"][:, j], r[f"pred_{k}"].values
                san.append(dict(fold=fi, section=s, programme=k, pearson=np.corrcoef(a, b)[0, 1],
                                spearman=spearmanr(a, b)[0], max_abs_diff_attn_vs_tokennet=rec["max_abs_diff"]))
    san = pd.DataFrame(san)
    san.to_csv(S / "sanity_vs_cv_predictions.tsv", sep="\t", index=False)
    ss = san.groupby(["fold", "programme"])[["pearson", "spearman", "max_abs_diff_attn_vs_tokennet"]].agg(
        ["median", "min"]).reset_index()
    ss.columns = ["_".join(c).strip("_") for c in ss.columns]
    ss.to_csv(S / "sanity_summary.tsv", sep="\t", index=False)
    pd.DataFrame(infos).to_csv(S / "folds.tsv", sep="\t", index=False)
    print(ss.to_string(), flush=True)

    if not need:  # resumed from the cache: token colour features do not depend on the fold projection
        tok = m14b.load_tokens(sections.section, fold=0)
    spots, enr, mor, secr, inj, dpart, dsum = analyse(cache, tok, sections, None)
    enr.to_csv(S / "enrichment.tsv", sep="\t", index=False)
    mor.to_csv(S / "morphology_correlation.tsv", sep="\t", index=False)
    secr.to_csv(S / "injury_dependence_sections.tsv", sep="\t", index=False)
    inj.to_csv(S / "injury_dependence.tsv", sep="\t", index=False)
    dpart.to_csv(S / "deletion_participants.tsv", sep="\t", index=False)
    dsum.to_csv(S / "deletion_test.tsv", sep="\t", index=False)
    ex = choose_examples(spots).reset_index(drop=True)
    ex.to_csv(S / "examples.tsv", sep="\t", index=False)
    (S / "example_rule.txt").write_text(EXAMPLE_RULE + "\n")
    img, att, cls = example_arrays(ex, cache)
    # figure source data
    enr.to_csv(SRC / "Fig5_attention_a_enrichment.tsv", sep="\t", index=False)
    mor.to_csv(SRC / "Fig5_attention_b_morphology.tsv", sep="\t", index=False)
    inj.to_csv(SRC / "Fig5_attention_c_injury.tsv", sep="\t", index=False)
    dpart.to_csv(SRC / "Fig5_attention_d_deletion_participants.tsv", sep="\t", index=False)
    dsum.to_csv(SRC / "Fig5_attention_d_deletion.tsv", sep="\t", index=False)
    ex.to_csv(SRC / "Fig5_attention_e_examples.tsv", sep="\t", index=False)
    np.savez_compressed(SRC / "Fig5_attention_e_examples.npz", img=img, att=att, cls=cls,
                        classes=np.array(CLASSES), tissue_min=TK.TISSUE_MIN,
                        tissue=np.stack([tok[s][1][list(tok[s][2]).index(b), :, 0].float().numpy().reshape(TK.G, TK.G)
                                         for s, b in zip(ex.section, ex.barcode)]))
    outs = sorted(S.glob("*.tsv")) + [S / "example_rule.txt"]
    write_provenance("16d_token_attention", cfg, [OUTC / f"{MODEL}_fold{fi}.parquet" for fi in cache],
                     outs, extra={"model": MODEL, "seed_used": SEED, "top_fracs": TOP_FRACS, "n_random": N_RAND,
                                  "min_tissue_tokens": MIN_TIS, "example_rule": EXAMPLE_RULE,
                                  "n_spots_analysed": int(len(spots))})
    print(json.dumps({"done": True, "n_spots": int(len(spots))}), flush=True)


if __name__ == "__main__":
    main()
