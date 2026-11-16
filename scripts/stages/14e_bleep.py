"""BLEEP baseline under the participant-held-out cross-validation on KPMP Visium.

Usage: python 14e_bleep.py [--resume] [--max-epochs N] [--folds 0,1,...]
Same sections, folds, validation participants and seeds as 14_train_cv.py. For each fold and
seed the official BLEEP model (scripts/third_party/bleep; see lib/baseline_bleep.py for the
adaptations) is trained on the fold's training participants, the epoch is chosen on the
held-aside validation participants, and each held-out spot is predicted as the average of the
measured programme scores of its 50 nearest training spots in the joint embedding space.
Outputs data/cache/cv_predictions/bleep_fold{fi}.parquet (pred_*, section, seed, fold).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import baseline_bleep as B  # noqa: E402
from lib import data as D  # noqa: E402
from lib.repro import load_config, p, stage_dir, write_provenance  # noqa: E402

cfg = load_config()
T = cfg["train"]
DEV = "cuda:0"
RESUME = "--resume" in sys.argv
ARGS = sys.argv[1:]
MAX_EP = int(ARGS[ARGS.index("--max-epochs") + 1]) if "--max-epochs" in ARGS else 8
PATIENCE = 2
FOLDS = [int(x) for x in ARGS[ARGS.index("--folds") + 1].split(",")] if "--folds" in ARGS else None
SEEDS = [int(x) for x in ARGS[ARGS.index("--seeds") + 1].split(",")] if "--seeds" in ARGS else cfg["seeds"]
TAG = ARGS[ARGS.index("--tag") + 1] if "--tag" in ARGS else "bleep"
INT = p(cfg["paths"]["interim"])
CACHE = p(cfg["paths"]["cache"])
OUTC = CACHE / "cv_predictions"
TMP = CACHE / "bleep_tmp" / "cv"
CROPS = CACHE / "bleep_crops" / "kpmp"
for d in (OUTC, TMP, CROPS):
    d.mkdir(parents=True, exist_ok=True)
PROG_GENES = sorted({g for v in cfg["programmes"].values() for g in v})


def log(msg: str) -> None:
    print(time.strftime("%m-%d %H:%M:%S"), msg, flush=True)


def sections_kpmp() -> pd.DataFrame:
    idx = pd.read_csv(p("results/10_kpmp_index/sections.tsv"), sep="\t")
    enc = set(pd.read_csv(p("results/12_image_features_kpmp/encoded_sections.tsv"), sep="\t").section)
    sm = pd.read_csv(p("results/11b_kpmp_targets/spot_summary.tsv"), sep="\t")
    keep = set(sm.section[sm.status == "ok"]) & enc
    return idx[idx.section.isin(keep)].reset_index(drop=True)


def prepare(sids: list[str]):
    """Crops (cached on disk), expression and targets for the KPMP sections."""
    todo = [s for s in sids if not (CROPS / f"{s}.npy").exists()]
    if todo:
        log(f"caching crops for {len(todo)} sections")
        Parallel(n_jobs=16, max_nbytes=None)(
            delayed(B.crop_section)(INT / "kpmp_spots", INT / "kpmp_sections", CROPS, s, cfg["image"]["scale1_um"])
            for s in todo)
    crops = B.load_crops(CROPS, sids)
    log("crops loaded")
    log("loading expression")
    res = Parallel(n_jobs=24, max_nbytes=None)(
        delayed(B.section_expression)(INT / "kpmp_sections", INT / "kpmp_spots", s) for s in sids)
    expr = {s: (X, names, hv, disp) for s, X, names, hv, disp in res}
    y = {}
    for s in sids:
        t = pd.read_parquet(INT / "kpmp_spots" / f"{s}.parquet")
        assert len(t) == len(crops[s]) == expr[s][0].shape[0], s
        y[s] = t[D.PROGS].values.astype(np.float32)
    return crops, expr, y


def main() -> None:
    torch.zeros(1, device=DEV)  # initialise CUDA before the large host allocations
    sections = sections_kpmp()
    sids = list(sections.section)
    crops, expr, y = prepare(sids)
    index = {s: pd.read_parquet(INT / "kpmp_spots" / f"{s}.parquet", columns=["section"]).index for s in sids}
    S = stage_dir(f"14_cv/{TAG}")
    hist_all, rows = [], []
    for fi, (tr_all, te) in enumerate(D.participant_folds(sections, T["n_folds"], cfg["seed"])):
        if FOLDS is not None and fi not in FOLDS:
            continue
        out = OUTC / f"{TAG}_fold{fi}.parquet"
        if RESUME and out.exists():
            log(f"fold {fi} done, skipping")
            continue
        rng = np.random.default_rng(cfg["seed"] + fi)
        tr, va = D.split_validation(sections, tr_all, T["val_frac"], rng)
        genes = B.select_genes(expr, tr, PROG_GENES)
        pool = B.Pool(crops, lambda s: expr[s][0][:, genes].toarray(), y, tr, DEV)
        log(f"fold {fi}: train {len(tr)} sections / {pool.n} spots, val {len(va)}, test {len(te)}, genes {len(genes)}")
        preds = []
        for seed in SEEDS:
            part = TMP / f"{TAG}_fold{fi}_seed{seed}.parquet"
            if RESUME and part.exists():
                preds.append(pd.read_parquet(part))
                log(f"fold {fi} seed {seed} loaded from partial")
                continue
            for attempt in range(20):
                try:
                    model, ep, val, hist = B.fit(pool, {s: crops[s] for s in va}, {s: y[s] for s in va}, seed, DEV,
                                                 MAX_EP, PATIENCE, log)
                    break
                except torch.cuda.OutOfMemoryError:
                    log(f"OOM (attempt {attempt}), retrying in 120 s")
                    torch.cuda.empty_cache()
                    time.sleep(120)
            else:
                raise RuntimeError("repeated OOM")
            hist_all += [dict(fold=fi, seed=seed, **h) for h in hist]
            dfs = []
            for s in te:
                yp = B.predict(model, pool, crops[s], DEV)
                df = pd.DataFrame(yp, columns=[f"pred_{c[2:]}" for c in D.PROGS], index=index[s])
                df["section"], df["seed"], df["fold"] = s, seed, fi
                dfs.append(df)
            df = pd.concat(dfs)
            df.to_parquet(part)
            preds.append(df)
            rows.append(dict(fold=fi, seed=seed, best_epoch=ep, val_rho_injury=val, n_genes=len(genes),
                             n_train=len(tr), n_val=len(va), n_test=len(te)))
            log(f"fold {fi} seed {seed}: best epoch {ep}, val rho {val:.3f}")
            del model
            torch.cuda.empty_cache()
        pd.concat(preds).to_parquet(out)
        del pool
        torch.cuda.empty_cache()
    if rows:
        sfx = "" if FOLDS is None else "_fold" + "-".join(map(str, FOLDS))
        f = S / f"training_log{sfx}.tsv"
        old = pd.read_csv(f, sep="\t") if f.exists() else pd.DataFrame()
        pd.concat([old, pd.DataFrame(rows)]).to_csv(f, sep="\t", index=False)
        f2 = S / f"epoch_history{sfx}.tsv"
        old = pd.read_csv(f2, sep="\t") if f2.exists() else pd.DataFrame()
        pd.concat([old, pd.DataFrame(hist_all)]).to_csv(f2, sep="\t", index=False)
    (S / "settings.json").write_text(json.dumps(dict(max_epochs=MAX_EP, patience=PATIENCE, top_k=B.TOP_K,
                                                     batch=B.BATCH, n_hvg=B.N_HVG, seeds=SEEDS), indent=1))
    write_provenance(f"14_cv/{TAG}", cfg, [], [S / "settings.json"], extra={"model": TAG})


if __name__ == "__main__":
    main()
