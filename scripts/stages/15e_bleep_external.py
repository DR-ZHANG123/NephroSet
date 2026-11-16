"""BLEEP baseline: train once on all KPMP participants and predict the external cohort (GSE211785).

Usage: python 15e_bleep_external.py [--seeds10] [--resume] [--max-epochs N]
Same training/validation participants as 15b_external_tokens.py (split_validation with the
master seed); the epoch is chosen on the validation participants, and every external spot is
predicted as the average measured programme scores of its 50 nearest KPMP training spots in the
joint embedding space. Only the external H&E crops are used (no external expression).
Outputs data/cache/external_predictions/bleep[_s10].parquet (pred_*, section, seed).
"""
from __future__ import annotations

import importlib.util
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


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    argv = sys.argv
    sys.argv = [argv[0]]
    spec.loader.exec_module(m)
    sys.argv = argv
    return m


m14e = _load("m14e", "14e_bleep.py")
cfg = load_config()
ARGS = sys.argv[1:]
S10 = "--seeds10" in ARGS
SEEDS = [cfg["seed"] + i for i in range(10)] if S10 else cfg["seeds"]
# --seeds a,b: train only this subset (parallel workers); the merged file is written by whichever
# worker finds all partials present
RUN = [int(x) for x in ARGS[ARGS.index("--seeds") + 1].split(",")] if "--seeds" in ARGS else SEEDS
assert set(RUN) <= set(SEEDS)
RESUME = "--resume" in ARGS
MAX_EP = int(ARGS[ARGS.index("--max-epochs") + 1]) if "--max-epochs" in ARGS else m14e.MAX_EP
T = cfg["train"]
DEV = "cuda:0"
INT = p(cfg["paths"]["interim"])
CACHE = p(cfg["paths"]["cache"])
OUT = CACHE / "external_predictions"
TMP = CACHE / "bleep_tmp" / "external"
XCROPS = CACHE / "bleep_crops" / "gse211785"
for d in (OUT, TMP, XCROPS):
    d.mkdir(parents=True, exist_ok=True)
NAME = "bleep" + ("_s10" if S10 else "")
log = m14e.log


def main() -> None:
    torch.zeros(1, device=DEV)  # initialise CUDA before the large host allocations
    sections = m14e.sections_kpmp()
    crops, expr, y = m14e.prepare(list(sections.section))
    ext = pd.read_csv(p("results/13_gse211785_prep/sections.tsv"), sep="\t").section.tolist()
    todo = [s for s in ext if not (XCROPS / f"{s}.npy").exists()]
    if todo:
        log(f"caching external crops for {len(todo)} sections")
        Parallel(n_jobs=13, max_nbytes=None)(
            delayed(B.crop_section)(INT / "gse211785_spots", INT / "gse211785_sections", XCROPS, s,
                                    cfg["image"]["scale1_um"]) for s in todo)
    xcrops = {s: np.load(XCROPS / f"{s}.npy") for s in ext}
    xindex = {s: pd.read_parquet(INT / "gse211785_spots" / f"{s}.parquet", columns=["section"]).index for s in ext}
    for s in ext:
        assert len(xindex[s]) == len(xcrops[s]), s
    rng = np.random.default_rng(cfg["seed"])
    tr, va = D.split_validation(sections, list(sections.section), T["val_frac"], rng)
    genes = B.select_genes(expr, tr, m14e.PROG_GENES)
    pool = B.Pool(crops, lambda s: expr[s][0][:, genes].toarray(), y, tr, DEV)
    log(f"external: train {len(tr)} sections / {pool.n} spots, val {len(va)}, genes {len(genes)}, seeds {SEEDS}")
    preds, rows = [], []
    for seed in RUN:
        part = TMP / f"{NAME}_seed{seed}.parquet"
        if RESUME and part.exists():
            preds.append(pd.read_parquet(part))
            log(f"seed {seed} loaded from partial")
            continue
        for attempt in range(20):
            try:
                model, ep, val, hist = B.fit(pool, {s: crops[s] for s in va}, {s: y[s] for s in va}, seed, DEV,
                                             MAX_EP, m14e.PATIENCE, log)
                break
            except torch.cuda.OutOfMemoryError:
                log(f"OOM (attempt {attempt}), retrying in 120 s")
                torch.cuda.empty_cache()
                time.sleep(120)
        else:
            raise RuntimeError("repeated OOM")
        dfs = []
        for s in ext:
            df = pd.DataFrame(B.predict(model, pool, xcrops[s], DEV), columns=[f"pred_{c[2:]}" for c in D.PROGS],
                              index=xindex[s])
            df["section"], df["seed"] = s, seed
            dfs.append(df)
        df = pd.concat(dfs)
        df.to_parquet(part)
        preds.append(df)
        rows.append(dict(seed=seed, best_epoch=ep, val_rho_injury=val, n_genes=len(genes)))
        log(f"seed {seed}: best epoch {ep}, val rho {val:.3f}")
        del model
        torch.cuda.empty_cache()
    S = stage_dir(f"15_external/{NAME}")
    if rows:
        f = S / f"training_log{'' if RUN == SEEDS else '_' + '-'.join(map(str, RUN))}.tsv"
        old = pd.read_csv(f, sep="\t") if f.exists() else pd.DataFrame()
        pd.concat([old, pd.DataFrame(rows)]).to_csv(f, sep="\t", index=False)
    parts = [TMP / f"{NAME}_seed{sd}.parquet" for sd in SEEDS]
    if not all(q.exists() for q in parts):
        log(f"seeds {RUN} done; waiting for the other workers before merging")
        return
    pd.concat([pd.read_parquet(q) for q in parts]).to_parquet(OUT / f"{NAME}.parquet")
    log(f"merged {len(parts)} seeds into {NAME}.parquet")
    (S / "done.json").write_text(json.dumps({"done": True, "max_epochs": MAX_EP, "seeds": SEEDS}))
    write_provenance(f"15_external/{NAME}", cfg, [], [S / "done.json"], extra={"model": NAME})


if __name__ == "__main__":
    main()
