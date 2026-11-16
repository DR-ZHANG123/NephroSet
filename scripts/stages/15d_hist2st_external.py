"""Hist2ST trained once on all KPMP participants, applied to the external cohort (GSE211785).

Same training/validation split as 15b_external_tokens.py (split_validation with default_rng(seed));
gene panel from the training participants; calibration to the target scale on the validation
participants. --seeds10: seeds 2026-2035 (file hist2st_s10.parquet), otherwise cfg seeds.

Usage: python 15d_hist2st_external.py [--seeds10] [--prep] [--resume] [--worker K --nworkers N]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import baseline_hist2st as B  # noqa: E402
from lib import data as D  # noqa: E402
from lib.repro import load_config, p, stage_dir, write_provenance  # noqa: E402

cfg = load_config()
MODE = "scores"  # direct regression of the programme scores; see lib/baseline_hist2st.py
T = cfg["train"]
OUT = p(cfg["paths"]["cache"]) / "external_predictions"
WORK = B.CACHE / "ext"
S = stage_dir("15d_hist2st_external")
MAX_EPOCHS, VAL_EVERY, PATIENCE, MIN_EPOCHS = 60, 2, 8, 20


def log(msg: str) -> None:
    print(time.strftime("%m-%d %H:%M:%S"), msg, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds10", action="store_true")
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    seeds = [cfg["seed"] + i for i in range(10)] if a.seeds10 else cfg["seeds"]
    name = "hist2st_s10" if a.seeds10 else "hist2st"
    sections = B.sections_kpmp()
    ext = B.sections_ext()
    WORK.mkdir(parents=True, exist_ok=True)
    if a.prep:
        B.prepare("kpmp", list(sections.section))
        B.prepare("gse211785", ext)
        log("prepared")
        return
    rng = np.random.default_rng(cfg["seed"])
    tr, va = D.split_validation(sections, list(sections.section), T["val_frac"], rng)
    mine = [s for s in seeds[a.worker::a.nworkers]
            if not (a.resume and (WORK / f"seed{s}.parquet").exists())]
    if mine:
        data = {s: B.load_section("kpmp", s, True) for s in sections.section}
        data.update({s: B.load_section("gse211785", s, False) for s in ext})
        pf = WORK / "panel.json"
        if MODE != "genes":
            panel = []
        elif pf.exists():
            panel = json.loads(pf.read_text())
        else:
            panel = B.select_panel(data, tr, cfg["seed"])
            pf.with_suffix(".tmp").write_text(json.dumps(panel))
            pf.with_suffix(".tmp").replace(pf)
        log(f"train {len(tr)} val {len(va)} external {len(ext)}; {len(panel)} genes; seeds {mine}")
    for seed in mine:
        wd = WORK / f"seed{seed}"
        if not a.resume and wd.exists():
            shutil.rmtree(wd)
        t0 = time.time()
        tn = B.Trainer(data, tr, va, panel, seed, wd, a.device, MAX_EPOCHS, VAL_EVERY, PATIENCE,
                       log=lambda m: log(f"[s{seed}] {m}"), mode=MODE, min_epochs=MIN_EPOCHS)
        model, best, ep = tn.fit()
        vs = pd.concat([tn.predict_scores(model, s) for s in va])
        cal = B.calibrate(vs, pd.concat([data[s]["table"] for s in va]))
        out = []
        for s in ext:
            df = B.apply_cal(tn.predict_scores(model, s), cal)
            df["section"], df["seed"] = s, seed
            out.append(df)
        pr = pd.concat(out)
        pr.index.name = "barcode"
        pr.to_parquet(WORK / f"seed{seed}.parquet")
        row = dict(model="hist2st", seed=seed, val_rho=best, epochs=ep, minutes=(time.time() - t0) / 60,
                   calibration=json.dumps(cal))
        lf = S / "training_log.tsv"
        pd.DataFrame([row]).to_csv(lf, sep="\t", index=False, mode="a", header=not lf.exists())
        log(f"done seed {seed} val {best:.4f} epochs {ep} ({row['minutes']:.0f} min)")
        (wd / "ckpt.pt").unlink(missing_ok=True)
        del model, tn
        torch.cuda.empty_cache()
    fs = [WORK / f"seed{s}.parquet" for s in seeds]
    if all(f.exists() for f in fs):
        pd.concat([pd.read_parquet(f) for f in fs]).to_parquet(OUT / f"{name}.parquet")
        (S / "done.json").write_text('{"done": true}')
        write_provenance("15d_hist2st_external", cfg, [], [S / "done.json"], extra={"model": name})
        log(f"wrote {OUT / (name + '.parquet')}")


if __name__ == "__main__":
    main()
