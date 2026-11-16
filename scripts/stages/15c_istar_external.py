"""iStar baseline trained once on all KPMP participants and applied to GSE211785.

Training/validation split as 15b_external_tokens.py (rng = default_rng(seed), split_validation on
all KPMP sections). Usage: python 15c_istar_external.py [--seeds10] [--phikon]
Features must exist: 14c_istar.py --extract kpmp and --extract gse211785.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import baseline_istar as BI  # noqa: E402
from lib import data as D  # noqa: E402
from lib.repro import load_config, p, stage_dir, write_provenance  # noqa: E402

_spec = importlib.util.spec_from_file_location("m14c", Path(__file__).with_name("14c_istar.py"))
m14c = importlib.util.module_from_spec(_spec)
_argv = sys.argv
sys.argv = [sys.argv[0]] + [a for a in _argv if a == "--phikon"]
_spec.loader.exec_module(m14c)
sys.argv = _argv

cfg = load_config()
S10 = "--seeds10" in sys.argv
if S10:
    cfg["seeds"] = [cfg["seed"] + i for i in range(10)]
T = cfg["train"]
OUT = p(cfg["paths"]["cache"]) / "external_predictions"
NAME = m14c.NAME + ("_s10" if S10 else "")


def main() -> None:
    sections = m14c.sections_kpmp()
    ext = m14c.ext_sections()
    tables = {s: pd.read_parquet(p(cfg["paths"]["interim"]) / "kpmp_spots" / f"{s}.parquet") for s in sections.section}
    tables.update({s: pd.read_parquet(p(cfg["paths"]["interim"]) / "gse211785_spots" / f"{s}.parquet") for s in ext})
    Y = {s: t[D.PROGS].values.astype(np.float32) for s, t in tables.items()}
    X = {**m14c.load_x("kpmp", sections.section, tables), **m14c.load_x("gse211785", ext, tables)}
    rng = np.random.default_rng(cfg["seed"])
    tr, va = D.split_validation(sections, list(sections.section), T["val_frac"], rng)
    S = stage_dir(f"15_external/{NAME}")
    part = OUT / f".{NAME}_partial.parquet"
    preds = [pd.read_parquet(part)] if part.exists() else []
    done = set(preds[0].seed.unique()) if preds else set()
    log = []
    for seed in cfg["seeds"]:
        if seed in done:
            continue
        out, lg = BI.fit_predict(X, Y, tr, va, ext, seed, T, m14c.DEV, m14c.N_STATES, tag="external")
        log += lg
        for s in ext:
            df = pd.DataFrame(out[s], columns=[f"pred_{c[2:]}" for c in D.PROGS], index=tables[s].index)
            df["section"], df["seed"] = s, seed
            preds.append(df)
        pd.concat(preds).to_parquet(part)  # resume point
        pd.DataFrame(log).to_csv(S / "training_log.tsv", sep="\t", index=False, mode="a",
                                 header=not (S / "training_log.tsv").exists())
        log = []
    pd.concat(preds).to_parquet(OUT / f"{NAME}.parquet")
    part.unlink(missing_ok=True)
    (S / "done.json").write_text('{"done": true}')
    write_provenance(f"15_external/{NAME}", cfg, [], [S / "done.json"], extra={"model": m14c.NAME})


if __name__ == "__main__":
    main()
