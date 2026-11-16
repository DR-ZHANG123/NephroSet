"""Index KPMP Visium sections: pair each H&E image with its spot positions and counts.

Each participant folder holds a full-resolution H&E TIFF, a Space Ranger archive
(spatial positions, scale factors, filtered matrix) and/or an expression-matrix archive.
A section enters the study only when image, positions and counts can all be matched and the
spots fall on stained tissue in the image (registration check on the TIFF itself).
"""
from __future__ import annotations

import io
import json
import re
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import load_config, p, stage_dir, write_provenance  # noqa: E402

cfg = load_config()
RAW = p(cfg["paths"]["raw"]) / "kpmp"
OUT = p(cfg["paths"]["interim"]) / "kpmp_sections"
OUT.mkdir(parents=True, exist_ok=True)
S = stage_dir("10_kpmp_index")
TOK = re.compile(r"(V\d{2}[A-Z]\d{2}-\d{3})_(XY\d{2})", re.I)


def read_archive(path: Path) -> dict[str, bytes]:
    want = ("tissue_positions.csv", "tissue_positions_list.csv", "scalefactors_json.json",
            "filtered_feature_bc_matrix.h5", "metrics_summary.csv")
    got = {}
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                b = Path(n).name
                if b in want and "raw_feature" not in n:
                    got[b] = z.read(n)
    else:
        with tarfile.open(path) as t:
            for m in t.getmembers():
                b = Path(m.name).name
                if b in want and "raw_feature" not in m.name:
                    got[b] = t.extractfile(m).read()
    return got


def positions(got: dict[str, bytes]) -> pd.DataFrame:
    if "tissue_positions.csv" in got:
        df = pd.read_csv(io.BytesIO(got["tissue_positions.csv"]))
    else:
        df = pd.read_csv(io.BytesIO(got["tissue_positions_list.csv"]), header=None,
                         names=["barcode", "in_tissue", "array_row", "array_col",
                                "pxl_row_in_fullres", "pxl_col_in_fullres"])
    return df


def tissue_overlap(tif: Path, pos: pd.DataFrame) -> tuple[float, tuple[int, int]]:
    """Fraction of in-tissue spots whose centre lies on stained (non-white) pixels."""
    with tifffile.TiffFile(tif) as t:
        page = t.pages[0]
        shape = page.shape[:2]
        img = page.asarray()
    sub = pos[pos.in_tissue == 1]
    r = np.clip(sub.pxl_row_in_fullres.values.astype(int), 0, shape[0] - 1)
    c = np.clip(sub.pxl_col_in_fullres.values.astype(int), 0, shape[1] - 1)
    rgb = img[r, c, :3].astype(float) if img.ndim == 3 else np.stack([img[r, c]] * 3, 1).astype(float)
    sat = rgb.max(1) - rgb.min(1)
    on = (rgb.mean(1) < 220) | (sat > 20)
    return float(on.mean()), shape


def main() -> None:
    man = pd.read_csv(p("metadata/kpmp_spatial_manifest.tsv"), sep="\t", dtype=str)
    from joblib import Parallel, delayed
    pids = sorted(x.name for x in RAW.iterdir() if x.is_dir() and not x.name.startswith("_"))
    rows = [r for rr in Parallel(n_jobs=16)(delayed(index_participant)(pid) for pid in pids) for r in rr]
    finish(rows, man)


def index_participant(pid: str) -> list[dict]:
    rows = []
    if True:
        files = list((RAW / pid).iterdir())
        tifs = [f for f in files if f.suffix.lower() == ".tif"]
        arch = [f for f in files if re.search(r"(spatial_.*\.(tar\.gz|zip)|space_ranger_outs.*\.zip)$", f.name)]
        for a in arch:
            m = TOK.search(a.name)
            if m:
                cand = [t for t in tifs if m.group(0).lower() in t.name.lower()]
            else:
                cand = tifs if len(tifs) == 1 else []
            rec = dict(participant=pid, archive=a.name, slide=m.group(0) if m else a.stem,
                       tif=cand[0].name if len(cand) == 1 else "", status="")
            if len(cand) != 1:
                rec["status"] = "no_unique_image"
                rows.append(rec)
                continue
            got = read_archive(a)
            if "filtered_feature_bc_matrix.h5" not in got or "scalefactors_json.json" not in got:
                rec["status"] = "missing_matrix_or_scale"
                rows.append(rec)
                continue
            pos = positions(got)
            sf = json.loads(got["scalefactors_json.json"])
            frac_on, shape = tissue_overlap(cand[0], pos)
            sid = f"{pid}__{rec['slide']}"
            d = OUT / sid
            d.mkdir(exist_ok=True)
            (d / "filtered_feature_bc_matrix.h5").write_bytes(got["filtered_feature_bc_matrix.h5"])
            pos.to_csv(d / "tissue_positions.csv", index=False)
            (d / "scalefactors_json.json").write_text(json.dumps(sf))
            (d / "image_path.txt").write_text(str(cand[0].resolve()))
            rec.update(section=sid, n_spots_tissue=int((pos.in_tissue == 1).sum()),
                       spot_diameter_px=sf["spot_diameter_fullres"],
                       um_per_px=55.0 / sf["spot_diameter_fullres"],
                       img_h=shape[0], img_w=shape[1], frac_spots_on_tissue=frac_on,
                       status="ok" if frac_on >= cfg["kpmp_qc"]["min_frac_on_tissue"] else "registration_fail")
            rows.append(rec)
        if not arch:
            rows.append(dict(participant=pid, status="no_spatial_archive", tif=";".join(t.name for t in tifs)))
    return rows


def finish(rows: list[dict], man: pd.DataFrame) -> None:
    idx = pd.DataFrame(rows)
    meta = (man[man.redcap_id != "_clinical"].drop_duplicates("redcap_id")
            .set_index("redcap_id")[["enrollment_category", "primary_adjudicated_category", "sample_type",
                                     "tissue_source", "baseline_egfr", "albuminuria", "proteinuria",
                                     "diabetes_history", "hypertension_history", "age_binned", "sex",
                                     "kdigo_stage", "a1c"]])
    idx = idx.join(meta, on="participant")
    idx.to_csv(S / "sections.tsv", sep="\t", index=False)
    print(idx.status.value_counts())
    ok = idx[idx.status == "ok"]
    print(ok.groupby("primary_adjudicated_category", dropna=False).participant.nunique())
    write_provenance("10_kpmp_index", cfg, [p("metadata/kpmp_spatial_manifest.tsv")], [S / "sections.tsv"],
                     extra={"n_sections_ok": int(len(ok)), "n_participants_ok": int(ok.participant.nunique())})


if __name__ == "__main__":
    main()
