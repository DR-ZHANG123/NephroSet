"""Fetch every public raw file used by the study and record its checksum.

Sources: KPMP open Visium (H&E TIFF + Space Ranger outputs + metadata), GEO GSE211785
(Visium counts, spot metadata, H&E TIFF and alignment JSON) and Zenodo 19868428
(CosMx/Xenium atlas). Identical public files already present on this host are hardlinked
instead of re-downloaded; every file is listed in metadata/data_files.tsv with its md5.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import load_config, md5, p, write_provenance  # noqa: E402

cfg = load_config()
RAW = p(cfg["paths"]["raw"])
MIRRORS = [Path(m) for m in cfg.get("mirrors", [])]
_index: dict[str, Path] = {}


def mirror_index() -> dict[str, Path]:
    if not _index:
        for m in MIRRORS:
            if m.exists():
                for f in m.rglob("*"):
                    if f.is_file():
                        _index.setdefault(f.name, f)
    return _index


def fetch(url: str, dest: Path, size: float | None = None) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (size is None or abs(dest.stat().st_size - size) < 1):
        return "present"
    src = mirror_index().get(dest.name)
    if src is not None and (size is None or abs(src.stat().st_size - size) < 1):
        if dest.exists():
            dest.unlink()
        try:
            os.link(src, dest)
        except OSError:
            subprocess.run(["cp", "--reflink=auto", str(src), str(dest)], check=True)
        return "mirror"
    for i in range(4):
        r = subprocess.run(["curl", "-sSfL", "--retry", "3", "-o", str(dest) + ".part", url])
        if r.returncode == 0:
            os.replace(str(dest) + ".part", dest)
            return "downloaded"
        time.sleep(5 * (i + 1))
    raise RuntimeError(f"download failed: {url}")


def kpmp_manifest() -> pd.DataFrame:
    url, key = cfg["kpmp"]["search_url"], cfg["kpmp"]["search_key"]

    def post(body):
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={
            "Authorization": "Bearer " + key, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)

    def flat(v):
        v = v.get("raw") if isinstance(v, dict) else v
        return "|".join(map(str, v)) if isinstance(v, list) else ("" if v is None else str(v))

    rows, page = [], 1
    while True:
        d = post({"query": "", "page": {"size": 1000, "current": page},
                  "filters": {"all": [{"experimental_strategy": cfg["kpmp"]["strategy"]}]},
                  "sort": [{"file_id": "asc"}]})
        rows += [{k: flat(v) for k, v in r.items() if k != "_meta"} for r in d["results"]]
        if page >= d["meta"]["page"]["total_pages"]:
            break
        page += 1
    m = pd.DataFrame(rows)
    m = m[m["access"] == "open"]
    m = m[m.data_format.isin(cfg["kpmp"]["formats"])].copy()
    m["download_url"] = ("https://atlas.kpmp.org/api/v1/file/download/" + m.package_id + "/" + m.file_name)
    # open clinical table (participant-level binned covariates)
    strats = [x["value"] for x in post({"query": "", "page": {"size": 1}, "facets": {
        "experimental_strategy": {"type": "value", "size": 250}}})["facets"]["experimental_strategy"][0]["data"]]
    d = post({"query": "", "page": {"size": 1000}, "filters": {"none": [{"experimental_strategy": strats}]}})
    clin = [{k: flat(v) for k, v in r.items() if k != "_meta"} for r in d["results"]]
    clin = [c for c in clin if c.get("file_name", "").startswith("OpenAccessClinicalData")]
    if clin:
        c = sorted(clin, key=lambda x: x["file_name"])[-1]
        c["download_url"] = f"https://atlas.kpmp.org/api/v1/file/download/{c['package_id']}/{c['file_name']}"
        c["redcap_id"] = "_clinical"
        m = pd.concat([m, pd.DataFrame([c])], ignore_index=True)
    return m


def main() -> None:
    log = []
    man = kpmp_manifest()
    man.to_csv(p("metadata/kpmp_spatial_manifest.tsv"), sep="\t", index=False)
    for r in man.itertuples():
        sub = "kpmp/_clinical" if r.redcap_id == "_clinical" else f"kpmp/{r.redcap_id}"
        dest = RAW / sub / r.file_name
        size = float(r.file_size) if str(r.file_size) not in ("", "nan") else None
        log.append(("KPMP", r.redcap_id, str(dest), fetch(r.download_url, dest, size)))
    g = "GSE211785"
    base = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE211nnn/GSE211785"
    for f in cfg["geo"][g]["series_files"]:
        sub = "matrix" if "series_matrix" in f else "suppl"
        dest = RAW / g / f
        log.append((g, "", str(dest), fetch(f"{base}/{sub}/{f}", dest)))
    lst = subprocess.run(["curl", "-sSfL", f"{base}/suppl/"], capture_output=True, text=True).stdout
    import re
    for f in sorted(set(re.findall(r'(GSE211785_V1[^"<>]+?\.(?:tif|json)\.gz)', lst))):
        dest = RAW / g / f
        log.append((g, "", str(dest), fetch(f"{base}/suppl/{f}", dest)))
    tar = RAW / g / "GSE211785_RAW.tar"
    subprocess.run(["tar", "xf", str(tar), "-C", str(RAW / g), "--skip-old-files"], check=True)
    z = cfg["zenodo"]["record"]
    for f in cfg["zenodo"]["files"]:
        dest = RAW / "zenodo" / f
        log.append(("Zenodo", str(z), str(dest),
                    fetch(f"https://zenodo.org/records/{z}/files/{f}?download=1", dest)))
    ref = cfg["reference"]
    dest = RAW / "reference" / ref["file"]
    log.append(("CELLxGENE", "", str(dest), fetch(ref["url"], dest)))
    df = pd.DataFrame(log, columns=["source", "participant", "path", "status"])
    df["bytes"] = [Path(x).stat().st_size for x in df.path]
    df["md5"] = [md5(x) if b < 3e9 else "" for x, b in zip(df.path, df.bytes)]
    df.to_csv(p("metadata/data_files.tsv"), sep="\t", index=False)
    print(df.groupby(["source", "status"]).size())
    write_provenance("00_download", cfg, [], [p("metadata/data_files.tsv")])


if __name__ == "__main__":
    main()
