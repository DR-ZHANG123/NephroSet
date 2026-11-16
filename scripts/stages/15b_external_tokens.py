"""Train token models once on all KPMP participants and predict the external cohort (GSE211785).

Same training split, roles and seeds as 15_external.py; token roles come from a head trained on
the KPMP training participants (out-of-fold for them) and applied unchanged to the external
sections. Usage: python 15b_external_tokens.py <model> [<model> ...]
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import data as D  # noqa: E402
from lib.models import REGISTRY  # noqa: E402
from lib.repro import load_config, p, stage_dir, write_provenance  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    argv = sys.argv
    sys.argv = [argv[0]]
    spec.loader.exec_module(m)
    sys.argv = argv
    return m


m14b = _load("m14b", "14b_train_tokens.py")
cfg = load_config()
# --seeds10: ten training seeds instead of three (external estimates are seed-sensitive)
S10 = "--seeds10" in sys.argv
if S10:
    sys.argv.remove("--seeds10")
    cfg["seeds"] = [cfg["seed"] + i for i in range(10)]
T = cfg["train"]
DEV = T["device"]
OUT = p(cfg["paths"]["cache"]) / "external_predictions"


def main() -> None:
    names = sys.argv[1:]
    fams = {REGISTRY[n][1].get("fam") for n in names} - {None}
    sections = m14b.sections_kpmp()
    kd = D.load_cohort(p(cfg["paths"]["interim"]) / "kpmp_spots", p(cfg["paths"]["cache"]) / "img_kpmp",
                       sections.section)
    ext = pd.read_csv(p("results/13_gse211785_prep/sections.tsv"), sep="\t").section.tolist()
    xd = D.load_cohort(p(cfg["paths"]["interim"]) / "gse211785_spots", p(cfg["paths"]["cache"]) / "img_gse211785", ext)
    tok = {**m14b.load_tokens(sections.section, "kpmp"), **m14b.load_tokens(ext, "gse211785")}
    for s in ext:
        assert (tok[s][2] == xd[s]["table"].index.values).all(), s
    rng = np.random.default_rng(cfg["seed"])
    tr, va = D.split_validation(sections, list(sections.section), T["val_frac"], rng)
    roles_tr = D.oof_roles(kd, tr, T["role_alpha"], 3, rng)
    rm = D.role_model(kd, tr, T["role_alpha"])
    roles = {**roles_tr, **{s: D.predict_roles(rm, kd[s]) for s in va}, **{s: D.predict_roles(rm, xd[s]) for s in ext}}
    allx = {**kd, **xd}
    mm = np.concatenate([kd[s]["morph"] for s in tr])
    mmu, msd = mm.mean(0), mm.std(0) + 1e-6
    yy = np.concatenate([kd[s]["table"][D.PROGS].values for s in tr])
    ymu, ysd = yy.mean(0), yy.std(0) + 1e-6
    troles = m14b.role_heads(tok, kd, tr, va + ext, cfg["seed"])
    preds = {n: [] for n in names}
    for seed in cfg["seeds"]:
        gr = np.random.default_rng(seed)
        G = {}
        for s in tr + va + ext:
            g = D.build_graph(allx[s], roles[s], mmu, msd, ymu, ysd, T["unit_grid"][0], 1, gr, DEV)
            G[s] = m14b.add_tokens(g, tok[s], troles[s], gr, fams)
        for name in names:
            model, val, ep = m14b.cv14.train_one(name, [G[s] for s in tr], [G[s] for s in va], seed,
                                                 G[tr[0]]["x"].shape[1])
            model.eval()
            with torch.no_grad():
                for s in ext:
                    yp, _ = model(G[s])
                    df = pd.DataFrame(yp.cpu().numpy() * ysd + ymu, columns=[f"pred_{c[2:]}" for c in D.PROGS],
                                      index=xd[s]["table"].index)
                    df["section"], df["seed"] = s, seed
                    preds[name].append(df)
            print(name, seed, round(val, 3), ep, flush=True)
            del model
            torch.cuda.empty_cache()
        del G
    for name in names:
        pd.concat(preds[name]).to_parquet(OUT / f"{name}{'_s10' if S10 else ''}.parquet")
        S = stage_dir(f"15_external/{name}{'_s10' if S10 else ''}")
        (S / "done.json").write_text('{"done": true}')
        write_provenance(f"15_external/{name}", cfg, [], [S / "done.json"], extra={"model": name})


if __name__ == "__main__":
    main()
