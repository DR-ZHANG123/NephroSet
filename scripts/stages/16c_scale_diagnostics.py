"""Scale of tissue structure relative to Visium spots, and quality of token roles.

1  Mixing within spots (measured composition): effective number of classes per spot
   (exp of entropy) and the share of spots with both PT and interstitial fractions >= 0.2;
   size of spot-level anatomical units and their overlap with 2-ring neighbourhoods.
2  Token roles: in each cross-validation fold, the token-role head trained on the training
   participants is applied to held-out participants, and the spot mean of token roles is
   correlated with measured composition within each section.
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
from lib import structure as st  # noqa: E402
from lib import tokens as TK  # noqa: E402
from lib.repro import load_config, p, stage_dir, write_provenance  # noqa: E402

spec = importlib.util.spec_from_file_location("m14b", Path(__file__).with_name("14b_train_tokens.py"))
m14b = importlib.util.module_from_spec(spec)
_argv = sys.argv
sys.argv = [sys.argv[0]]
spec.loader.exec_module(m14b)
sys.argv = _argv
cfg = load_config()
T = cfg["train"]
S = stage_dir("16c_scale_diagnostics")


def main() -> None:
    sections = m14b.sections_kpmp()
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for s in sections.section:
        t = pd.read_parquet(p(cfg["paths"]["interim"]) / "kpmp_spots" / f"{s}.parquet")
        c = t[D.COMP].values
        pr = np.clip(c, 1e-9, 1)
        H = -(pr * np.log(pr)).sum(1)
        nb = st.hex_neighbours(t.array_row.values, t.array_col.values)
        ring2 = st.ring_sets(nb, 2)
        edges, et = st.anatomical_hyperedges(nb, st.compartment(c), T["unit_grid"][0], rng)
        units = [e for e, k in zip(edges, et) if k == 0]
        jac = [len(set(ring2[e[len(e) // 2]]) & set(e)) / len(set(ring2[e[len(e) // 2]]) | set(e))
               for e in units if len(e) >= 3]
        rows.append(dict(section=s, n_spots=len(t), eff_classes=np.exp(H).mean(),
                         frac_pt_and_interstitium=((c[:, 0] >= 0.2) & (c[:, 5:].sum(1) >= 0.2)).mean(),
                         unit_size=np.mean([len(e) for e in units]), interface_per_spot=(et == 1).sum() / len(t),
                         jaccard_unit_ring2=np.mean(jac)))
    mix = pd.DataFrame(rows)
    mix.to_csv(S / "spot_mixing.tsv", sep="\t", index=False)
    # token roles, held-out participants
    data = D.load_cohort(p(cfg["paths"]["interim"]) / "kpmp_spots", p(cfg["paths"]["cache"]) / "img_kpmp",
                         sections.section)
    dev = T["device"]
    rr = []
    for fi, (tr_all, te) in enumerate(D.participant_folds(sections, T["n_folds"], cfg["seed"])):
        tok = m14b.load_tokens(sections.section, fold=fi)
        comp = {s: torch.as_tensor(data[s]["table"][D.COMP].values, dtype=torch.float32, device=dev) for s in tr_all}
        head = TK.train_role_head([tok[s][0] for s in tr_all], [tok[s][1] for s in tr_all],
                                  [comp[s] for s in tr_all], dev, seed=cfg["seed"] + fi)
        for s in te:
            r = TK.token_roles(head, tok[s][0], tok[s][1])
            w = tok[s][1][..., 0].float().numpy() >= TK.TISSUE_MIN
            est = (r * w[..., None]).sum(1) / np.maximum(w.sum(1, keepdims=True), 1)
            meas = data[s]["table"][D.COMP].values
            for j, c in enumerate(st.CLASSES):
                if meas[:, j].std() > 0 and est[:, j].std() > 0:
                    rr.append(dict(fold=fi, section=s, cls=c, r=np.corrcoef(est[:, j], meas[:, j])[0, 1]))
    rr = pd.DataFrame(rr)
    rr.to_csv(S / "token_roles_heldout.tsv", sep="\t", index=False)
    summ = pd.concat([mix.drop(columns="section").mean().rename("mean").to_frame().T.assign(what="spot_mixing"),
                      rr.groupby("cls").r.mean().to_frame().T.assign(what="token_role_r")], axis=0)
    summ.to_csv(S / "summary.tsv", sep="\t", index=False)
    print(mix.describe().round(3).to_string())
    print(rr.groupby("cls").r.mean().round(3).to_string())
    write_provenance("16c_scale_diagnostics", cfg, [], [S / "summary.tsv"])


if __name__ == "__main__":
    main()
