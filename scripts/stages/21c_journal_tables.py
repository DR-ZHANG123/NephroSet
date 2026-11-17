"""Component ablation and computational cost tables of NephroSet (Tables 3 and 4 of the journal versions).

ablation.tsv : for each model, which NephroSet components it contains, its cross-validated agreement for
               PT injury and transport, and its paired difference from the spot-only network (participant
               bootstrap, as in 16_evaluate_cv).
cost.tsv     : trainable parameters, inference time per section and training time per model (21b).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.evaluate import paired  # noqa: E402
from lib.repro import load_config, p, stage_dir, write_provenance  # noqa: E402

cfg = load_config()
S = stage_dir("21c_journal_tables")
# label, model, fine-scale tokens, token summary, neighbourhood set, anatomical hyperedges
ROWS = [("Spot only", "mlp", "–", "–", "×", "×"),
        ("Neighbourhood set", "deepsets", "–", "–", "✓", "×"),
        ("Token mean", "tok_mean", "All 196", "Mean", "×", "×"),
        ("Token attention", "tok_abmil", "All 196", "Attention", "×", "×"),
        ("Token attention, capture area", "tok_abmil_disc", "Inside 32", "Attention", "×", "×"),
        ("Token attention, crop edge", "tok_abmil_outer32", "Outermost 32", "Attention", "×", "×"),
        ("Anatomical token hypergraph", "tok_hyper_anat", "All 196", "Hypergraph", "×", "✓"),
        ("Anatomical token hypergraph + set", "tok_hyper_anat_ctx", "All 196", "Hypergraph", "✓", "✓"),
        ("NephroSet", "tok_abmil_ctx", "All 196", "Attention", "✓", "×")]
COST = [("Spot only", "mlp"), ("Neighbourhood set", "deepsets"), ("Token mean", "tok_mean"),
        ("Token attention", "tok_abmil"), ("Token graph", "tok_gat"), ("Anatomical token hypergraph", "tok_hyper_anat"),
        ("Anatomical token hypergraph + set", "tok_hyper_anat_ctx"), ("NephroSet", "tok_abmil_ctx")]


def main() -> None:
    pm = pd.read_csv(p("results/16_evaluate_cv/participant_metrics.tsv"), sep="\t")
    rng = np.random.default_rng(cfg["seed"])
    out = []
    for lab, m, tok, summ, ctx, anat in ROWS:
        r = dict(label=lab, model=m, tokens=tok, token_summary=summ, neighbourhood_set=ctx, anatomical_hyperedges=anat)
        for k in ["pt_injury", "pt_transport"]:
            a = pm[(pm.model == m) & (pm.programme == k)].set_index("participant").rho
            b = pm[(pm.model == "mlp") & (pm.programme == k)].set_index("participant").rho
            r[f"{k}_rho"] = a.mean()
            if m != "mlp":
                d = paired(a, b, 2000, rng)
                r[f"{k}_diff"], r[f"{k}_lo"], r[f"{k}_hi"] = d["diff"], d["lo"], d["hi"]
        out.append(r)
    A = pd.DataFrame(out)
    A.to_csv(S / "ablation.tsv", sep="\t", index=False)
    c = pd.read_csv(p("results/21b_model_cost/model_cost.tsv"), sep="\t").set_index("model")
    C = pd.DataFrame([dict(label=lab, model=m, parameters_m=c.loc[m, "parameters"] / 1e6,
                           inference_ms=c.loc[m, "inference_s_per_section"] * 1e3,
                           train_min=c.loc[m, "train_min_per_model"]) for lab, m in COST])
    C.to_csv(S / "cost.tsv", sep="\t", index=False)
    print(A.round(3).to_string())
    print(C.round(3).to_string())
    write_provenance("21c_journal_tables", cfg, [], [S / "ablation.tsv", S / "cost.tsv"])


if __name__ == "__main__":
    main()
