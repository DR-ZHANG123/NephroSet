"""Table 1 (agreement with measured programme scores) for the Word manuscripts.

Pandoc tables have a single header row, so the table is written into the converted .docx with python-docx:
a grouped two-row header (PT injury | PT transport over cohort and adjustment), one row per model, NephroSet and
the best image model in each column in bold, booktabs-style rules and a header that repeats on a new page.
Usage: put table_markdown(...) where the table belongs before pandoc, then call insert_table(docx) afterwards.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
MARK = "WORDTABLEONEINSERT"
NAMES = {"gbm_measured_comp": "Measured compositionᵃ", "gbm_comp_morph": "Image composition and morphology",
         "ridge_image": "Linear probe", "mlp": "Spot only", "gcn": "Lattice graph (GCN)",
         "set_anat": "Anatomical set hypergraph", "deepsets": "Neighbourhood set", "tok_abmil": "Token attention",
         "tok_hyper_anat": "Anatomical token hypergraph", "tok_abmil_ctx": "NephroSet", "istar": "iStar",
         "bleep": "BLEEP", "hist2st": "Hist2ST", "knn_encoder": "Retrieval, same encoderᵇ",
         "tok_pred_mean": "Token-level prediction, same encoderᶜ"}
COLS = [f"{k}_{c}_{m}" for k in ["pt_injury", "pt_transport"] for c in ["KPMP", "External"]
        for m in ["rho", "rho_beyond_comp"]]
SUB = ["KPMP", "KPMP, adj.", "Second centre", "Second centre, adj."]


def table_markdown(intervals: str) -> str:
    cap = ("**Table 1. Agreement between estimated and measured programme scores in held-out KPMP participants and "
           "at the second centre.** Values are within-section Spearman correlations, averaged per participant (KPMP, "
           "*n* = 201) or donor (second centre, *n* = 13). Each neural model is the ensemble of three training seeds, "
           "in both cohorts. adj., after regressing measured composition out of both scores within the section. Bold, "
           "NephroSet and the highest value among image models in each column. Intervals and the inflammatory "
           f"programme are given in {intervals}. ^a^Reference model given the sequencing-derived composition of each "
           "spot, which is not available from an image. ^b^Prediction rule of BLEEP (mean of the 50 most similar "
           "training spots) on the image encoder of this study. ^c^Prediction rule of iStar (mean of token-level "
           "predictions within the spot) on the image encoder of this study.")
    return f'::: {{custom-style="Table Caption"}}\n{cap}\n:::\n\n{MARK}\n'


def _border(cell, edge: str, sz: int = 8) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    b = tcPr.find(qn("w:tcBorders"))
    if b is None:
        b = OxmlElement("w:tcBorders")
        tcPr.append(b)
    e = OxmlElement(f"w:{edge}")
    for k, v in (("w:val", "single"), ("w:sz", str(sz)), ("w:space", "0"), ("w:color", "000000")):
        e.set(qn(k), v)
    b.append(e)


def _write(cell, text: str, bold=False, align=WD_ALIGN_PARAGRAPH.RIGHT) -> None:
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(1)
    r = p.add_run(text)
    r.bold = bold
    r.font.size = Pt(8)


def insert_table(docx: Path) -> None:
    tab = pd.read_csv(ROOT / "results/21_tables/table2.tsv", sep="\t")
    image = tab.model != "gbm_measured_comp"
    best = {c: tab.loc[image, c].round(2).max() for c in COLS}
    fmt = lambda v: f"{v:.2f}".replace("-0.00", "0.00").replace("-", "−")  # noqa: E731
    doc = Document(str(docx))
    mark = next((p for p in doc.paragraphs if p.text.strip() == MARK), None)
    if mark is None:  # documents without Table 1 (title page)
        return
    t = doc.add_table(rows=len(tab) + 2, cols=9)
    widths = [Inches(1.85)] + [Inches(0.56)] * 8
    for row in t.rows:
        for c, w in zip(row.cells, widths):
            c.width = w
    t.cell(0, 0).merge(t.cell(1, 0))
    _write(t.cell(0, 0), "Model", align=WD_ALIGN_PARAGRAPH.LEFT)
    for j, lab in ((1, "PT injury"), (5, "PT transport")):
        t.cell(0, j).merge(t.cell(0, j + 3))
        _write(t.cell(0, j), lab, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
        for k in range(4):
            _border(t.cell(0, j + k), "bottom", 4)
    for j in range(8):
        _write(t.cell(1, j + 1), SUB[j % 4])
    for i, r in tab.iterrows():
        nep = r.model == "tok_abmil_ctx"
        _write(t.cell(i + 2, 0), NAMES[r.model], bold=nep, align=WD_ALIGN_PARAGRAPH.LEFT)
        for j, c in enumerate(COLS):
            _write(t.cell(i + 2, j + 1), fmt(r[c]), bold=image[i] and round(r[c], 2) == best[c])
    for j in range(9):
        _border(t.cell(0, j), "top", 10)
        _border(t.cell(1, j), "bottom", 6)
        _border(t.cell(len(tab) + 1, j), "bottom", 10)
    for row in t.rows[:2]:  # header repeats on a new page
        trPr = row._tr.get_or_add_trPr()
        h = OxmlElement("w:tblHeader")
        h.set(qn("w:val"), "true")
        trPr.append(h)
    mark._p.addnext(t._tbl)
    mark._p.getparent().remove(mark._p)
    doc.save(str(docx))
