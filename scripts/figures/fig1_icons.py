#!/usr/bin/env python3
"""Figure 1, step 1: one illustrated pictogram per API call, traced to colour SVG.

Each pictogram is text-free on a white background; all labels, arrows, the token grid, the neighbour lattice
and the attention map are drawn in Fig1_design.py, which reads only the saved icons.

    GPTIMAGE_KEY=... python scripts/figures/fig1_icons.py [--only slide encoder ...] [--trace-only]

Output: figures/fig1_icons/<name>.png (as generated), <name>.svg (traced), prompts.json
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures" / "fig1_icons"
BASE = os.environ.get("GPTIMAGE_BASE_URL", "https://cpa.onelyi.com/v1")
MODEL = os.environ.get("GPTIMAGE_MODEL", "gpt-image-2")

STYLE = ("Single scientific illustration icon in BioRender style for a journal figure: clean flat shapes, "
         "soft muted colours with gentle shading, thin dark-grey outline, centred with a generous margin, "
         "plain pure white background, no shadow on the ground. Absolutely no text, no letters, no numbers, "
         "no labels, no arrows, no watermark.")

ICONS = {
    "slide": "A glass microscope slide lying flat, seen from a slight angle, carrying one elongated kidney "
             "needle-biopsy tissue section stained pink and purple with haematoxylin and eosin.",
    "histology": "A square close-up of haematoxylin and eosin stained human kidney cortex: several round "
                 "proximal tubule cross-sections with pink granular epithelium, purple nuclei and small open "
                 "lumens, thin pale interstitium between them, filling the whole square edge to edge.",
    "encoder": "A pretrained deep vision transformer encoder drawn as a stack of six thin rounded translucent "
               "layers in cool light grey, slightly offset in depth, with a small blue snowflake symbol on the "
               "front layer to indicate frozen weights.",
    "network": "A compact neural network: three columns of round nodes connected by thin lines, nodes in soft "
               "orange and peach.",
    "kidney": "A human kidney in front view with a small cut-away showing the cortex and medulla, soft "
              "reddish-brown.",
    "tubule": "A single kidney proximal tubule cross-section: a ring of cuboidal epithelial cells with a brush "
              "border facing the central lumen, pink cytoplasm and purple round nuclei, surrounded by a few "
              "small interstitial cells.",
    "cells": "Single-cell spatial transcriptomics map: hundreds of tiny dots of different soft colours (blue, "
             "green, orange, red, purple) forming a kidney tissue section, with one ring of light-blue dots "
             "outlining a tubule in the middle.",
    "rna": "A short single-stranded messenger RNA molecule drawn as a smooth curved ribbon with coloured "
           "nucleotide bases, soft purple and teal.",
    "cohort": "A group of five simple human figures standing side by side, front view, neutral soft colours.",
    "hospital": "A modern hospital building, front view, white and light blue, with a small red cross sign.",
    "map": "A thin elongated kidney biopsy tissue section seen from above, coloured as a smooth spatial heat "
           "map from pale yellow through orange to deep red, flat on white.",
}

TRACE = dict(colormode="color", hierarchical="stacked", mode="spline", filter_speckle=6,
             color_precision=6, layer_difference=16, corner_threshold=60, length_threshold=4.0,
             max_iterations=10, splice_threshold=45, path_precision=3)


def generate(name: str, key: str) -> str:
    prompt = f"{STYLE} Subject: {ICONS[name]}"
    body = json.dumps({"model": MODEL, "prompt": prompt, "size": "1024x1024", "quality": "high",
                       "n": 1}).encode()
    req = urllib.request.Request(f"{BASE}/images/generations", data=body,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    (OUT / f"{name}.png").write_bytes(base64.b64decode(d["data"][0]["b64_json"]))
    print(f"generated {name}", flush=True)
    return prompt


def trace(name: str) -> None:
    import vtracer
    vtracer.convert_image_to_svg_py(str(OUT / f"{name}.png"), str(OUT / f"{name}.svg"), **TRACE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--trace-only", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    names = a.only or list(ICONS)
    if not a.trace_only:
        key = os.environ["GPTIMAGE_KEY"]
        with ThreadPoolExecutor(6) as ex:
            prompts = dict(zip(names, ex.map(lambda n: generate(n, key), names)))
        rec = OUT / "prompts.json"
        old = json.loads(rec.read_text()) if rec.exists() else {}
        rec.write_text(json.dumps({**old, **prompts}, indent=1))
    for n in names:
        trace(n)
    print("traced:", ", ".join(names))


if __name__ == "__main__":
    main()
