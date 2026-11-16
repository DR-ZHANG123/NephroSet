"""Spot-centred H&E crops at two physical scales, encoded with a frozen pathology encoder.

Scale 1 covers 112 um around the spot centre (0.5 um/px), scale 2 covers 448 um (2 um/px).
Simple colour-deconvolution descriptors of the scale-1 crop are kept alongside as a
transparent morphology baseline.
"""
from __future__ import annotations

import numpy as np
import torch
from skimage.color import rgb2hed
from skimage.measure import label
import cv2

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def crop(img: np.ndarray, r: float, c: float, half_px: float, out: int = 224) -> np.ndarray:
    h, w = img.shape[:2]
    r0, r1 = int(round(r - half_px)), int(round(r + half_px))
    c0, c1 = int(round(c - half_px)), int(round(c + half_px))
    pad = np.full((r1 - r0, c1 - c0, 3), 255, np.uint8)
    rr0, rr1, cc0, cc1 = max(r0, 0), min(r1, h), max(c0, 0), min(c1, w)
    if rr1 > rr0 and cc1 > cc0:
        pad[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0] = img[rr0:rr1, cc0:cc1, :3]
    return cv2.resize(pad, (out, out), interpolation=cv2.INTER_AREA)


def morphology(x: np.ndarray) -> np.ndarray:
    """Tissue fraction, haematoxylin/eosin optical density, nuclear blob density, lumen fraction."""
    f = x.astype(np.float32) / 255.0
    tissue = (f.mean(-1) < 0.85) | ((f.max(-1) - f.min(-1)) > 0.08)
    hed = rgb2hed(f)
    hch, ech = hed[..., 0], hed[..., 1]
    tf = tissue.mean()
    hm = hch[tissue].mean() if tissue.any() else 0.0
    em = ech[tissue].mean() if tissue.any() else 0.0
    thr = np.percentile(hch, 90)
    nuc = hch > max(thr, 0.05)
    lab = label(nuc)
    n_blobs = lab.max()
    lumen = (~tissue).mean()
    return np.array([tf, hm, em, n_blobs / 224.0, nuc.mean(), lumen, hch.std(), ech.std()], np.float32)


MORPH_NAMES = ["tissue_frac", "haem_od", "eosin_od", "nuclear_blobs", "nuclear_area", "lumen_frac",
               "haem_sd", "eosin_sd"]


class Encoder:
    def __init__(self, name: str, device: str):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(name).to(device).eval().half()
        self.device = device

    @torch.no_grad()
    def __call__(self, batch: np.ndarray) -> np.ndarray:
        x = (batch.astype(np.float32) / 255.0 - MEAN) / STD
        x = torch.from_numpy(x).permute(0, 3, 1, 2).to(self.device).half()
        out = self.model(pixel_values=x).last_hidden_state[:, 0]
        return out.float().cpu().numpy()
