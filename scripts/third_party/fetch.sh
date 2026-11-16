#!/usr/bin/env bash
# Fetch the published methods compared in the study, at the commits that were used, and the HIPT weights iStar needs.
# Their code is not redistributed here (iStar GPL-3.0, BLEEP Apache-2.0, Hist2ST without a licence file).
set -euo pipefail
cd "$(dirname "$0")"
get() { [ -d "$1/.git" ] || git clone -q "$2" "$1"; git -C "$1" checkout -q "$3"; }
get istar   https://github.com/daviddaiweizhang/istar 3cb0e5352a86df337f41c5841d0808da0003457b
get bleep   https://github.com/bowang-lab/BLEEP       23959670c7407510c1716a173184bdcc18833a35
get hist2st https://github.com/biomed-AI/Hist2ST      7480e5d1f7712282e6c96d8430684e7830517769
# HIPT DINO weights (https://github.com/mahmoodlab/HIPT, HIPT_4K/Checkpoints)
mkdir -p istar/checkpoints
for f in vit256_small_dino.pth vit4k_xs_dino.pth; do
  [ -s "istar/checkpoints/$f" ] || curl -sSL -o "istar/checkpoints/$f" "https://github.com/mahmoodlab/HIPT/raw/master/HIPT_4K/Checkpoints/$f"
done
md5sum -c <<MD5
fea69d926d14eb86107c185a1c02f85b  istar/checkpoints/vit256_small_dino.pth
ba9a1952515fafeff027d165fa910d0f  istar/checkpoints/vit4k_xs_dino.pth
MD5
