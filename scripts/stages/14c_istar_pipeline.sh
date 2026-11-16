#!/bin/bash
# iStar baseline driver: wait for KPMP features, encode GSE211785, then CV (GPU 0) and external (GPU 1).
cd "$(dirname "$0")/../.."
eval "$(conda shell.bash hook)"; conda activate nephroset
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while [ "$(pgrep -u "$USER" -fc '[s]tages/14c_istar.py --extract')" != "0" ]; do sleep 60; done
echo "kpmp features: $(ls data/cache/istar_kpmp | wc -l)  $(date +%H:%M)"
CUDA_VISIBLE_DEVICES=0 python -u scripts/stages/14c_istar.py --extract gse211785 --part 0/2 > logs/14c_istar_extract_gse211785_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u scripts/stages/14c_istar.py --extract gse211785 --part 1/2 > logs/14c_istar_extract_gse211785_1.log 2>&1 &
# CV needs only KPMP features: start as soon as all KPMP sections are present (second pass picks up any missing ones)
CUDA_VISIBLE_DEVICES=0 python -u scripts/stages/14c_istar.py --extract kpmp > logs/14c_istar_extract_kpmp_final.log 2>&1
wait
echo "gse features: $(ls data/cache/istar_gse211785 | wc -l)  $(date +%H:%M)"
CUDA_VISIBLE_DEVICES=0 python -u scripts/stages/14c_istar.py --resume > logs/14c_istar_cv.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u scripts/stages/15c_istar_external.py --seeds10 > logs/14c_istar_external_s10.log 2>&1 &
wait
echo "all done $(date +%H:%M)"
