#!/bin/bash
# =============================================================================
# exp_047 lora — クラスタ実行の「器」（design.md §5）。
#   RF100 6 ドメイン逐次・バッファ不使用（約 30〜38 h）。
#   2 条件は別ジョブなので並列に投入できる（同時実行 4 本制限。超過分はキュー待ち）。
#
#   使い方（マスターノードで）:
#     sbatch experiments/exp_045/sbatch_inflora.sh
#   別のクローンから走らせる場合のみ REPO=<パス> を付ける。
#
#   ★ ZCOCO 評価に MSCOCO、θ0 に /home/kouyou/ckpt の bind が必要。
#   ★ node03 は /local_cache が用意されず mkdir で落ちるため除外する。
#   ※ 実行は design.md の承認後（行動原理3。2026-08-16 承認済み）。
# =============================================================================
#SBATCH --job-name=exp047_lora
#SBATCH --partition=a6000_ada
#SBATCH --exclude=node03
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=96:00:00
#SBATCH --output=/home/kouyou/logs/result_%x_%j.txt
#SBATCH --error=/home/kouyou/logs/error_%x_%j.txt

#SLACK: notify-start
#SLACK: notify-end
#SLACK: notify-error
set -e

HOME_DATA=/home/kouyou/datasets
# 作業ディレクトリは exp_047 専用クローン（design.md §5。2026-08-16 前提）
REPO="${REPO:-/home/kouyou/VLM_OD_CL_exp047}"
SIF=/home/kouyou/sif/docker-image-of-mmdetection4singularity.sif
CACHE=/local_cache/${SLURM_JOB_ID}/datasets

mkdir -p "$CACHE"
cp -r "$HOME_DATA/rf100_domain" "$CACHE/"

echo "== local_cache staging =="
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}   CACHE = $CACHE   exp_047 lora"
df -h /local_cache 2>/dev/null | tail -1
du -sh "$CACHE"/* 2>/dev/null
echo "=========================="

singularity exec --nv \
  --bind "$REPO":/workspace/kouyou/mmdetection \
  --bind "$CACHE/rf100_domain":/workspace/kouyou/datasets/rf100_domain \
  --bind "$HOME_DATA/bert-base-uncased":/workspace/kouyou/datasets/bert-base-uncased \
  --bind /dataset01/MSCOCO:/workspace/kouyou/datasets/coco2017 \
  --bind /home/kouyou/ckpt:/workspace/kouyou/ckpt \
  "$SIF" bash -c "GPUS=4 bash /workspace/kouyou/mmdetection/experiments/exp_047/run_lora.sh"
