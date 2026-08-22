#!/bin/bash
# =============================================================================
# exp_052 decoder — クラスタ実行の「器」（design.md §5）。
#   蒸留E × 学習可能モジュールアブレーション（旧バッチ [4,1,1]×6・リプレイあり）。
#   前半 3 ドメイン逐次。5 条件は別ジョブ（同時実行 4 本制限。超過分はキュー待ち）。
#
#   使い方（マスターノードで）: sbatch experiments/exp_052/sbatch_decoder.sh
#   別のクローンから走らせる場合のみ REPO=<パス> を付ける。
#
#   ★ リプレイに Objects365、ZCOCO に MSCOCO、教師 θ0 に /home/kouyou/ckpt の bind が必要。
#   ★ node03 は /local_cache が用意されず mkdir で落ちるため除外する。
#   ※ 実行は design.md の承認後（行動原理3。2026-08-22 承認済み）。
# =============================================================================
#SBATCH --job-name=exp052_decoder
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
REPO="${REPO:-/home/kouyou/VLM_OD_CL_exp052}"
SIF=/home/kouyou/sif/docker-image-of-mmdetection4singularity.sif
CACHE=/local_cache/${SLURM_JOB_ID}/datasets

mkdir -p "$CACHE"
cp -r "$HOME_DATA/rf100_domain" "$CACHE/"

echo "== local_cache staging =="
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}   CACHE = $CACHE   exp_052 decoder"
df -h /local_cache 2>/dev/null | tail -1
du -sh "$CACHE"/* 2>/dev/null
echo "=========================="

singularity exec --nv \
  --bind "$REPO":/workspace/kouyou/mmdetection \
  --bind "$CACHE/rf100_domain":/workspace/kouyou/datasets/rf100_domain \
  --bind "$HOME_DATA/o365v1_stage":/workspace/kouyou/datasets/o365v1_stage \
  --bind "$HOME_DATA/bert-base-uncased":/workspace/kouyou/datasets/bert-base-uncased \
  --bind /dataset01/MSCOCO:/workspace/kouyou/datasets/coco2017 \
  --bind /home/kouyou/ckpt:/workspace/kouyou/ckpt \
  "$SIF" bash -c "COND=decoder GPUS=4 bash /workspace/kouyou/mmdetection/experiments/exp_052/run_sequential.sh"
