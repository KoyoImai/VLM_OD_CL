#!/bin/bash
# =============================================================================
# exp_036_d sbatch.sh — クラスタ実行の「器」。
#   note07 の分割に対応: exp_036_d = 蒸留D（融合後）+ 蒸留E（3項平均）、λ = 3.0
#   exp_036 の6ジョブは独立に投入できる（各2条件×32h ≒ 64h）。同時実行上限は4。
#
#   使い方（マスターノードで）:
#     RUN_STAGE=D sbatch experiments/exp_036/sbatch_d.sh
#     RUN_STAGE=E sbatch experiments/exp_036/sbatch_d.sh
#     sbatch experiments/exp_036/sbatch_d.sh          # all（2条件を直列・約64h）
#
#   ★ リプレイを使うため Objects365 の bind が必要（exp_027 と同じ）。
#   ★ node03 は /local_cache が用意されず mkdir で落ちるため除外する（2026-07-25）。
#   ※ 実行は design.md の承認後（行動原理3）。
# =============================================================================
#SBATCH --job-name=exp036d_kdDE_condA_l2w30
#SBATCH --partition=a6000_ada
#SBATCH --exclude=node03
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=120:00:00
#SBATCH --output=/home/kouyou/logs/result_%x_%j.txt
#SBATCH --error=/home/kouyou/logs/error_%x_%j.txt

#SLACK: notify-start
#SLACK: notify-end
#SLACK: notify-error
set -e

HOME_DATA=/home/kouyou/datasets
REPO=/home/kouyou/VLM_OD_CL
SIF=/home/kouyou/sif/docker-image-of-mmdetection4singularity.sif
CACHE=/local_cache/${SLURM_JOB_ID}/datasets
RUN_STAGE="${RUN_STAGE:-all}"

mkdir -p "$CACHE"
cp -r "$HOME_DATA/rf100_domain" "$CACHE/"

echo "== local_cache staging =="
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}   CACHE = $CACHE   RUN_STAGE = $RUN_STAGE"
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
  "$SIF" bash /workspace/kouyou/mmdetection/experiments/exp_036/train_val_d.sh "$RUN_STAGE"
