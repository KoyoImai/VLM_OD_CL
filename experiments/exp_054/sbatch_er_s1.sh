#!/bin/bash
# =============================================================================
# exp_054 er seed=1 — クラスタ実行の「器」（design.md §3）。
#   seed 頑健性確認（6 手法 × seed 1,2 の 12 ジョブの 1 本）。6 ドメイン逐次。
#   使い方（マスターノードで）: sbatch experiments/exp_054/sbatch_er_s1.sh
#   ★ node03 は /local_cache が無く落ちるため除外。
#   ※ 実行は design.md の承認後（行動原理3。2026-08-31 承認済み）。
# =============================================================================
#SBATCH --job-name=exp054_er_s1
#SBATCH --partition=a6000_ada
#SBATCH --exclude=node03
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=144:00:00
#SBATCH --output=/home/kouyou/logs/result_%x_%j.txt
#SBATCH --error=/home/kouyou/logs/error_%x_%j.txt

#SLACK: notify-start
#SLACK: notify-end
#SLACK: notify-error
set -e

HOME_DATA=/home/kouyou/datasets
REPO="${REPO:-/home/kouyou/VLM_OD_CL_exp054}"
SIF=/home/kouyou/sif/docker-image-of-mmdetection4singularity.sif
CACHE=/local_cache/${SLURM_JOB_ID}/datasets

mkdir -p "$CACHE"
cp -r "$HOME_DATA/rf100_domain" "$CACHE/"

echo "== local_cache staging =="
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}   CACHE = $CACHE   exp_054 er seed=1"
df -h /local_cache 2>/dev/null | tail -1

singularity exec --nv \
  --bind "$REPO":/workspace/kouyou/mmdetection \
  --bind "$CACHE/rf100_domain":/workspace/kouyou/datasets/rf100_domain \
  --bind "$HOME_DATA/o365v1_stage":/workspace/kouyou/datasets/o365v1_stage \
  --bind "$HOME_DATA/bert-base-uncased":/workspace/kouyou/datasets/bert-base-uncased \
  --bind /dataset01/MSCOCO:/workspace/kouyou/datasets/coco2017 \
  --bind /home/kouyou/ckpt:/workspace/kouyou/ckpt \
  "$SIF" bash -c "SEED=1 GPUS=4 bash /workspace/kouyou/mmdetection/experiments/exp_054/run_er.sh"
