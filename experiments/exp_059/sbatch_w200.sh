#!/bin/bash
# =============================================================================
# exp_059 w200 — クラスタ実行の「器」（design.md §3）。
#   蒸留係数 λ 感度分析（Ours・旧バッチ [4,1,1]×6）。6 ドメイン逐次。
#   ★ λ は config に焼き込み済み。--cfg-options で loss_weight を渡さない
#     （exp_033 事故防止。design §1.1）。
#   使い方（マスターノードで）: sbatch experiments/exp_059/sbatch_w200.sh
#   ★ node03 は /local_cache が無く落ちるため除外。
#   ※ 実行は design.md の承認後（行動原理3。2026-09-01 承認済み）。
# =============================================================================
#SBATCH --job-name=exp059_w200
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
REPO="${REPO:-/home/kouyou/VLM_OD_CL_exp059}"
SIF=/home/kouyou/sif/docker-image-of-mmdetection4singularity.sif
CACHE=/local_cache/${SLURM_JOB_ID}/datasets

mkdir -p "$CACHE"
cp -r "$HOME_DATA/rf100_domain" "$CACHE/"

echo "== local_cache staging =="
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}   CACHE = $CACHE   exp_059 w200"
df -h /local_cache 2>/dev/null | tail -1

singularity exec --nv \
  --bind "$REPO":/workspace/kouyou/mmdetection \
  --bind "$CACHE/rf100_domain":/workspace/kouyou/datasets/rf100_domain \
  --bind "$HOME_DATA/o365v1_stage":/workspace/kouyou/datasets/o365v1_stage \
  --bind "$HOME_DATA/bert-base-uncased":/workspace/kouyou/datasets/bert-base-uncased \
  --bind /dataset01/MSCOCO:/workspace/kouyou/datasets/coco2017 \
  --bind /home/kouyou/ckpt:/workspace/kouyou/ckpt \
  "$SIF" bash -c "COND=w200 GPUS=4 bash /workspace/kouyou/mmdetection/experiments/exp_059/run_sequential.sh"
