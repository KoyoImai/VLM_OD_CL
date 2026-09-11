#!/bin/bash
# exp_061 er b3 — クラスタ実行の器（design.md 承認後）。
#   バッファサイズ感度（内訳 base）。6 ドメイン逐次。
#   ★ config は exp_058/buffer を参照する。クローンに exp_058/buffer を必ず含めること（README §0）。
#SBATCH --job-name=exp061_er_b3
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
REPO="${REPO:-/home/kouyou/VLM_OD_CL_exp061}"
SIF=/home/kouyou/sif/docker-image-of-mmdetection4singularity.sif
CACHE=/local_cache/${SLURM_JOB_ID}/datasets
mkdir -p "$CACHE"
cp -r "$HOME_DATA/rf100_domain" "$CACHE/"
echo "SLURM_JOB_ID=${SLURM_JOB_ID} CACHE=$CACHE  exp_061 er b3"
singularity exec --nv \
  --bind "$REPO":/workspace/kouyou/mmdetection \
  --bind "$CACHE/rf100_domain":/workspace/kouyou/datasets/rf100_domain \
  --bind "$HOME_DATA/o365v1_stage":/workspace/kouyou/datasets/o365v1_stage \
  --bind "$HOME_DATA/bert-base-uncased":/workspace/kouyou/datasets/bert-base-uncased \
  --bind /dataset01/MSCOCO:/workspace/kouyou/datasets/coco2017 \
  --bind /home/kouyou/ckpt:/workspace/kouyou/ckpt \
  "$SIF" bash -c "METHOD=er COND=b3 GPUS=4 bash /workspace/kouyou/mmdetection/experiments/exp_061/run_sequential.sh"
