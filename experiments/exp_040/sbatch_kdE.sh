#!/bin/bash
# =============================================================================
# exp_040 kdE — クラスタ実行の「器」（design.md §5）。
#   後半3ドメイン（aerial -> microscopic -> documents）を逐次で流す。
#   3条件は**別ジョブなので並列に投入できる**（同時実行4ジョブ制限内。design.md §5.2）。
#
#   使い方（マスターノードで）:
#     REPO=<clone先> sbatch experiments/exp_040/sbatch_kdE.sh
#
#   ★ リプレイ／蒸留は Objects365 の bind が必要（exp_027 / exp_028 と同じ）。
#   ★ node03 は /local_cache が用意されず mkdir で落ちるため除外する。
#   ※ 実行は design.md の承認後（行動原理3）。
# =============================================================================
#SBATCH --job-name=exp040_kdE
#SBATCH --partition=a6000_ada
#SBATCH --exclude=node03
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=72:00:00
#SBATCH --output=/home/kouyou/logs/result_%x_%j.txt
#SBATCH --error=/home/kouyou/logs/error_%x_%j.txt

#SLACK: notify-start
#SLACK: notify-end
#SLACK: notify-error
set -e

HOME_DATA=/home/kouyou/datasets
# clone 先を変えたい場合は投入時に REPO=... で上書きする
REPO="${REPO:-/home/kouyou/VLM_OD_CL}"
SIF=/home/kouyou/sif/docker-image-of-mmdetection4singularity.sif
CACHE=/local_cache/${SLURM_JOB_ID}/datasets

mkdir -p "$CACHE"
cp -r "$HOME_DATA/rf100_domain" "$CACHE/"

echo "== local_cache staging =="
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}   CACHE = $CACHE   COND = kdE"
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
  "$SIF" bash -c "COND=kdE GPUS=4 bash /workspace/kouyou/mmdetection/experiments/exp_040/run_sequential.sh"
