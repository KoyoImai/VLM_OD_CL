#!/bin/bash
# =============================================================================
# exp_048 kdEall — クラスタ実行の「器」（design.md §5）。
#   条件2: リプレイ [現在4, 汎用1, 過去1]×6・全6枚に蒸留E。6 ドメイン逐次
#   （見積り 80〜110 h）。条件1（sbatch_kdEonly.sh）とは別ジョブなので並列に投入できる。
#
#   使い方（マスターノードで）:
#     sbatch experiments/exp_048/sbatch_kdEall.sh
#   別のクローンから走らせる場合のみ REPO=<パス> を付ける。
#
#   ★ リプレイに Objects365、ZCOCO に MSCOCO、t=1 の教師 θ0 に /home/kouyou/ckpt の
#     bind が必要（exp_043 と同じ）。
#   ★ node03 は /local_cache が用意されず mkdir で落ちるため除外する。
#   ※ 実行は design.md の承認後（行動原理3。2026-08-20 承認済み）。
# =============================================================================
#SBATCH --job-name=exp048_kdEall
#SBATCH --partition=a6000_ada
#SBATCH --exclude=node03
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=168:00:00
#SBATCH --output=/home/kouyou/logs/result_%x_%j.txt
#SBATCH --error=/home/kouyou/logs/error_%x_%j.txt

#SLACK: notify-start
#SLACK: notify-end
#SLACK: notify-error
set -e

HOME_DATA=/home/kouyou/datasets
# 作業ディレクトリは exp_048 専用クローン（投入時に無ければ README §2 の手順で作る）
REPO="${REPO:-/home/kouyou/VLM_OD_CL_exp048}"
SIF=/home/kouyou/sif/docker-image-of-mmdetection4singularity.sif
CACHE=/local_cache/${SLURM_JOB_ID}/datasets

mkdir -p "$CACHE"
cp -r "$HOME_DATA/rf100_domain" "$CACHE/"

echo "== local_cache staging =="
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}   CACHE = $CACHE   exp_048 kdEall"
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
  "$SIF" bash -c "GPUS=4 bash /workspace/kouyou/mmdetection/experiments/exp_048/run_kdEall.sh"
