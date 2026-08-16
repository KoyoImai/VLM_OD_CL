#!/bin/bash
# =============================================================================
# exp_043 replay — クラスタ実行の「器」（design.md §5）。
#   6 ドメイン（underwater -> ... -> documents）を逐次で流す（約 75〜85 h）。
#   2 条件は**別ジョブなので並列に投入できる**（同時実行 4 ジョブ制限内）。
#
#   使い方（マスターノードで）:
#     sbatch experiments/exp_043/sbatch_replay.sh
#   別のクローンから走らせる場合のみ REPO=<パス> を付ける。
#
#   ★ リプレイに Objects365、ZCOCO に MSCOCO の bind が必要（exp_040 と同じ）。
#   ★ kdE の t=1 の教師 θ0 は /home/kouyou/ckpt の bind から解決される。
#   ★ node03 は /local_cache が用意されず mkdir で落ちるため除外する。
#   ※ 実行は design.md の承認後（行動原理3。2026-08-16 承認済み）。
# =============================================================================
#SBATCH --job-name=exp043_replay
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
# 作業ディレクトリは exp_043 専用クローン（design.md §8。2026-08-16 指示）
REPO="${REPO:-/home/kouyou/VLM_OD_CL_exp043}"
SIF=/home/kouyou/sif/docker-image-of-mmdetection4singularity.sif
CACHE=/local_cache/${SLURM_JOB_ID}/datasets

mkdir -p "$CACHE"
cp -r "$HOME_DATA/rf100_domain" "$CACHE/"

echo "== local_cache staging =="
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}   CACHE = $CACHE   COND=replay"
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
  "$SIF" bash -c "COND=replay GPUS=4 bash /workspace/kouyou/mmdetection/experiments/exp_043/run_sequential.sh"
