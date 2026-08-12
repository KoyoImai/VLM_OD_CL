#!/bin/bash
# =============================================================================
# exp_039 ZiRa — クラスタ実行の「器」（design.md §5）。
#   リプレイ無し → 有り を直列に流す（約 54 h）。DitHub は sbatch_dithub.sh で
#   **別ジョブとして並列に投入できる**（同時実行 4 ジョブ制限内）。
#
#   使い方（マスターノードで）:
#     sbatch experiments/exp_039/sbatch_zira.sh                    # 2 条件を直列
#     RUN_STAGE=replayfree sbatch experiments/exp_039/sbatch_zira.sh
#     RUN_STAGE=replay     sbatch experiments/exp_039/sbatch_zira.sh
#
#   ★ リプレイを使うため Objects365 の bind が必要（exp_027 / exp_028 と同じ）。
#   ★ node03 は /local_cache が用意されず mkdir で落ちるため除外する。
#   ※ 実行は design.md の承認後（行動原理3）。
# =============================================================================
#SBATCH --job-name=exp039_zira
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
# （exp_036 実行中に別クローンから走らせる場合など。README.md §2）
REPO="${REPO:-/home/kouyou/VLM_OD_CL}"
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
  "$SIF" bash /workspace/kouyou/mmdetection/experiments/exp_039/run_stages.sh \
  zira "$RUN_STAGE"
