#!/bin/bash
# =============================================================================
# exp_023.5 sbatch.sh — クラスタ実行の「器」（案A）。
#   #SBATCH（Slurm）＋ rf100 の local_cache ステージング＋ Singularity 起動を担い、
#   コンテナ内で train_val.sh を実行する。学習/評価ロジックは train_val.sh 側。
#
#   使い方（マスターノードで）:
#     sbatch experiments/exp_023.5/sbatch.sh                    # all（ゼロショット→1ep学習→評価）
#     RUN_STAGE=zeroshot sbatch experiments/exp_023.5/sbatch.sh # ゼロショットのみ（最速）
#     RUN_STAGE=train    sbatch experiments/exp_023.5/sbatch.sh # 1ep学習のみ
#
#   事前準備:
#     - mkdir -p /home/kouyou/logs
#     - θ0 を /home/kouyou/ckpt/ に配置（下記 URL を wget、README §2 参照。train_val.sh が明示パスで渡す）
#         https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/
#           grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/
#           grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth
#     - データ転送済み（README §4: rf100_domain, o365v1_stage）
#     - コード同期済み（README §3: git pull）
# =============================================================================
#SBATCH --job-name=exp0235_debug
#SBATCH --partition=a6000_ada
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64            # node01-04 は 64 スレッド。≤64
#SBATCH --time=04:00:00              # デバッグ（ゼロショット＋1ep＋評価）。フル20epは行わない
#SBATCH --output=/home/kouyou/logs/result_%x_%j.txt
#SBATCH --error=/home/kouyou/logs/error_%x_%j.txt

#SLACK: notify-start
#SLACK: notify-end
#SLACK: notify-error
set -e

HOME_DATA=/home/kouyou/datasets
REPO=/home/kouyou/VLM_OD_CL          # クラスタ上の repo クローン先（git remote 名 = VLM_OD_CL）
SIF=/home/kouyou/sif/docker-image-of-mmdetection4singularity.sif
CACHE=/local_cache/${SLURM_JOB_ID}/datasets
RUN_STAGE="${RUN_STAGE:-all}"        # all|zeroshot|train|eval を train_val.sh へ渡す

# ① rf100 のみ local_cache へステージング（o365 は home から bind、§6.1 の方針）
mkdir -p "$CACHE"
cp -r "$HOME_DATA/rf100_domain" "$CACHE/"

# ステージング検証（local_cache が動いているかをログに残す）
echo "== local_cache staging =="
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}   CACHE = $CACHE"
df -h /local_cache 2>/dev/null | tail -1
du -sh "$CACHE"/* 2>/dev/null
echo "=========================="

# ② コンテナ起動 → train_val.sh 実行（--bind でパス間接化。config 無変更）
#    各データソースを /workspace/kouyou/datasets 直下の独立した葉として bind する
#    （親 datasets を丸ごと bind して入れ子にしない＝Tier1 で実証済みの構造に一致させる）。
#    rf100 は local_cache から、o365/bert/coco はホーム・共有SSD から直接。
singularity exec --nv \
  --bind "$REPO":/workspace/kouyou/mmdetection \
  --bind "$CACHE/rf100_domain":/workspace/kouyou/datasets/rf100_domain \
  --bind "$HOME_DATA/o365v1_stage":/workspace/kouyou/datasets/o365v1_stage \
  --bind "$HOME_DATA/bert-base-uncased":/workspace/kouyou/datasets/bert-base-uncased \
  --bind /dataset01/MSCOCO:/workspace/kouyou/datasets/coco2017 \
  --bind /home/kouyou/ckpt:/workspace/kouyou/ckpt \
  "$SIF" bash /workspace/kouyou/mmdetection/experiments/exp_023.5/train_val.sh "$RUN_STAGE"
