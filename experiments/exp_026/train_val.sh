#!/bin/bash
# =============================================================================
# exp_026 train_val.sh — コンテナ内で走る学習評価ロジック（案A の「中身」）。
#   Slurm も Singularity も知らない。sbatch.sh から singularity exec 経由で呼ばれる。
#
#   段階（$1 / 既定 all）:
#     indiv_condA  個別チューニング（条件A＝全モジュール）3ドメイン
#     indiv_condB  個別チューニング（条件B＝特徴抽出+融合）3ドメイン
#     indiv_zira   個別 ZiRa 3ドメイン
#     oracle       オラクル（条件A・3ドメイン同時学習）
#     zeroshot     ゼロショット再確認（学習なし・6ドメイン＋COCO）
#     all          上を順に全部
# =============================================================================
set -euo pipefail
cd /workspace/kouyou/mmdetection

# HuggingFace/transformers を書込可キャッシュに向ける（.sif 焼込みの
# HF_HOME=/workspace/kouyou/datasets/HuggingFace は書込不可。BERT はローカルパスから読む）
export HF_HOME=/tmp/hf
export TRANSFORMERS_CACHE=/tmp/hf/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p /tmp/hf/hub

export PORT="${PORT:-29580}"

# θ0 を明示パスで渡す（exp_023.5 以来の方針。URL・キャッシュ状態に依存させない）
export THETA0=/workspace/kouyou/ckpt/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth

STAGE="${1:-all}"

case "$STAGE" in
  indiv_condA) bash experiments/exp_026/run_indiv.sh condA ;;
  indiv_condB) bash experiments/exp_026/run_indiv.sh condB ;;
  indiv_zira)  bash experiments/exp_026/run_indiv.sh zira ;;
  oracle)      bash experiments/exp_026/run_oracle.sh ;;
  zeroshot)    bash experiments/exp_026/run_zeroshot.sh ;;
  all)
    bash experiments/exp_026/run_zeroshot.sh          # 学習不要・最短なので先に
    bash experiments/exp_026/run_indiv.sh all
    bash experiments/exp_026/run_oracle.sh
    ;;
  *) echo "usage: $0 [indiv_condA|indiv_condB|indiv_zira|oracle|zeroshot|all]"; exit 1 ;;
esac

echo "==== exp_026 train_val.sh ($STAGE) 完了 ===="
