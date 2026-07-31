#!/bin/bash
# =============================================================================
# exp_030_b train_val.sh — コンテナ内で走る学習評価ロジック。
#   note05 の分割に対応: exp_030_b = 蒸留D（融合後）+ 蒸留E（全部）
#   条件A（全モジュール）・コサイン類似度・リプレイあり。各条件は3ドメイン逐次。
#   段階（$1 / 既定 all）: D|E | all
# =============================================================================
set -euo pipefail
cd /workspace/kouyou/mmdetection

export HF_HOME=/tmp/hf
export TRANSFORMERS_CACHE=/tmp/hf/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p /tmp/hf/hub

export PORT="${PORT:-29710}"
export THETA0=/workspace/kouyou/ckpt/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth

run () { bash experiments/exp_028/run_sequential_kd.sh exp_030 condA cos "$1"; }

STAGE="${1:-all}"
case "$STAGE" in
  D) run D ;;
  E) run E ;;
  all) for k in D E; do run "$k"; done ;;
  *) echo "usage: $0 [D|E|all]"; exit 1 ;;
esac

echo "==== exp_030_b train_val ($STAGE) 完了 ===="
