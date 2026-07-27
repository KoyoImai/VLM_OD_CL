#!/bin/bash
# =============================================================================
# exp_029_a train_val.sh — コンテナ内で走る学習評価ロジック。
#   note05 の分割に対応: exp_029_a = 蒸留A（画像）+ 蒸留B（テキスト）
#   条件B（特徴抽出+融合）・L2ノルム・リプレイあり。各蒸留条件は3ドメイン逐次（約43h）。
#   段階（$1 / 既定 all）: A|B | all
# =============================================================================
set -euo pipefail
cd /workspace/kouyou/mmdetection

export HF_HOME=/tmp/hf
export TRANSFORMERS_CACHE=/tmp/hf/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p /tmp/hf/hub

export PORT="${PORT:-29680}"
export THETA0=/workspace/kouyou/ckpt/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth

run () { bash experiments/exp_028/run_sequential_kd.sh exp_029 condB l2 "$1"; }

STAGE="${1:-all}"
case "$STAGE" in
  A) run A ;;
  B) run B ;;
  all) for k in A B; do run "$k"; done ;;
  *) echo "usage: $0 [A|B|all]"; exit 1 ;;
esac

echo "==== exp_029_a train_val ($STAGE) 完了 ===="
