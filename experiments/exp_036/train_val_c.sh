#!/bin/bash
# =============================================================================
# exp_036_c train_val.sh — 条件A + L2 蒸留 + リプレイ / **λ = 3.0 固定**
#   note07 の分割に対応: 蒸留A（画像）+ 蒸留B（テキスト）
#   段階（$1 / 既定 all）: A|B | all
# =============================================================================
set -euo pipefail
cd /workspace/kouyou/mmdetection

export HF_HOME=/tmp/hf
export TRANSFORMERS_CACHE=/tmp/hf/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p /tmp/hf/hub

export PORT="${PORT:-29860}"
export THETA0=/workspace/kouyou/ckpt/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth

# ★ λ はここで明示する。ドライバ run_sequential_kd.sh は
#   lam="${LAMBDA:-1.0}" を --cfg-options model.kd.loss_weight に渡すため、
#   LAMBDA を設定しないと config の loss_weight=3.0 が 1.0 で上書きされる。
#   （exp_033 はこれが原因で全12条件が実効 λ=1.0 で学習されていた。2026-08-06 検出）
export LAMBDA=3.0

# 起動時に config と LAMBDA の一致を確認する（食い違ったら学習を始めない）
python - <<'PY'
import os
from mmengine.config import Config
w = Config.fromfile(
    'experiments/exp_036/configs/kdA_condA_l2w30_underwater.py').model.kd.loss_weight
lam = float(os.environ['LAMBDA'])
assert w == lam, f'config の loss_weight={w} と LAMBDA={lam} が食い違っています'
print(f'[exp_036_c] 実効 λ = {lam}（config と一致）')
PY

run () { bash experiments/exp_028/run_sequential_kd.sh exp_036 condA l2w30 "$1"; }

STAGE="${1:-all}"
case "$STAGE" in
  A) run A ;;
  B) run B ;;
  all) for k in A B; do run "$k"; done ;;
  *) echo "usage: $0 [A|B|all]"; exit 1 ;;
esac

echo "==== exp_036_c train_val ($STAGE) 完了 ===="
