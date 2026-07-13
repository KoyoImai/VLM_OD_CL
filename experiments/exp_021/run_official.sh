#!/usr/bin/env bash
# =============================================================================
# exp_021 学習＋ZCOCO評価 駆動スクリプト（DitHub 再現・公式準拠版）
#   3000 iter (warmup1500+spec1500) / batch 2 / lr 1e-3 / 1 GPU 学習 →
#   iter_3000.pth を dithub_eval_coco_{domain}.py で 4 GPU ZCOCO 評価
#   - work_dir: experiments/exp_021/{domain}_dithub_official_work_dir
#   - ZCOCO   : experiments/exp_021/{domain}_dithub_official_zcoco
#   - ログ    : experiments/exp_021/train.log
# 使い方（リポジトリルートで実行）:
#   bash experiments/exp_021/run_official.sh              # 全6ドメイン
#   bash experiments/exp_021/run_official.sh videogames   # ドメイン指定
# =============================================================================
set -uo pipefail

REPO=/workspace/kouyou/mmdetection
cd "$REPO"

PORT=${PORT:-29531}
FAILED=()

DEFAULT_DOMAINS=(underwater aerial videogames microscopic documents electromagnetic)
if [ "$#" -ge 1 ]; then DOMAINS=("$@"); else DOMAINS=("${DEFAULT_DOMAINS[@]}"); fi

EXP_DIR=experiments/exp_021
LOG="$EXP_DIR/train.log"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] exp_021 公式ハイパラ版 開始: ドメイン=${DOMAINS[*]}" | tee -a "$LOG"
for d in "${DOMAINS[@]}"; do
  cfg="$EXP_DIR/configs/dithub_official_${d}.py"
  zcfg="$EXP_DIR/configs/dithub_eval_coco_${d}.py"
  wd="$EXP_DIR/${d}_dithub_official_work_dir"
  if [ ! -f "$cfg" ] || [ ! -f "$zcfg" ]; then
    echo "[$(ts)] [${d}/official] SKIP: config 無し" | tee -a "$LOG"; FAILED+=("${d} config"); continue
  fi

  echo "[$(ts)] [${d}/official] train 開始 -> $wd" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=0 python tools/train.py "$cfg" --work-dir "$wd" >> "$LOG" 2>&1
  if [ $? -ne 0 ]; then
    echo "[$(ts)] [${d}/official] FAIL (train)" | tee -a "$LOG"; FAILED+=("${d} train"); continue
  fi

  ckpt="$wd/iter_3000.pth"
  if [ ! -f "$ckpt" ]; then
    echo "[$(ts)] [${d}/official] FAIL: iter_3000.pth 無し" | tee -a "$LOG"; FAILED+=("${d} ckpt"); continue
  fi
  echo "[$(ts)] [${d}/official] ZCOCO 評価 -> ${d}_dithub_official_zcoco" | tee -a "$LOG"
  PORT=$PORT bash tools/dist_test.sh "$zcfg" "$ckpt" 4 \
    --work-dir "$EXP_DIR/${d}_dithub_official_zcoco" >> "$LOG" 2>&1
  if [ $? -ne 0 ]; then
    echo "[$(ts)] [${d}/official] FAIL (zcoco)" | tee -a "$LOG"; FAILED+=("${d} zcoco")
  fi
done

echo "[$(ts)] exp_021 公式ハイパラ版 完了" | tee -a "$LOG"
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "=== 失敗 ${#FAILED[@]} 件 ===" | tee -a "$LOG"
  printf '  - %s\n' "${FAILED[@]}" | tee -a "$LOG"
  exit 1
fi
echo "=== 公式ハイパラ版 全ドメイン 正常終了 ===" | tee -a "$LOG"
