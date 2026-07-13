#!/usr/bin/env bash
# =============================================================================
# exp_020 追加実験: ZiRa 公式ハイパラ版の学習＋ZCOCO評価
#   2000 iter / batch 2 / lr 1e-3 / 1 GPU 学習 → 4 GPU で ZCOCO 評価
#   underwater は対照 run (2026-07-11) で実施済みのため既定リストから除外
#   - work_dir: experiments/exp_020/{domain}_zira_official_work_dir
#   - ZCOCO   : experiments/exp_020/{domain}_zira_official_zcoco
#   - ログ    : experiments/exp_020/train.log
# 使い方（リポジトリルートで実行）:
#   bash experiments/exp_020/run_official.sh              # 既定 5 ドメイン
#   bash experiments/exp_020/run_official.sh videogames   # ドメイン指定
# =============================================================================
set -uo pipefail

REPO=/workspace/kouyou/mmdetection
cd "$REPO"

PORT=${PORT:-29522}
FAILED=()

DEFAULT_DOMAINS=(aerial videogames microscopic documents electromagnetic)
if [ "$#" -ge 1 ]; then DOMAINS=("$@"); else DOMAINS=("${DEFAULT_DOMAINS[@]}"); fi

EXP_DIR=experiments/exp_020
LOG="$EXP_DIR/train.log"
ZCOCO_CFG="$EXP_DIR/configs/zira_eval_coco.py"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] exp_020 公式ハイパラ版 開始: ドメイン=${DOMAINS[*]}" | tee -a "$LOG"
for d in "${DOMAINS[@]}"; do
  cfg="$EXP_DIR/configs/zira_official_${d}.py"
  wd="$EXP_DIR/${d}_zira_official_work_dir"
  if [ ! -f "$cfg" ]; then
    echo "[$(ts)] [${d}/official] SKIP: config 無し ($cfg)" | tee -a "$LOG"; FAILED+=("${d} config"); continue
  fi

  echo "[$(ts)] [${d}/official] train 開始 -> $wd" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=0 python tools/train.py "$cfg" --work-dir "$wd" >> "$LOG" 2>&1
  if [ $? -ne 0 ]; then
    echo "[$(ts)] [${d}/official] FAIL (train)" | tee -a "$LOG"; FAILED+=("${d} train"); continue
  fi

  ckpt="$wd/iter_2000.pth"
  if [ ! -f "$ckpt" ]; then
    echo "[$(ts)] [${d}/official] FAIL: iter_2000.pth 無し" | tee -a "$LOG"; FAILED+=("${d} ckpt"); continue
  fi
  echo "[$(ts)] [${d}/official] ZCOCO 評価: $ckpt -> ${d}_zira_official_zcoco" | tee -a "$LOG"
  PORT=$PORT bash tools/dist_test.sh "$ZCOCO_CFG" "$ckpt" 4 \
    --work-dir "$EXP_DIR/${d}_zira_official_zcoco" >> "$LOG" 2>&1
  if [ $? -ne 0 ]; then
    echo "[$(ts)] [${d}/official] FAIL (zcoco)" | tee -a "$LOG"; FAILED+=("${d} zcoco")
  fi
done

echo "[$(ts)] exp_020 公式ハイパラ版 完了" | tee -a "$LOG"
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "=== 失敗 ${#FAILED[@]} 件 ===" | tee -a "$LOG"
  printf '  - %s\n' "${FAILED[@]}" | tee -a "$LOG"
  exit 1
fi
echo "=== 公式ハイパラ版 全ドメイン 正常終了 ===" | tee -a "$LOG"
