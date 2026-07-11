#!/usr/bin/env bash
# =============================================================================
# exp_020 学習＋ZCOCO評価 駆動スクリプト（ZiRa 再現・単発ドメイン）
#   対象ドメイン: underwater / aerial / videogames / microscopic /
#                 documents / electromagnetic（6。real_world は除外）
#   - 各: dist_train(4GPU・20ep) → best ckpt を ZCOCO(zira_eval_coco.py) で評価
#     (ZiRa ckpt は RDB キーを含むため、評価もモデル type を ZiRa 版に揃える)
#   - work_dir: experiments/exp_020/{domain}_zira_work_dir
#   - ZCOCO   : experiments/exp_020/{domain}_zira_zcoco
#   - ログ    : experiments/exp_020/train.log
# 使い方（リポジトリルートで実行）:
#   bash experiments/exp_020/run_train.sh                # 全6本
#   bash experiments/exp_020/run_train.sh underwater     # ドメイン指定
# 注: 実学習（高コスト）。design.md 承認後に実行。seed=0 は config 側で固定。
# =============================================================================
set -uo pipefail

REPO=/workspace/kouyou/mmdetection
cd "$REPO"

GPUS=4
PORT=${PORT:-29520}
FAILED=()

DEFAULT_DOMAINS=(underwater aerial videogames microscopic documents electromagnetic)
if [ "$#" -ge 1 ]; then DOMAINS=("$@"); else DOMAINS=("${DEFAULT_DOMAINS[@]}"); fi

EXP_DIR=experiments/exp_020
LOG="$EXP_DIR/train.log"
ZCOCO_CFG="$EXP_DIR/configs/zira_eval_coco.py"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] exp_020 学習開始: ドメイン=${DOMAINS[*]} (ZiRa)" | tee -a "$LOG"
for d in "${DOMAINS[@]}"; do
  cfg="$EXP_DIR/configs/zira_${d}.py"
  wd="$EXP_DIR/${d}_zira_work_dir"
  if [ ! -f "$cfg" ]; then
    echo "[$(ts)] [${d}] SKIP: config 無し ($cfg)" | tee -a "$LOG"; FAILED+=("${d} config"); continue
  fi

  # --- 学習 ---
  echo "[$(ts)] [${d}] train 開始 -> $wd" | tee -a "$LOG"
  PORT=$PORT bash tools/dist_train.sh "$cfg" "$GPUS" --work-dir "$wd" 2>&1 | tee -a "$LOG"
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "[$(ts)] [${d}] FAIL (train)" | tee -a "$LOG"; FAILED+=("${d} train"); continue
  fi

  # --- best ckpt を ZCOCO 評価 ---
  best=$(ls -t "$wd"/best_coco_bbox_mAP_epoch_*.pth 2>/dev/null | head -1)
  if [ -z "$best" ]; then
    echo "[$(ts)] [${d}] FAIL: best ckpt 無し" | tee -a "$LOG"; FAILED+=("${d} best_ckpt"); continue
  fi
  echo "[$(ts)] [${d}] ZCOCO 評価: $best -> ${d}_zira_zcoco" | tee -a "$LOG"
  PORT=$PORT bash tools/dist_test.sh "$ZCOCO_CFG" "$best" "$GPUS" \
    --work-dir "$EXP_DIR/${d}_zira_zcoco" 2>&1 | tee -a "$LOG"
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "[$(ts)] [${d}] FAIL (zcoco)" | tee -a "$LOG"; FAILED+=("${d} zcoco")
  fi
done

echo "[$(ts)] exp_020 完了" | tee -a "$LOG"
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "=== 失敗 ${#FAILED[@]} 件 ===" | tee -a "$LOG"
  printf '  - %s\n' "${FAILED[@]}" | tee -a "$LOG"
  exit 1
fi
echo "=== 全ドメイン 正常終了 ===" | tee -a "$LOG"
