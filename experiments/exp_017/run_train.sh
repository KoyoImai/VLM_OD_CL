#!/usr/bin/env bash
# =============================================================================
# exp_017 学習＋ZCOCO評価 駆動スクリプト（射影アダプタ neck+text_feat_map のみ学習）
#   - 各ドメイン: dist_train（4GPU・20ep）→ best ckpt を ZCOCO(eval_base_coco.py) で評価
#   - 適応 mAP は学習中の domain valid best_coco_bbox_mAP をそのまま採用（別途評価不要）
#   - work_dir: experiments/exp_017/{domain}_neckTfm_work_dir
#   - ZCOCO   : experiments/exp_017/{domain}_neckTfm_zcoco
#   - ログ    : experiments/exp_017/train.log
#
# 使い方（リポジトリルートで実行）:
#   bash experiments/exp_017/run_train.sh aerial              # 1ドメイン検証
#   bash experiments/exp_017/run_train.sh underwater aerial   # 複数指定
#   bash experiments/exp_017/run_train.sh                     # 全6ドメイン（既定）
#
# 注: 実学習（高コスト）。承認後に実行する（design.md 承認ゲート）。seed=0 は config 側で固定。
# =============================================================================
set -uo pipefail

REPO=/workspace/kouyou/mmdetection
cd "$REPO"

GPUS=4
PORT=${PORT:-29517}
FAILED=()
DEFAULT_DOMAINS=(underwater aerial microscopic videogames documents electromagnetic)
if [ "$#" -gt 0 ]; then DOMAINS=("$@"); else DOMAINS=("${DEFAULT_DOMAINS[@]}"); fi

EXP_DIR=experiments/exp_017
LOG="$EXP_DIR/train.log"
ZCOCO_CFG=configs/mm_grounding_dino/eval_base_coco.py

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] exp_017 学習開始: ${DOMAINS[*]}" | tee -a "$LOG"
for d in "${DOMAINS[@]}"; do
  cfg="$EXP_DIR/configs/neck_tfm_only_${d}.py"
  wd="$EXP_DIR/${d}_neckTfm_work_dir"
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
  echo "[$(ts)] [${d}] ZCOCO 評価: $best -> ${d}_neckTfm_zcoco" | tee -a "$LOG"
  PORT=$PORT bash tools/dist_test.sh "$ZCOCO_CFG" "$best" "$GPUS" \
    --work-dir "$EXP_DIR/${d}_neckTfm_zcoco" 2>&1 | tee -a "$LOG"
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "[$(ts)] [${d}] FAIL (zcoco)" | tee -a "$LOG"; FAILED+=("${d} zcoco")
  fi
done

echo "[$(ts)] exp_017 完了" | tee -a "$LOG"
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "=== 失敗 ${#FAILED[@]} 件 ===" | tee -a "$LOG"
  printf '  - %s\n' "${FAILED[@]}" | tee -a "$LOG"
  exit 1
fi
echo "=== 全ドメイン 正常終了 ===" | tee -a "$LOG"
