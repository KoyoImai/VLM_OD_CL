#!/usr/bin/env bash
# =============================================================================
# exp_015 評価駆動スクリプト（exp_014 と同一形式）
#   - 対象: hybrids/{domain}_{whole|decoder|cls|reg|qsel}_theta0.pth（make_hybrid_downstream.py）
#   - ZCOCO: configs/mm_grounding_dino/eval_base_coco.py（(800,1333) keep_ratio・4GPU）
#   - 適応 : 各ドメイン FT config（自ドメイン valid・(800,1333)・4GPU）
#   - 出力 : experiments/exp_015/{domain}_{group}_theta0_{coco|adapt}/
#   - ログ : experiments/exp_015/{zcoco|adapt}_eval.log
#
# 使い方（リポジトリルートで実行）:
#   bash experiments/exp_015/run_eval.sh zcoco   # ZCOCO のみ
#   bash experiments/exp_015/run_eval.sh adapt   # 適応のみ
#   bash experiments/exp_015/run_eval.sh all      # 両方（既定）
#
# 注: 学習は一切行わない（forward のみ）。承認後に実行する（design.md 承認ゲート）。
#     GROUPS は bash 特殊変数のため使わない（MOD_GROUPS を使用）。
# =============================================================================
# set -e は使わない。1本の eval 失敗で残り（最大60本）を巻き込まないため継続実行＋失敗追跡。
set -uo pipefail

REPO=/workspace/kouyou/mmdetection
cd "$REPO"

FAILED=()      # "domain/group [zcoco|adapt]" の失敗を記録

GPUS=4
PORT=${PORT:-29515}                      # exp_013/014 と被らない既定ポート
MODE="${1:-all}"                         # zcoco | adapt | all
DOMAINS=(underwater aerial microscopic videogames documents electromagnetic)
MOD_GROUPS=(whole decoder cls reg qsel)

EXP_DIR=experiments/exp_015
HYB_DIR="$EXP_DIR/hybrids"
ZCOCO_CFG=configs/mm_grounding_dino/eval_base_coco.py
ZCOCO_LOG="$EXP_DIR/zcoco_eval.log"
ADAPT_LOG="$EXP_DIR/adapt_eval.log"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

run_zcoco() {
  echo "[$(ts)] exp_015 hybrid ZCOCO 評価 開始" | tee -a "$ZCOCO_LOG"
  for d in "${DOMAINS[@]}"; do
    for g in "${MOD_GROUPS[@]}"; do
      local ckpt="$HYB_DIR/${d}_${g}_theta0.pth"
      local wd="$EXP_DIR/${d}_${g}_theta0_coco"
      if [ ! -f "$ckpt" ]; then
        echo "[$(ts)] [${d}/${g}] SKIP: hybrid 未生成 ($ckpt)" | tee -a "$ZCOCO_LOG"
        continue
      fi
      echo "[$(ts)] [${d}/${g}] eval 開始: $ckpt -> $wd" | tee -a "$ZCOCO_LOG"
      PORT=$PORT bash tools/dist_test.sh "$ZCOCO_CFG" "$ckpt" "$GPUS" \
        --work-dir "$wd" 2>&1 | tee -a "$ZCOCO_LOG"
      if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "[$(ts)] [${d}/${g}] FAIL (zcoco)" | tee -a "$ZCOCO_LOG"
        FAILED+=("${d}/${g} zcoco")
      fi
    done
  done
  echo "[$(ts)] exp_015 hybrid ZCOCO 評価 完了" | tee -a "$ZCOCO_LOG"
}

run_adapt() {
  echo "[$(ts)] exp_015 hybrid 適応評価 開始" | tee -a "$ADAPT_LOG"
  for d in "${DOMAINS[@]}"; do
    local cfg="configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_${d}.py"
    for g in "${MOD_GROUPS[@]}"; do
      local ckpt="$HYB_DIR/${d}_${g}_theta0.pth"
      local wd="$EXP_DIR/${d}_${g}_theta0_adapt"
      if [ ! -f "$ckpt" ]; then
        echo "[$(ts)] [${d}/${g}] SKIP: hybrid 未生成 ($ckpt)" | tee -a "$ADAPT_LOG"
        continue
      fi
      echo "[$(ts)] [${d}/${g}] adapt eval 開始 -> $wd" | tee -a "$ADAPT_LOG"
      PORT=$PORT bash tools/dist_test.sh "$cfg" "$ckpt" "$GPUS" \
        --work-dir "$wd" 2>&1 | tee -a "$ADAPT_LOG"
      if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "[$(ts)] [${d}/${g}] FAIL (adapt)" | tee -a "$ADAPT_LOG"
        FAILED+=("${d}/${g} adapt")
      fi
    done
  done
  echo "[$(ts)] exp_015 hybrid 適応評価 完了" | tee -a "$ADAPT_LOG"
}

case "$MODE" in
  zcoco) run_zcoco ;;
  adapt) run_adapt ;;
  all)   run_zcoco; run_adapt ;;
  *) echo "usage: bash $0 {zcoco|adapt|all}"; exit 1 ;;
esac

if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "=== 失敗 ${#FAILED[@]} 件 ==="
  printf '  - %s\n' "${FAILED[@]}"
  exit 1
fi
echo "=== 全評価 正常終了 ==="
