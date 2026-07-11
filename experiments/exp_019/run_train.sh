#!/usr/bin/env bash
# =============================================================================
# exp_019 学習＋ZCOCO評価 駆動スクリプト（二重乖離）
#   条件B: downstream (下流のみ学習)
#   条件A: extfusion  (抽出・融合部のみ学習)
#   対象ドメイン: underwater / videogames / electromagnetic（3）
#   実行順序（ユーザー指定）: ドメインごとに downstream → extfusion
#   - 各: dist_train(4GPU・20ep) → best ckpt を ZCOCO(eval_base_coco.py) で評価
#   - work_dir: experiments/exp_019/{domain}_{downstream|extfusion}_work_dir
#   - ZCOCO   : experiments/exp_019/{domain}_{downstream|extfusion}_zcoco
#   - ログ    : experiments/exp_019/train.log
# 使い方（リポジトリルートで実行）:
#   bash experiments/exp_019/run_train.sh                        # 全6本
#   bash experiments/exp_019/run_train.sh underwater             # ドメイン指定（2条件とも）
# 注: 実学習（高コスト）。design.md 承認後に実行。seed=0 は config 側で固定。
# =============================================================================
set -uo pipefail

REPO=/workspace/kouyou/mmdetection
cd "$REPO"

GPUS=4
PORT=${PORT:-29519}
FAILED=()

DEFAULT_DOMAINS=(underwater videogames electromagnetic)
if [ "$#" -ge 1 ]; then DOMAINS=("$@"); else DOMAINS=("${DEFAULT_DOMAINS[@]}"); fi
CONDS=(downstream extfusion)   # ドメイン内の実行順（ユーザー指定: 下流 → 抽出・融合部）

EXP_DIR=experiments/exp_019
LOG="$EXP_DIR/train.log"
ZCOCO_CFG=configs/mm_grounding_dino/eval_base_coco.py

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] exp_019 学習開始: ドメイン=${DOMAINS[*]} / 条件順=${CONDS[*]}" | tee -a "$LOG"
for d in "${DOMAINS[@]}"; do
  for c in "${CONDS[@]}"; do
    cfg="$EXP_DIR/configs/${c}_${d}.py"
    wd="$EXP_DIR/${d}_${c}_work_dir"
    if [ ! -f "$cfg" ]; then
      echo "[$(ts)] [${d}/${c}] SKIP: config 無し ($cfg)" | tee -a "$LOG"; FAILED+=("${d}/${c} config"); continue
    fi

    # --- 学習 ---
    echo "[$(ts)] [${d}/${c}] train 開始 -> $wd" | tee -a "$LOG"
    PORT=$PORT bash tools/dist_train.sh "$cfg" "$GPUS" --work-dir "$wd" 2>&1 | tee -a "$LOG"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
      echo "[$(ts)] [${d}/${c}] FAIL (train)" | tee -a "$LOG"; FAILED+=("${d}/${c} train"); continue
    fi

    # --- best ckpt を ZCOCO 評価 ---
    best=$(ls -t "$wd"/best_coco_bbox_mAP_epoch_*.pth 2>/dev/null | head -1)
    if [ -z "$best" ]; then
      echo "[$(ts)] [${d}/${c}] FAIL: best ckpt 無し" | tee -a "$LOG"; FAILED+=("${d}/${c} best_ckpt"); continue
    fi
    echo "[$(ts)] [${d}/${c}] ZCOCO 評価: $best -> ${d}_${c}_zcoco" | tee -a "$LOG"
    PORT=$PORT bash tools/dist_test.sh "$ZCOCO_CFG" "$best" "$GPUS" \
      --work-dir "$EXP_DIR/${d}_${c}_zcoco" 2>&1 | tee -a "$LOG"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
      echo "[$(ts)] [${d}/${c}] FAIL (zcoco)" | tee -a "$LOG"; FAILED+=("${d}/${c} zcoco")
    fi
  done
done

echo "[$(ts)] exp_019 完了" | tee -a "$LOG"
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "=== 失敗 ${#FAILED[@]} 件 ===" | tee -a "$LOG"
  printf '  - %s\n' "${FAILED[@]}" | tee -a "$LOG"
  exit 1
fi
echo "=== 全ドメイン×条件 正常終了 ===" | tee -a "$LOG"
