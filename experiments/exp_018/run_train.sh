#!/usr/bin/env bash
# =============================================================================
# exp_018 学習＋ZCOCO評価 駆動スクリプト（部分学習）
#   条件A: freeze_enc  (Feature Enhancer=encoder "だけ" 凍結、他は全学習)
#   条件B: bert_tfm    (BERT=language_model + text_feat_map のみ学習)
#   対象ドメイン: underwater / videogames / electromagnetic（3）
#   - 各: dist_train(4GPU・20ep) → best ckpt を ZCOCO(eval_base_coco.py) で評価
#   - 適応 mAP は学習中の domain valid best をそのまま採用
#   - work_dir: experiments/exp_018/{domain}_{freezeEnc|bertTfm}_work_dir
#   - ZCOCO   : experiments/exp_018/{domain}_{freezeEnc|bertTfm}_zcoco
#   - ログ    : experiments/exp_018/train.log
#
# 使い方（リポジトリルートで実行）:
#   bash experiments/exp_018/run_train.sh                         # 全6本（3ドメイン×2条件）
#   bash experiments/exp_018/run_train.sh freeze_enc underwater   # 条件・ドメイン指定（検証用）
#     引数1 = 条件 {freeze_enc|bert_tfm|all}、引数2以降 = ドメイン（省略時は3ドメイン既定）
#
# 注: 実学習（高コスト）。承認後に実行する（design.md 承認ゲート）。seed=0 は config 側で固定。
# =============================================================================
set -uo pipefail

REPO=/workspace/kouyou/mmdetection
cd "$REPO"

GPUS=4
PORT=${PORT:-29518}
FAILED=()

DEFAULT_DOMAINS=(underwater videogames electromagnetic)
COND="${1:-all}"
if [ "$#" -gt 1 ]; then DOMAINS=("${@:2}"); else DOMAINS=("${DEFAULT_DOMAINS[@]}"); fi
case "$COND" in
  freeze_enc) CONDS=(freeze_enc) ;;
  bert_tfm)   CONDS=(bert_tfm) ;;
  all)        CONDS=(freeze_enc bert_tfm) ;;
  *) echo "usage: bash $0 {freeze_enc|bert_tfm|all} [domains...]"; exit 1 ;;
esac

# 条件名 -> work_dir タグ
tag_of() { case "$1" in freeze_enc) echo freezeEnc ;; bert_tfm) echo bertTfm ;; esac; }

EXP_DIR=experiments/exp_018
LOG="$EXP_DIR/train.log"
ZCOCO_CFG=configs/mm_grounding_dino/eval_base_coco.py

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] exp_018 学習開始: 条件=${CONDS[*]} / ドメイン=${DOMAINS[*]}" | tee -a "$LOG"
for c in "${CONDS[@]}"; do
  tag=$(tag_of "$c")
  for d in "${DOMAINS[@]}"; do
    cfg="$EXP_DIR/configs/${c}_${d}.py"
    wd="$EXP_DIR/${d}_${tag}_work_dir"
    if [ ! -f "$cfg" ]; then
      echo "[$(ts)] [${c}/${d}] SKIP: config 無し ($cfg)" | tee -a "$LOG"; FAILED+=("${c}/${d} config"); continue
    fi

    # --- 学習 ---
    echo "[$(ts)] [${c}/${d}] train 開始 -> $wd" | tee -a "$LOG"
    PORT=$PORT bash tools/dist_train.sh "$cfg" "$GPUS" --work-dir "$wd" 2>&1 | tee -a "$LOG"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
      echo "[$(ts)] [${c}/${d}] FAIL (train)" | tee -a "$LOG"; FAILED+=("${c}/${d} train"); continue
    fi

    # --- best ckpt を ZCOCO 評価 ---
    best=$(ls -t "$wd"/best_coco_bbox_mAP_epoch_*.pth 2>/dev/null | head -1)
    if [ -z "$best" ]; then
      echo "[$(ts)] [${c}/${d}] FAIL: best ckpt 無し" | tee -a "$LOG"; FAILED+=("${c}/${d} best_ckpt"); continue
    fi
    echo "[$(ts)] [${c}/${d}] ZCOCO 評価: $best -> ${d}_${tag}_zcoco" | tee -a "$LOG"
    PORT=$PORT bash tools/dist_test.sh "$ZCOCO_CFG" "$best" "$GPUS" \
      --work-dir "$EXP_DIR/${d}_${tag}_zcoco" 2>&1 | tee -a "$LOG"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
      echo "[$(ts)] [${c}/${d}] FAIL (zcoco)" | tee -a "$LOG"; FAILED+=("${c}/${d} zcoco")
    fi
  done
done

echo "[$(ts)] exp_018 完了" | tee -a "$LOG"
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "=== 失敗 ${#FAILED[@]} 件 ===" | tee -a "$LOG"
  printf '  - %s\n' "${FAILED[@]}" | tee -a "$LOG"
  exit 1
fi
echo "=== 全条件×ドメイン 正常終了 ===" | tee -a "$LOG"
