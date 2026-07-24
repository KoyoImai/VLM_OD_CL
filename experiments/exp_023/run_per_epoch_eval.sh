#!/usr/bin/env bash
# =============================================================================
# exp_023 方法B: 保存済み全 epoch checkpoint をオフラインで評価する。
#   各ドメイン t の epoch_N.pth を「現ドメイン + それまでの全過去ドメイン + ZCOCO」で評価し、
#   適応曲線（現ドメイン）と忘却曲線（過去ドメイン・ZCOCO）を epoch 方向に取得する。
#   逐次順・評価対象の組は逐次ドライバ（run_sequential_*）と一致させる。
#
#   ※ 学習ドライバの完了後（GPU が空いてから）に実行する。学習中は GPU 競合するため走らせない。
#   ※ 既存の eval config（eval_{domain}.py / eval_base_coco.py）をそのまま使用。既存プログラム無変更。
#
# 粒度（ユーザー方針 2026-07-23「現ドメインのみ密」）:
#   - 現ドメイン（適応）: 全 epoch を評価（CUR_EPOCHS, 既定 1..20）。
#   - 過去ドメイン＋ZCOCO（忘却）: 間引き評価（FORGET_EPOCHS, 既定 1 4 8 12 16 20）。
#
# 使い方:
#   bash experiments/exp_023/run_per_epoch_eval.sh fullft                       # 既定粒度
#   bash experiments/exp_023/run_per_epoch_eval.sh condA "$(seq 1 20)" "1 4 8 12 16 20"
#   PORT=29530 bash experiments/exp_023/run_per_epoch_eval.sh fullft
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."   # experiments/exp_023/ -> リポジトリルート

GPUS=4
export PORT="${PORT:-29520}"   # 学習(29500)と衝突しないポート
CFG=experiments/exp_023/configs
OUT=experiments/exp_023
ZCOCO_CFG=configs/mm_grounding_dino/eval_base_coco.py

METHOD="${1:-fullft}"
CUR_EPOCHS="${2:-$(seq 1 20)}"        # 現ドメイン（適応）: 密
FORGET_EPOCHS="${3:-1 4 8 12 16 20}"  # 過去ドメイン＋ZCOCO（忘却）: 粗
case "$METHOD" in
  fullft) WD_PREFIX="full_ft_replay"; TAG="fullft" ;;
  condA)  WD_PREFIX="condA_replay";   TAG="condA"  ;;
  *) echo "usage: $0 [fullft|condA] [\"current epochs\"] [\"forget epochs\"]"; exit 1 ;;
esac

_eval_one() {  # $1=eval config, $2=ckpt, $3=out subdir
  bash tools/dist_test.sh "$1" "$2" "$GPUS" --work-dir "$3"
}

ORDER=(underwater electromagnetic videogames)
PAST=()
for dom in "${ORDER[@]}"; do
  WD="$OUT/${WD_PREFIX}_${dom}_work_dir"

  # --- 現ドメイン（適応）: 全 epoch ---
  for ep in $CUR_EPOCHS; do
    CKPT="$WD/epoch_${ep}.pth"
    if [ ! -f "$CKPT" ]; then echo "SKIP (no ckpt): $CKPT"; continue; fi
    echo "==== [$TAG] $dom epoch$ep on $dom (適応) ===="
    _eval_one "$CFG/eval_${dom}.py" "$CKPT" \
      "$OUT/eval_per_epoch/${TAG}/${dom}/epoch_${ep}/on_${dom}"
  done

  # --- 過去ドメイン＋ZCOCO（忘却）: 間引き ---
  for ep in $FORGET_EPOCHS; do
    CKPT="$WD/epoch_${ep}.pth"
    if [ ! -f "$CKPT" ]; then echo "SKIP (no ckpt): $CKPT"; continue; fi
    for evd in "${PAST[@]}"; do
      echo "==== [$TAG] $dom epoch$ep on $evd (忘却) ===="
      _eval_one "$CFG/eval_${evd}.py" "$CKPT" \
        "$OUT/eval_per_epoch/${TAG}/${dom}/epoch_${ep}/on_${evd}"
    done
    echo "==== [$TAG] $dom epoch$ep on ZCOCO (忘却) ===="
    _eval_one "$ZCOCO_CFG" "$CKPT" \
      "$OUT/eval_per_epoch/${TAG}/${dom}/epoch_${ep}/zcoco"
  done

  PAST+=("$dom")
done

echo "==== 方法B per-epoch eval ($TAG) 完了 ===="
