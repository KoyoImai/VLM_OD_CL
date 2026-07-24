#!/usr/bin/env bash
# =============================================================================
# exp_023 DitHub 逐次ドライバ（replay-free / +replay 共通。第1引数で切替）
#   DitHub はクラス別 LoRA を「育つ単一ライブラリ」で保持し、過去ドメインも
#   学習済み A を当てて評価する（案Y・公式忠実）。各ドメイン t で:
#     1. 学習（load_from = 前ライブラリ lib。t=1 は config の θ0）。
#        現ドメイン∩過去のクラス（t=3=person,car）は specialization 切替時に
#        式3 fetch+merge（DitHubSeqPhaseHook, config 側で配線済み）。
#     2. ライブラリ更新: merge_dithub.py（式4 B融合 + 過去A和集合）で成長ライブラリ
#        lib_after_<dom>.pth を作る。t=1 は最初のタスクなので学習 ckpt をそのまま採用。
#     3. 評価: 全クラス(152) dithub_eval_* モデルでライブラリ ckpt を読み、
#        現在 + 全過去ドメイン + ZCOCO を評価（過去は学習済み A が当たる）。
#   ※ 実行は design.md 承認後（行動原理3）。既存プログラムは無変更。
# 使い方: bash experiments/exp_023/run_sequential_dithub.sh [replayfree|replay]
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."   # experiments/exp_023/ -> リポジトリルート

VARIANT="${1:-replayfree}"   # replayfree | replay
case "$VARIANT" in
  replayfree|replay) ;;
  *) echo "usage: $0 [replayfree|replay]"; exit 1;;
esac

GPUS=4
CFG=experiments/exp_023/configs
OUT=experiments/exp_023
MERGE=experiments/exp_021/merge_dithub.py
LIBDIR="$OUT/dithub_${VARIANT}_library"
mkdir -p "$LIBDIR"

ORDER=(underwater electromagnetic videogames)
PAST=()
lib=""   # 前ドメインの成長ライブラリ ckpt（空なら config の θ0 を使う）

for dom in "${ORDER[@]}"; do
  WD="$OUT/dithub_${VARIANT}_${dom}_work_dir"
  echo "==================== [DitHub/$VARIANT] $dom 学習 ===================="
  if [ -z "$lib" ]; then
    bash tools/dist_train.sh "$CFG/dithub_${VARIANT}_${dom}.py" "$GPUS" --work-dir "$WD"
  else
    bash tools/dist_train.sh "$CFG/dithub_${VARIANT}_${dom}.py" "$GPUS" --work-dir "$WD" \
      --cfg-options load_from="$lib"
  fi
  ckpt="$(cat "$WD/last_checkpoint")"
  echo "[$dom] last ckpt: $ckpt"

  # --- ライブラリ更新（式4 B融合 + 過去A和集合）---
  newlib="$LIBDIR/lib_after_${dom}.pth"
  if [ -z "$lib" ]; then
    cp "$ckpt" "$newlib"                 # t=1: 最初のタスク（merge 不要）
    echo "[$dom] library 初期化: $newlib"
  else
    python "$MERGE" "$lib" "$ckpt" "$newlib"
    echo "[$dom] library 更新: $newlib（式4 + A和集合）"
  fi
  lib="$newlib"

  # --- 評価（現在 + 全過去ドメイン、全クラス DitHub モデルでライブラリ ckpt を読む）---
  for evd in "$dom" "${PAST[@]}"; do
    echo "-------------------- [$dom] 評価 on $evd --------------------"
    bash tools/dist_test.sh "$CFG/dithub_eval_${evd}.py" "$lib" "$GPUS" \
      --work-dir "$OUT/dithub_${VARIANT}_eval_after_${dom}/on_${evd}"
  done

  # --- ZCOCO ---
  echo "-------------------- [$dom] ZCOCO --------------------"
  bash tools/dist_test.sh "$CFG/dithub_eval_zcoco.py" "$lib" "$GPUS" \
    --work-dir "$OUT/dithub_${VARIANT}_eval_after_${dom}/zcoco"

  PAST+=("$dom")
done

echo "==================== DitHub/$VARIANT 逐次 完了（library: $lib）===================="
