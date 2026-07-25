#!/usr/bin/env bash
# =============================================================================
# exp_023 再開ドライバ（2026-07-25）
#
# 停止点:
#   手法5（full finetuning + リプレイ）の electromagnetic 学習（epoch_20 まで完走）と
#   自ドメイン評価（mAP 0.4860）が終わった直後、2026-07-24 23:45:19 に dist_test の
#   プロセス終了処理で `double free or corruption (!prev)` → SIGABRT（exitcode -6）が発生。
#   結果はログに完全に書き出された後の異常終了だが、`set -e` により
#   run_remaining_fullft_condA.sh 全体が停止した。
#
# 本スクリプトが実行する残り作業:
#   5-a  electromagnetic 学習後の残り評価（on_underwater ／ ZCOCO）
#   5-b  videogames の学習（load_from = electromagnetic の last）＋評価4本
#   6    条件A + リプレイ（3ドメイン逐次。run_sequential_condA.sh をそのまま呼ぶ）
#
# 完了済みのためスキップするもの（再学習しない）:
#   手法1・2（ZiRa なし/あり）、手法3（DitHub なし）、
#   手法5 の underwater 学習＋評価3本、electromagnetic 学習＋自ドメイン評価。
#   手法4（DitHub + リプレイ）はデバッグ待ちのため本スクリプトでは扱わない。
#
# 実験条件（config・ckpt・評価プロトコル・seed）は既存のものを一切変更していない。
# 唯一の運用上の差分は run_eval() の扱い（下記コメント参照）。
#
# 使い方: bash experiments/exp_023/run_resume_fullft_from_em_eval.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."   # experiments/exp_023/ -> リポジトリルート

GPUS=4
D=experiments/exp_023
CFG=$D/configs
OUT=$D
ZCOCO_CFG=configs/mm_grounding_dino/eval_base_coco.py

# -----------------------------------------------------------------------------
# 評価の実行ラッパ
#   dist_test が異常終了した場合、work_dir の最新ランに評価結果（coco/bbox_mAP）が
#   書き出されているかを確認する。
#     - 書き出されている  -> 今回と同じ「終了処理でのクラッシュ」とみなし、警告を出して続行。
#       （評価結果そのものは完全なので、後続の長時間ジョブを巻き添えで落とさない）
#     - 書き出されていない -> 本当に評価が失敗しているので、停止する。
#   評価の設定・入力・出力先は元のドライバと同一。判定に使う値も変わらない。
# -----------------------------------------------------------------------------
run_eval () {
  local cfg="$1" ckpt="$2" wd="$3" tag="$4"
  echo "-------------------- $tag --------------------"
  if bash tools/dist_test.sh "$cfg" "$ckpt" "$GPUS" --work-dir "$wd"; then
    return 0
  fi
  local latest
  latest="$(ls -1dt "$wd"/*/ 2>/dev/null | head -1 || true)"
  if [ -n "$latest" ] && grep -qs "coco/bbox_mAP:" "$latest"/*.log; then
    echo "WARNING: dist_test が異常終了しましたが、評価結果は書き出されています。続行します。"
    echo "         $latest"
    grep -hoE "coco/bbox_mAP: [0-9.]+" "$latest"/*.log | tail -1
    return 0
  fi
  echo "ERROR: 評価に失敗し、結果も残っていません: $tag"
  return 1
}

echo "########## 5-a/6: full FT+リプレイ electromagnetic の残り評価 ##########"
EM_WD="$OUT/full_ft_replay_electromagnetic_work_dir"
EM_CKPT="$(cat "$EM_WD/last_checkpoint")"
[ -f "$EM_CKPT" ] || { echo "ERROR: electromagnetic の last ckpt が無い: $EM_CKPT"; exit 1; }
echo "[electromagnetic] last ckpt: $EM_CKPT"

run_eval "$CFG/eval_underwater.py" "$EM_CKPT" \
  "$OUT/eval_after_electromagnetic/on_underwater" "[electromagnetic] 評価 on underwater"
run_eval "$ZCOCO_CFG" "$EM_CKPT" \
  "$OUT/eval_after_electromagnetic/zcoco" "[electromagnetic] ZCOCO"

echo "########## 5-b/6: full FT+リプレイ videogames 学習＋評価 ##########"
VG_WD="$OUT/full_ft_replay_videogames_work_dir"
bash tools/dist_train.sh "$CFG/fullft_replay_videogames.py" "$GPUS" --work-dir "$VG_WD" \
  --cfg-options load_from="$EM_CKPT"
VG_CKPT="$(cat "$VG_WD/last_checkpoint")"
echo "[videogames] last ckpt: $VG_CKPT"

for evd in videogames underwater electromagnetic; do
  run_eval "$CFG/eval_${evd}.py" "$VG_CKPT" \
    "$OUT/eval_after_videogames/on_${evd}" "[videogames] 評価 on $evd"
done
run_eval "$ZCOCO_CFG" "$VG_CKPT" "$OUT/eval_after_videogames/zcoco" "[videogames] ZCOCO"

echo "==================== 手法5 full finetuning + リプレイ 完了 ===================="

echo "########## 6/6: 条件A + リプレイ ##########"
# run_sequential_condA.sh と同一の処理を、評価だけ run_eval 経由にして展開する。
#   （既存の run_sequential_condA.sh は無変更。config・work_dir 名・評価出力先・逐次順は同一。
#     終了処理のクラッシュで 40 時間級の逐次学習を巻き添えにしないための展開）
CONDA_ORDER=(underwater electromagnetic videogames)
CONDA_PAST=()
conda_prev=""   # 空なら config の load_from(θ0) を使う（t=1）

for dom in "${CONDA_ORDER[@]}"; do
  WD="$OUT/condA_replay_${dom}_work_dir"
  echo "==================== [条件A/replay] $dom 学習 ===================="
  if [ -z "$conda_prev" ]; then
    bash tools/dist_train.sh "$CFG/condA_replay_${dom}.py" "$GPUS" --work-dir "$WD"
  else
    bash tools/dist_train.sh "$CFG/condA_replay_${dom}.py" "$GPUS" --work-dir "$WD" \
      --cfg-options load_from="$conda_prev"
  fi
  last="$(cat "$WD/last_checkpoint")"
  echo "[条件A/replay][$dom] last ckpt: $last"

  for evd in "$dom" "${CONDA_PAST[@]:-}"; do
    [ -n "$evd" ] || continue
    run_eval "$CFG/eval_${evd}.py" "$last" \
      "$OUT/condA_replay_eval_after_${dom}/on_${evd}" "[条件A/replay][$dom] 評価 on $evd"
  done
  run_eval "$ZCOCO_CFG" "$last" \
    "$OUT/condA_replay_eval_after_${dom}/zcoco" "[条件A/replay][$dom] ZCOCO"

  conda_prev="$last"
  CONDA_PAST+=("$dom")
done

echo "==================== 条件A + リプレイ 逐次 完了 ===================="

echo "########## exp_023 再開分（手法5残り・手法6）完了 ##########"
