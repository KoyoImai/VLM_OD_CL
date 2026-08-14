#!/usr/bin/env bash
# =============================================================================
# exp_040: 後半3ドメインの逐次学習ドライバ（design.md の承認後に実行する）
#
#   前半3ドメイン（underwater -> electromagnetic -> videogames）の t=3 終了時の
#   パラメータを初期値として、aerial -> microscopic -> documents を逐次学習する。
#
#   各ドメイン t（4->5->6）で:
#     1. 学習（20 epoch）。load_from は θ_{t-1}（t=4 は §2.1 の初期パラメータ）。
#        条件 kdE は teacher_ckpt にも θ_{t-1} を渡す。
#     2. 評価: 学習済み6ドメインのうち t までのものと ZCOCO。
#        評価 config は前半3ドメインが exp_023、後半3ドメインが exp_026 のものを使う。
#
#   途中で止まった場合は同じコマンドで再開できる（epoch_20.pth があるドメインはスキップ）。
#
# 使い方:
#   COND=replayfree bash experiments/exp_040/run_sequential.sh
#   COND=replay     bash experiments/exp_040/run_sequential.sh
#   COND=kdE        bash experiments/exp_040/run_sequential.sh
#   GPUS=4 が既定（クラスタ）。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

COND="${COND:?COND=replayfree|replay|kdE を指定}"
GPUS="${GPUS:-4}"

# 前半3タスク目（videogames）の学習結果 = 後半の初期パラメータ（design.md §2.1）
case "$COND" in
  replayfree) INIT=experiments/exp_024/fullft_replayfree_videogames_work_dir/epoch_20.pth ;;
  replay)     INIT=experiments/exp_027/fullft_replay_videogames_work_dir/epoch_20.pth ;;
  kdE)        INIT=experiments/exp_035/kdE_condA_l2w100_videogames_work_dir/epoch_20.pth ;;
  *) echo "COND は replayfree / replay / kdE"; exit 1 ;;
esac
[ -f "$INIT" ] || { echo "初期パラメータが無い: $INIT"; exit 1; }

EXP=experiments/exp_040
CFG="$EXP/configs"
EVALDIR="$EXP/eval_${COND}"
mkdir -p "$EVALDIR"

ORDER=(aerial microscopic documents)                 # 後半3ドメイン（t=4,5,6）
LEARNED=(underwater electromagnetic videogames)      # 前半3ドメイン（学習済み）

# 評価 config の在り処（前半は exp_023、後半は exp_026）
eval_cfg() {
  case "$1" in
    underwater|electromagnetic|videogames) echo "experiments/exp_023/configs/eval_$1.py" ;;
    aerial|microscopic|documents)          echo "experiments/exp_026/configs/eval_$1.py" ;;
  esac
}

echo "==================== exp_040 $COND ===================="
echo "GPUS=$GPUS / 初期パラメータ: $INIT / 出力: $EVALDIR"

prev="$INIT"

for i in "${!ORDER[@]}"; do
  t=$((4 + i))
  dom="${ORDER[$i]}"
  WD="$EXP/${COND}_${dom}_work_dir"
  ckpt="$WD/epoch_20.pth"

  echo "==================== [t=$t/6] $dom 学習 ===================="
  if [ -f "$ckpt" ]; then
    echo "[t=$t] $ckpt が既にあるため学習をスキップ"
  else
    OPTS=(--cfg-options "load_from=$prev")
    if [ "$COND" = "kdE" ]; then
      # 教師 = θ_{t-1}。**model. を付ける**（付けないとトップレベルのキーになり
      # model.teacher_ckpt は None のままで assertion に掛かる。exp_028 と同じ書式）。
      OPTS+=("model.teacher_ckpt=$prev")
    fi
    bash tools/dist_train.sh "$CFG/${COND}_${dom}.py" "$GPUS" \
      --work-dir "$WD" "${OPTS[@]}"
  fi
  prev="$ckpt"
  LEARNED+=("$dom")

  for d in "${LEARNED[@]}"; do
    echo "-------------------- [t=$t] $d 評価 --------------------"
    bash tools/dist_test.sh "$(eval_cfg "$d")" "$prev" "$GPUS" \
      --work-dir "$EVALDIR/t${t}_on_${d}"
  done

  echo "-------------------- [t=$t] ZCOCO 評価 --------------------"
  bash tools/dist_test.sh configs/mm_grounding_dino/eval_base_coco.py "$prev" \
    "$GPUS" --work-dir "$EVALDIR/t${t}_zcoco"
done

echo "==================== exp_040 $COND 完了（最終: $prev）===================="
