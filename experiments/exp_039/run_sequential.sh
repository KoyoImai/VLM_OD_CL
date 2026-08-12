#!/usr/bin/env bash
# =============================================================================
# exp_039: ZiRa / DitHub の RF100 逐次学習ドライバ（design.md の承認後に実行する）
#
#   各ドメイン t で:
#     1. 学習（20 epoch）。load_from は前ドメインの融合済み ckpt（t=1 は config の θ0）。
#     2. タスク境界処理:
#          ZiRa   : experiments/exp_020/merge_hlrb.py で Rep+
#                   （W_llrb += s*W_hlrb、HLRB を 1e-8・s を 0.1 に再初期化）
#          DitHub : experiments/exp_021/merge_dithub.py で式4（B の λ_B=0.7 融合）
#                   ＋ クラス別 A の和集合。t=1 は式4 を実行せず ckpt をそのまま採用。
#     3. 評価: 学習済みドメイン（1..t）と ZCOCO。exp_023 の評価 config を使う。
#
#   逐次順: underwater -> electromagnetic -> videogames
#   途中で止まった場合は同じコマンドで再開できる（融合済み ckpt があるドメインはスキップ）。
#
# 使い方:
#   METHOD=zira   REPLAY=replayfree bash experiments/exp_039/run_sequential.sh
#   METHOD=dithub REPLAY=replay     bash experiments/exp_039/run_sequential.sh
#   GPUS=4 を指定すると dist_train.sh / dist_test.sh を使う（クラスタ既定）。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

METHOD="${METHOD:?METHOD=zira|dithub を指定}"
REPLAY="${REPLAY:?REPLAY=replay|replayfree を指定}"
GPUS="${GPUS:-4}"
PORT="${PORT:-29500}"

case "$METHOD" in
  zira)   MERGE=experiments/exp_020/merge_hlrb.py ;;
  dithub) MERGE=experiments/exp_021/merge_dithub.py ;;
  *) echo "METHOD は zira か dithub"; exit 1 ;;
esac

EXP=experiments/exp_039
CFG="$EXP/configs"
EVALCFG=experiments/exp_023/configs
TAG="${METHOD}_${REPLAY}"
MERGEDIR="$EXP/${TAG}_merged"
EVALDIR="$EXP/eval_${TAG}"
mkdir -p "$MERGEDIR" "$EVALDIR"

ORDER=(underwater electromagnetic videogames)

echo "==================== exp_039 $METHOD / $REPLAY ===================="
echo "GPUS=$GPUS / 融合: $MERGE / 出力: $MERGEDIR, $EVALDIR"

train() {   # $1=config  $2=work_dir  $3=load_from（空なら config の θ0）
  if [ -n "$3" ]; then
    bash tools/dist_train.sh "$1" "$GPUS" --work-dir "$2" \
      --cfg-options load_from="$3"
  else
    bash tools/dist_train.sh "$1" "$GPUS" --work-dir "$2"
  fi
}

evaluate() {  # $1=config  $2=ckpt  $3=work_dir
  bash tools/dist_test.sh "$1" "$2" "$GPUS" --work-dir "$3"
}

prev=""
LEARNED=()

for i in "${!ORDER[@]}"; do
  t=$((i + 1))
  dom="${ORDER[$i]}"
  WD="$EXP/${TAG}_${dom}_work_dir"
  merged="$MERGEDIR/merged_after_t${t}_${dom}.pth"

  echo "==================== [t=$t/3] $dom 学習 ===================="
  if [ -f "$merged" ]; then
    echo "[t=$t] $merged が既にあるため学習と融合をスキップ"
  else
    train "$CFG/${TAG}_${dom}.py" "$WD" "$prev"
    ckpt="$(cat "$WD/last_checkpoint")"
    echo "[t=$t] last ckpt: $ckpt"
    if [ "$METHOD" = "dithub" ] && [ -z "$prev" ]; then
      cp "$ckpt" "$merged"           # t=1 は式4 を実行しない（公式と同じ）
      echo "[t=$t] library 初期化: $merged"
    elif [ "$METHOD" = "dithub" ]; then
      python "$MERGE" "$prev" "$ckpt" "$merged"
      echo "[t=$t] library 更新: $merged（式4 + A 和集合）"
    else
      python "$MERGE" "$ckpt" "$merged"
      echo "[t=$t] Rep+: $merged"
    fi
  fi
  prev="$merged"
  LEARNED+=("$dom")

  for d in "${LEARNED[@]}"; do
    echo "-------------------- [t=$t] $d 評価 --------------------"
    evaluate "$EVALCFG/${METHOD}_eval_${d}.py" "$prev" \
      "$EVALDIR/t${t}_on_${d}"
  done

  echo "-------------------- [t=$t] ZCOCO 評価 --------------------"
  evaluate "$EVALCFG/${METHOD}_eval_zcoco.py" "$prev" "$EVALDIR/t${t}_zcoco"
done

echo "==================== exp_039 $METHOD / $REPLAY 完了（最終: $prev）===================="
