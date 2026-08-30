#!/usr/bin/env bash
# =============================================================================
# exp_054 ZiRa / DitHub（replay free 変種）: seed 変更版の 6 ドメイン通しドライバ
#   exp_039（t1-3）と exp_041（t4-6）のドライバを 1 本に結合し、
#   randomness.seed=$SEED / 出力先 / last のみ保存だけを変更（design.md §1）。
#   融合・評価の手順は元実験と同一:
#     ZiRa   : merge_hlrb.py（Rep+）。評価は t<=3 と前半+zcoco が exp_023 の zira_eval_*、
#              後半ドメインが exp_041 の zira_eval_*
#     DitHub : merge_dithub.py（式4 + A 和集合。t=1 はコピー）。評価は t<=3 が exp_023、
#              t>=4 が exp_041 の 260 クラス和集合版
# 使い方: METHOD=zira SEED=1 bash experiments/exp_054/run_zira_dithub.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
METHOD="${METHOD:?METHOD=zira|dithub を指定}"
SEED="${SEED:?SEED=1|2 を指定}"
GPUS="${GPUS:-4}"
case "$METHOD" in
  zira)   MERGE=experiments/exp_020/merge_hlrb.py ;;
  dithub) MERGE=experiments/exp_021/merge_dithub.py ;;
  *) echo "METHOD は zira か dithub"; exit 1 ;;
esac

EXP=experiments/exp_054
TAG="${METHOD}_replayfree"
MERGEDIR="$EXP/${TAG}_s${SEED}_merged"
EVALDIR="$EXP/eval_${TAG}_s${SEED}"
mkdir -p "$MERGEDIR" "$EVALDIR"
ORDER=(underwater electromagnetic videogames aerial microscopic documents)

train_cfg() {  # $1=domain
  case "$1" in
    underwater|electromagnetic|videogames) echo "experiments/exp_039/configs/${TAG}_$1.py" ;;
    *) echo "experiments/exp_041/configs/${TAG}_$1.py" ;;
  esac
}
eval_cfg() {  # $1=domain|zcoco  $2=t
  if [ "$METHOD" = "zira" ]; then
    case "$1" in
      underwater|electromagnetic|videogames|zcoco)
        echo "experiments/exp_023/configs/zira_eval_$1.py" ;;
      *) echo "experiments/exp_041/configs/zira_eval_$1.py" ;;
    esac
  else
    if [ "$2" -le 3 ]; then
      echo "experiments/exp_023/configs/dithub_eval_$1.py"
    else
      echo "experiments/exp_041/configs/dithub_eval_$1.py"
    fi
  fi
}
CKPT_OPTS=("default_hooks.checkpoint.interval=20" "default_hooks.checkpoint.save_optimizer=False" \
           "default_hooks.checkpoint.save_param_scheduler=False")

echo "==================== exp_054 $METHOD (replay free) seed=$SEED ===================="
prev=""; LEARNED=()
for i in "${!ORDER[@]}"; do
  t=$((i + 1)); dom="${ORDER[$i]}"
  WD="$EXP/${TAG}_s${SEED}_${dom}_work_dir"
  merged="$MERGEDIR/merged_after_t${t}_${dom}.pth"
  echo "==================== [t=$t/6] $dom 学習 ===================="
  if [ -f "$merged" ]; then
    echo "[t=$t] $merged があるためスキップ"
  else
    OPTS=("randomness.seed=$SEED" "${CKPT_OPTS[@]}")
    [ -n "$prev" ] && OPTS+=("load_from=$prev")
    bash tools/dist_train.sh "$(train_cfg "$dom")" "$GPUS" --work-dir "$WD" \
      --cfg-options "${OPTS[@]}"
    ckpt="$(cat "$WD/last_checkpoint")"
    if [ "$METHOD" = "dithub" ] && [ -z "$prev" ]; then
      cp "$ckpt" "$merged"
    elif [ "$METHOD" = "dithub" ]; then
      python "$MERGE" "$prev" "$ckpt" "$merged"
    else
      python "$MERGE" "$ckpt" "$merged"
    fi
  fi
  prev="$merged"; LEARNED+=("$dom")
  for d in "${LEARNED[@]}"; do
    [ -d "$EVALDIR/t${t}_on_${d}" ] || \
      bash tools/dist_test.sh "$(eval_cfg "$d" "$t")" "$prev" "$GPUS" \
        --work-dir "$EVALDIR/t${t}_on_${d}"
  done
  [ -d "$EVALDIR/t${t}_zcoco" ] || \
    bash tools/dist_test.sh "$(eval_cfg zcoco "$t")" "$prev" "$GPUS" \
      --work-dir "$EVALDIR/t${t}_zcoco"
done
echo "==================== exp_054 $METHOD seed=$SEED 完了 ===================="
