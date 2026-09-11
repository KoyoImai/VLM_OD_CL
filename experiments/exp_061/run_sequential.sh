#!/usr/bin/env bash
# =============================================================================
# exp_061: バッファサイズ感度（内訳 base） の逐次ドライバ
#   Ours(kdE)/ER(replay) 両対応。ER は teacher_ckpt を持たないため load_from のみ渡す。
# 使い方: METHOD=ours|er COND=b2|b3|b4|b5 GPUS=4 bash experiments/exp_061/run_sequential.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
METHOD="${METHOD:?METHOD=ours|er を指定}"
case "$METHOD" in ours|er) ;; *) echo "METHOD は ours|er"; exit 1;; esac
COND="${COND:?COND=b2|b3|b4|b5 を指定}"
case "$COND" in b2|b3|b4|b5) ;; *) echo "COND は b2|b3|b4|b5"; exit 1;; esac
GPUS="${GPUS:-4}"
if [ -z "${THETA0:-}" ]; then
  for p in \
    grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth \
    /workspace/kouyou/ckpt/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth; do
    [ -f "$p" ] && THETA0="$p" && break
  done
fi
[ -n "${THETA0:-}" ] || { echo "θ0 が見つからない"; exit 1; }

EXP=experiments/exp_061
EVALDIR="$EXP/eval_${METHOD}_${COND}"
mkdir -p "$EVALDIR"
ORDER=(underwater electromagnetic videogames aerial microscopic documents)
eval_cfg() {
  case "$1" in
    underwater|electromagnetic|videogames) echo "experiments/exp_023/configs/eval_$1.py" ;;
    *) echo "experiments/exp_026/configs/eval_$1.py" ;;
  esac
}

echo "==================== exp_061 $METHOD $COND ===================="
prev=""; LEARNED=()
for i in "${!ORDER[@]}"; do
  t=$((i + 1)); dom="${ORDER[$i]}"
  WD="$EXP/${METHOD}_${COND}_${dom}_work_dir"; ckpt="$WD/epoch_20.pth"
  echo "==================== [t=$t/6] $dom 学習 ===================="
  if [ -f "$ckpt" ]; then
    echo "[t=$t] $ckpt があるためスキップ"
  else
    teacher="${prev:-$THETA0}"
    if [ "$METHOD" = "ours" ]; then
      bash tools/dist_train.sh "$EXP/configs/${METHOD}_${COND}_${dom}.py" "$GPUS" \
        --work-dir "$WD" --cfg-options "model.teacher_ckpt=$teacher" "load_from=$teacher"
    else
      bash tools/dist_train.sh "$EXP/configs/${METHOD}_${COND}_${dom}.py" "$GPUS" \
        --work-dir "$WD" --cfg-options "load_from=$teacher"
    fi
  fi
  prev="$ckpt"; LEARNED+=("$dom")
  for d in "${LEARNED[@]}"; do
    [ -d "$EVALDIR/t${t}_on_${d}" ] || \
      bash tools/dist_test.sh "$(eval_cfg "$d")" "$prev" "$GPUS" \
        --work-dir "$EVALDIR/t${t}_on_${d}"
  done
  [ -d "$EVALDIR/t${t}_zcoco" ] || \
    bash tools/dist_test.sh configs/mm_grounding_dino/eval_base_coco.py "$prev" \
      "$GPUS" --work-dir "$EVALDIR/t${t}_zcoco"
done
echo "==================== exp_061 $METHOD $COND 完了 ===================="
