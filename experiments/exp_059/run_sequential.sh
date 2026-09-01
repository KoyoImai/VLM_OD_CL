#!/usr/bin/env bash
# =============================================================================
# exp_059: 蒸留係数 λ 感度分析（Ours・6 ドメイン）の逐次ドライバ
#          （design.md の承認後に実行。2026-09-01 承認済み）
#   exp_058 と同じ構造。違いは config が exp_059（λ 焼き込み版）であることだけ。
#   ★ loss_weight は --cfg-options で渡さない（exp_033 事故防止。design §1.1）。
# 使い方: COND=w50 GPUS=4 bash experiments/exp_059/run_sequential.sh   # w50|w150|w200
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

COND="${COND:?COND=w50|w150|w200 を指定}"
case "$COND" in w50|w150|w200) ;; *) echo "COND は w50|w150|w200"; exit 1;; esac
GPUS="${GPUS:-4}"
if [ -z "${THETA0:-}" ]; then
  for p in \
    grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth \
    /workspace/kouyou/ckpt/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth; do
    [ -f "$p" ] && THETA0="$p" && break
  done
fi
[ -n "${THETA0:-}" ] || { echo "θ0 が見つからない"; exit 1; }

EXP=experiments/exp_059
EVALDIR="$EXP/eval_${COND}"
mkdir -p "$EVALDIR"
ORDER=(underwater electromagnetic videogames aerial microscopic documents)
eval_cfg() {
  case "$1" in
    underwater|electromagnetic|videogames) echo "experiments/exp_023/configs/eval_$1.py" ;;
    *) echo "experiments/exp_026/configs/eval_$1.py" ;;
  esac
}

echo "==================== exp_059 $COND ===================="
prev=""; LEARNED=()
for i in "${!ORDER[@]}"; do
  t=$((i + 1)); dom="${ORDER[$i]}"
  WD="$EXP/${COND}_${dom}_work_dir"; ckpt="$WD/epoch_20.pth"
  echo "==================== [t=$t/6] $dom 学習 ===================="
  if [ -f "$ckpt" ]; then
    echo "[t=$t] $ckpt があるためスキップ"
  else
    teacher="${prev:-$THETA0}"
    bash tools/dist_train.sh "$EXP/configs/${COND}_${dom}.py" "$GPUS" \
      --work-dir "$WD" --cfg-options \
      "model.teacher_ckpt=$teacher" "load_from=$teacher"
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
echo "==================== exp_059 $COND 完了 ===================="
