#!/usr/bin/env bash
# =============================================================================
# exp_047 素の LoRA: RF100 6 ドメイン逐次（バッファ不使用）ドライバ
#                    （design.md の承認後に実行する）
#
#   各ドメイン t（1..6）で:
#     1. 学習（20 epoch、4 GPU）。load_from は θ_{t-1}（t=1 は θ0 の実パス）。
#        LoRA は毎タスク B=0 から引き直すので開始時点は θ_{t-1} と厳密に等価。
#     2. マージ: merge_lora.py で W += (alpha/r)·B·A（scaling=1）を焼き込み、
#        分解した MHA を再融合。θ_t は plain 構造。
#     3. 評価: θ_t で学習済みドメイン 1..t と ZCOCO（plain 評価 config）。
#
#   途中で止まった場合は同じコマンドで再開できる（θ_t があるドメインはスキップ）。
#
# 使い方:
#   bash experiments/exp_047/run_lora.sh      # GPUS=4 が既定（クラスタ）
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

GPUS="${GPUS:-4}"

if [ -z "${THETA0:-}" ]; then
  for p in \
    grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth \
    /workspace/kouyou/ckpt/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth; do
    [ -f "$p" ] && THETA0="$p" && break
  done
fi
[ -n "${THETA0:-}" ] || { echo "θ0 が見つからない（THETA0= で指定可）"; exit 1; }

EXP=experiments/exp_047
CFG="$EXP/configs"
THETADIR="$EXP/lora_theta"
EVALDIR="$EXP/eval_lora"
mkdir -p "$THETADIR" "$EVALDIR"

ORDER=(underwater electromagnetic videogames aerial microscopic documents)

eval_cfg() {
  case "$1" in
    underwater|electromagnetic|videogames) echo "experiments/exp_023/configs/eval_$1.py" ;;
    aerial|microscopic|documents)          echo "experiments/exp_026/configs/eval_$1.py" ;;
  esac
}

echo "==================== exp_047 素の LoRA ===================="
echo "GPUS=$GPUS / θ0: $THETA0 / 出力: $THETADIR, $EVALDIR"

prev="$THETA0"
LEARNED=()

for i in "${!ORDER[@]}"; do
  t=$((i + 1))
  dom="${ORDER[$i]}"
  WD="$EXP/lora_${dom}_work_dir"
  theta="$THETADIR/theta_t${t}_${dom}.pth"

  echo "==================== [t=$t/6] $dom ===================="
  if [ -f "$theta" ]; then
    echo "[t=$t] $theta があるためスキップ"
  else
    bash tools/dist_train.sh "$CFG/lora_${dom}.py" "$GPUS" \
      --work-dir "$WD" --cfg-options "load_from=$prev"
    python projects/lora_cl/merge_lora.py "$WD/epoch_20.pth" "$theta"
  fi
  prev="$theta"
  LEARNED+=("$dom")

  for d in "${LEARNED[@]}"; do
    echo "-------------------- [t=$t] $d 評価 --------------------"
    [ -d "$EVALDIR/t${t}_on_${d}" ] || \
      bash tools/dist_test.sh "$(eval_cfg "$d")" "$prev" "$GPUS" \
        --work-dir "$EVALDIR/t${t}_on_${d}"
  done
  echo "-------------------- [t=$t] ZCOCO 評価 --------------------"
  [ -d "$EVALDIR/t${t}_zcoco" ] || \
    bash tools/dist_test.sh configs/mm_grounding_dino/eval_base_coco.py "$prev" \
      "$GPUS" --work-dir "$EVALDIR/t${t}_zcoco"
done

echo "==================== exp_047 素の LoRA 完了（最終: $prev）===================="
