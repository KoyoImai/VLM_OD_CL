#!/usr/bin/env bash
# =============================================================================
# exp_052: 蒸留E × 学習可能モジュールアブレーションの前半 3 ドメイン逐次ドライバ
#          （design.md の承認後に実行する。2026-08-22 承認済み）
#
#   各ドメイン t（1..3）で:
#     1. 学習（20 epoch）。load_from は θ_{t-1}（t=1 は θ0 の実パス。教師と同一
#        ファイル由来でビット一致させる exp_028/046 の規約）。
#        教師 θ_{t-1} を model.teacher_ckpt に渡す（**model. を付ける**）。
#     2. 評価: 学習済みドメイン 1..t と ZCOCO（plain 評価 config）。
#
#   再開: epoch_20.pth のあるドメインは学習をスキップ。評価はディレクトリ有無で
#   スキップ（json の無い評価ディレクトリは消してから再投入すること）。
#
# 使い方:
#   COND=swinNeck bash experiments/exp_052/run_sequential.sh
#   COND は swinNeck|bertTfm|enhancer|qsel|decoder。GPUS=4 が既定（クラスタ）。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

COND="${COND:?COND=swinNeck|bertTfm|enhancer|qsel|decoder を指定}"
case "$COND" in swinNeck|bertTfm|enhancer|qsel|decoder) ;; \
  *) echo "COND は swinNeck|bertTfm|enhancer|qsel|decoder"; exit 1;; esac
GPUS="${GPUS:-4}"

if [ -z "${THETA0:-}" ]; then
  for p in \
    grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth \
    /workspace/kouyou/ckpt/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth; do
    [ -f "$p" ] && THETA0="$p" && break
  done
fi
[ -n "${THETA0:-}" ] || { echo "θ0 が見つからない（THETA0= で指定可）"; exit 1; }

EXP=experiments/exp_052
CFG="$EXP/configs"
EVALDIR="$EXP/eval_${COND}"
mkdir -p "$EVALDIR"

ORDER=(underwater electromagnetic videogames)   # 前半 3 ドメイン（design.md §1）

eval_cfg() {
  case "$1" in
    underwater|electromagnetic|videogames) echo "experiments/exp_023/configs/eval_$1.py" ;;
  esac
}

echo "==================== exp_052 $COND ===================="
echo "GPUS=$GPUS / θ0: $THETA0 / 出力: $EVALDIR"

prev=""
LEARNED=()

for i in "${!ORDER[@]}"; do
  t=$((i + 1))
  dom="${ORDER[$i]}"
  WD="$EXP/${COND}_${dom}_work_dir"
  ckpt="$WD/epoch_20.pth"

  echo "==================== [t=$t/3] $dom 学習 ===================="
  if [ -f "$ckpt" ]; then
    echo "[t=$t] $ckpt が既にあるため学習をスキップ"
  else
    teacher="${prev:-$THETA0}"
    echo "[t=$t] 教師 θ^T = $teacher"
    bash tools/dist_train.sh "$CFG/${COND}_${dom}.py" "$GPUS" \
      --work-dir "$WD" --cfg-options \
      "model.teacher_ckpt=$teacher" "load_from=$teacher"
  fi
  prev="$ckpt"
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

echo "==================== exp_052 $COND 完了（最終: $prev）===================="
