#!/usr/bin/env bash
# =============================================================================
# exp_038: 素の LoRA の ODinW-13 逐次学習ドライバ（design.md の承認後に実行する）
#
#   t=0: θ0 のゼロショット評価（ODinW-13 全13タスク ＋ ZCOCO）。
#        exp_034 の実測値（Avg 0.4971 / ZCOCO 0.5040）と一致することを確認する。
#
#   各タスク t で:
#     1. 学習（3000 iter, batch 2, 1 GPU）。load_from は θ_{t-1}（t=1 は config の θ0）。
#        LoRA は毎タスク B=0 から引き直すので、開始時点は θ_{t-1} と厳密に等価。
#     2. マージ: merge_lora.py で W += (alpha/r)·BA を焼き込み、分解した MHA を
#        in_proj_weight へ再融合する。出力 θt は plain な GroundingDINO 構造。
#     3. 評価: θt に対して「学習済みタスク（1..t）」と ZCOCO を plain config で評価する。
#
#   学習・評価とも 1 GPU（並列実行しない）。使う GPU は環境変数 GPU で指定（既定 0）。
#   途中で止まった場合は同じコマンドで再開できる（θt が出来ているタスクはスキップ）。
#
# 事前に:
#   python experiments/exp_038/gen_configs.py
#   python experiments/exp_038/check_lora_odinw13_setup.py   # 全項目 OK を確認
#
# 使い方:
#   bash experiments/exp_038/run_sequential_lora_odinw13.sh
#   GPU=1 bash experiments/exp_038/run_sequential_lora_odinw13.sh 42
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."   # experiments/exp_038/ -> リポジトリルート

SEED="${1:-42}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

# 条件の切り替え（design.md §2 = A / §10 = B）。既定は A で従来どおり。
#   A: 232 層（Swin + BERT + encoder + text_feat_map）、lr 1e-3
#   B: 160 層（BERT を凍結）、lr 1e-3
#   C: 232 層（A と同一）、lr 1e-4
#   B / C の θ0 評価は A で実測済みなので行わない。
COND="${COND:-A}"
case "$COND" in
  A) CFGPFX="lora_odinw13";       TAG="s${SEED}";       EVALSUB="eval" ;;
  B) CFGPFX="lora_odinw13_condB"; TAG="condB_s${SEED}"; EVALSUB="eval_condB" ;;
  C) CFGPFX="lora_odinw13_condC"; TAG="condC_s${SEED}"; EVALSUB="eval_condC" ;;
  *) echo "COND は A / B / C"; exit 1 ;;
esac

EXP=experiments/exp_038
CFG="$EXP/configs"
MERGE=projects/lora_cl/merge_lora.py
THETADIR="$EXP/lora_${TAG}_theta"
EVALDIR="$EXP/$EVALSUB"
THETA0='https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'
mkdir -p "$THETADIR" "$EVALDIR"

mapfile -t ORDER < <(python -c "
import sys; sys.path.insert(0, 'experiments/exp_034')
from odinw_official_tasks import task_order
print('\n'.join(task_order($SEED)))")

echo "==================== exp_038 素の LoRA 条件$COND / seed $SEED ===================="
echo "GPU: $CUDA_VISIBLE_DEVICES / config prefix: $CFGPFX / 出力: $THETADIR, $EVALDIR"
echo "順序: ${ORDER[*]}"

# --- t=0: θ0 のゼロショット評価（条件A のみ。B は再測定しない）--------------
if [ "$COND" != "A" ] || [ -d "$EVALDIR/odinw_theta0" ]; then
  echo "[t=0] θ0 評価をスキップ（条件$COND、または既に測定済み）"
else
  echo "==================== [t=0] θ0 ゼロショット評価 ===================="
  python tools/test.py "$CFG/odinw13_plain_eval.py" "$THETA0" \
    --work-dir "$EVALDIR/odinw_theta0"
  python tools/test.py "$CFG/zcoco_eval.py" "$THETA0" \
    --work-dir "$EVALDIR/zcoco_theta0"
fi

prev=""   # θ_{t-1}（空なら config の θ0 を使う）

for i in "${!ORDER[@]}"; do
  t=$((i + 1))
  tt=$(printf '%02d' "$t")
  task="${ORDER[$i]}"
  WD="$EXP/lora_${TAG}_t${tt}_${task}_work_dir"
  theta="$THETADIR/theta_t${tt}_${task}.pth"

  echo "==================== [t=$t/13] $task 学習 ===================="
  if [ -f "$theta" ]; then
    echo "[t=$t] $theta が既にあるため学習とマージをスキップ"
  else
    if [ -z "$prev" ]; then
      python tools/train.py "$CFG/${CFGPFX}_${task}.py" --work-dir "$WD"
    else
      python tools/train.py "$CFG/${CFGPFX}_${task}.py" --work-dir "$WD" \
        --cfg-options load_from="$prev"
    fi
    ckpt="$(cat "$WD/last_checkpoint")"
    echo "[t=$t] last ckpt: $ckpt"
    python "$MERGE" "$ckpt" "$theta"
    echo "[t=$t] θ$t: $theta（LoRA をマージ＋MHA を再融合）"
  fi
  prev="$theta"

  echo "-------------------- [t=$t] ODinW 評価（$t タスク） --------------------"
  python tools/test.py "$CFG/_eval_after_t${tt}.py" "$prev" \
    --work-dir "$EVALDIR/odinw_after_t${tt}"

  echo "-------------------- [t=$t] ZCOCO 評価 --------------------"
  python tools/test.py "$CFG/zcoco_eval.py" "$prev" \
    --work-dir "$EVALDIR/zcoco_after_t${tt}"
done

echo "==================== exp_038 素の LoRA 条件$COND / seed $SEED 完了（最終 θ: $prev）===================="
