#!/bin/bash
# =============================================================================
# exp_023.5 train_val.sh — コンテナ内で走るデバッグ検証（案A の「中身」）。
#   Slurm も Singularity も知らない。sbatch.sh から singularity exec 経由で呼ばれる。
#   Tier 2-0: COCO ゼロショット（θ0, 期待 ≈0.504。学習不要の数値パリティ本命）
#   Tier 2-1: 1 epoch デバッグ学習（fullft_replay_underwater_debug.py）
#   Tier 2-2: 学習済み ckpt の評価（適応 on underwater + ZCOCO）
#   すべて既存 config を無変更で使用（experiments/exp_023/configs/, eval_base_coco.py）。
#   使い方: bash train_val.sh [all|zeroshot|train|eval]   （$1 省略時 all）
# =============================================================================
set -euo pipefail
cd /workspace/kouyou/mmdetection

RUN_STAGE="${1:-all}"
GPUS=4
export PORT="${PORT:-29522}"          # 分散ポート（dist_* の既定 29500 と衝突回避）

CFG=experiments/exp_023/configs
OUT=experiments/exp_023.5
ZCOCO_CFG=configs/mm_grounding_dino/eval_base_coco.py
# θ0（事前学習重み）。オフライン用にホスト /home/kouyou/ckpt を /workspace/kouyou/ckpt へ bind 済み。
THETA0=/workspace/kouyou/ckpt/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth

run_zeroshot() {
  echo "==== [Tier2-0] COCO ゼロショット（θ0, 期待 ≈0.504） ===="
  bash tools/dist_test.sh "$ZCOCO_CFG" "$THETA0" "$GPUS" --work-dir "$OUT/zeroshot_coco"
}
run_train() {
  echo "==== [Tier2-1] 1 epoch デバッグ学習 ===="
  # load_from は config の URL ではなくオフラインの θ0 パスへ差し替え（config 無変更）
  bash tools/dist_train.sh "$CFG/fullft_replay_underwater_debug.py" "$GPUS" \
    --work-dir "$OUT/debug_underwater_work_dir" \
    --cfg-options load_from="$THETA0"
}
run_eval() {
  echo "==== [Tier2-2] 学習済み ckpt の評価（適応 + ZCOCO） ===="
  local ckpt="$OUT/debug_underwater_work_dir/epoch_1.pth"
  if [ -f "$OUT/debug_underwater_work_dir/last_checkpoint" ]; then
    ckpt="$(cat "$OUT/debug_underwater_work_dir/last_checkpoint")"
  fi
  echo "ckpt = $ckpt"
  bash tools/dist_test.sh "$CFG/eval_underwater.py" "$ckpt" "$GPUS" \
    --work-dir "$OUT/eval_after_debug/on_underwater"
  bash tools/dist_test.sh "$ZCOCO_CFG" "$ckpt" "$GPUS" \
    --work-dir "$OUT/eval_after_debug/zcoco"
}

case "$RUN_STAGE" in
  zeroshot) run_zeroshot ;;
  train)    run_train ;;
  eval)     run_eval ;;
  all)      run_zeroshot; run_train; run_eval ;;
  *) echo "usage: $0 [all|zeroshot|train|eval]"; exit 1 ;;
esac
echo "==== exp_023.5 train_val.sh ($RUN_STAGE) 完了 ===="
