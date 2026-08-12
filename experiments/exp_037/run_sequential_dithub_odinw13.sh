#!/usr/bin/env bash
# =============================================================================
# exp_037: DitHub の ODinW-13 逐次学習ドライバ（design.md の承認後に実行する）
#
#   各タスク t で:
#     1. 学習（3000 iter, batch 2, 1 GPU）。load_from は前タスクの成長ライブラリ。
#        t=1 は config の θ0。iter 1500 で warmup -> specialization に切り替わり、
#        現タスクのクラスのうち先行タスクに出現したものには式3が発火する。
#     2. ライブラリ更新: merge_dithub.py で式4（B 融合 λ_B=0.7）＋過去クラス A の和集合。
#        t=1 は最初のタスクなので学習 ckpt をそのまま採用（公式も式4 を実行しない）。
#     3. 評価: 43 クラス分のスロットを持つ評価モデルでライブラリ ckpt を読み、
#        「学習済みタスク（1..t）」と ZCOCO を評価する（design.md §5）。
#
#   学習・評価とも 1 GPU（並列実行しない）。使う GPU は環境変数 GPU で指定（既定 0）。
#   途中で止まった場合は同じコマンドで再開できる（ライブラリが出来ているタスクはスキップ）。
#
# 事前に:
#   python experiments/exp_037/gen_configs.py
#   python experiments/exp_037/check_dithub_odinw13_setup.py   # 全項目 OK を確認
#
# 使い方:
#   bash experiments/exp_037/run_sequential_dithub_odinw13.sh
#   GPU=1 bash experiments/exp_037/run_sequential_dithub_odinw13.sh 42
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."   # experiments/exp_037/ -> リポジトリルート

SEED="${1:-42}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

EXP=experiments/exp_037
CFG="$EXP/configs"
MERGE=experiments/exp_021/merge_dithub.py
LIBDIR="$EXP/dithub_s${SEED}_library"
EVALDIR="$EXP/eval"
mkdir -p "$LIBDIR" "$EVALDIR"

# 逐次順は gen_configs.py と同一（exp_034 の odinw_official_tasks.task_order）
mapfile -t ORDER < <(python -c "
import sys; sys.path.insert(0, 'experiments/exp_034')
from odinw_official_tasks import task_order
print('\n'.join(task_order($SEED)))")

echo "==================== exp_037 DitHub / seed $SEED ===================="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "順序: ${ORDER[*]}"

lib=""   # 前タスクの成長ライブラリ ckpt（空なら config の θ0 を使う）

for i in "${!ORDER[@]}"; do
  t=$((i + 1))
  tt=$(printf '%02d' "$t")
  task="${ORDER[$i]}"
  WD="$EXP/dithub_s${SEED}_t${tt}_${task}_work_dir"
  newlib="$LIBDIR/lib_after_t${tt}_${task}.pth"

  echo "==================== [t=$t/13] $task 学習 ===================="
  if [ -f "$newlib" ]; then
    echo "[t=$t] $newlib が既にあるため学習とライブラリ更新をスキップ"
  else
    if [ -z "$lib" ]; then
      python tools/train.py "$CFG/dithub_odinw13_${task}.py" --work-dir "$WD"
    else
      python tools/train.py "$CFG/dithub_odinw13_${task}.py" --work-dir "$WD" \
        --cfg-options load_from="$lib"
    fi
    ckpt="$(cat "$WD/last_checkpoint")"
    echo "[t=$t] last ckpt: $ckpt"
    if [ -z "$lib" ]; then
      cp "$ckpt" "$newlib"                 # t=1: 式4 は実行しない
      echo "[t=$t] library 初期化: $newlib"
    else
      python "$MERGE" "$lib" "$ckpt" "$newlib"
      echo "[t=$t] library 更新: $newlib（式4 + A 和集合）"
    fi
  fi
  lib="$newlib"

  # --- 評価: 学習済みタスク（1..t）---
  echo "-------------------- [t=$t] ODinW 評価（$t タスク） --------------------"
  python tools/test.py "$CFG/_eval_after_t${tt}.py" "$lib" \
    --work-dir "$EVALDIR/odinw_after_t${tt}"

  # --- 評価: ZCOCO ---
  echo "-------------------- [t=$t] ZCOCO 評価 --------------------"
  python tools/test.py "$CFG/zcoco_eval.py" "$lib" \
    --work-dir "$EVALDIR/zcoco_after_t${tt}"
done

echo "==================== exp_037 DitHub / seed $SEED 完了（library: $lib）===================="
