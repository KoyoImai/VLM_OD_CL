#!/usr/bin/env bash
# =============================================================================
# exp_021 学習＋ZCOCO評価 駆動スクリプト（DitHub 再現・標準版・単発ドメイン）
#   対象ドメイン: underwater / aerial / videogames / microscopic /
#                 documents / electromagnetic（6。real_world は除外）
#   - 各: dist_train(4GPU・20ep, warmup10+spec10) →
#         specialization 期 (ep11-20) の best epoch を scalars.json から選び
#         (裁定④)、その ckpt を dithub_eval_coco_{domain}.py で ZCOCO 評価 →
#         余分な ckpt を削除（best と epoch_20 のみ残す）
#   - work_dir: experiments/exp_021/{domain}_dithub_work_dir
#   - ZCOCO   : experiments/exp_021/{domain}_dithub_zcoco
#   - ログ    : experiments/exp_021/train.log
# 使い方（リポジトリルートで実行）:
#   bash experiments/exp_021/run_train.sh                # 全6本
#   bash experiments/exp_021/run_train.sh underwater     # ドメイン指定
# 注: 実学習（高コスト）。design.md 承認後に実行。seed=0 は config 側で固定。
# =============================================================================
set -uo pipefail

REPO=/workspace/kouyou/mmdetection
cd "$REPO"

GPUS=4
PORT=${PORT:-29530}
FAILED=()

DEFAULT_DOMAINS=(underwater aerial videogames microscopic documents electromagnetic)
if [ "$#" -ge 1 ]; then DOMAINS=("$@"); else DOMAINS=("${DEFAULT_DOMAINS[@]}"); fi

EXP_DIR=experiments/exp_021
LOG="$EXP_DIR/train.log"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

# specialization 期 (ep11-20) の best epoch 番号を scalars.json から選ぶ
best_spec_epoch() {
  local wd=$1
  python3 - "$wd" <<'EOF'
import json, sys, glob
scalars = sorted(glob.glob(f'{sys.argv[1]}/*/vis_data/scalars.json'))
best = (-1.0, None)
for path in scalars:
    for line in open(path):
        d = json.loads(line)
        if 'coco/bbox_mAP' in d and d.get('step', 0) >= 11:
            if d['coco/bbox_mAP'] > best[0]:
                best = (d['coco/bbox_mAP'], d['step'])
print(best[1] if best[1] is not None else '')
EOF
}

echo "[$(ts)] exp_021 学習開始: ドメイン=${DOMAINS[*]} (DitHub 標準版)" | tee -a "$LOG"
for d in "${DOMAINS[@]}"; do
  cfg="$EXP_DIR/configs/dithub_${d}.py"
  zcfg="$EXP_DIR/configs/dithub_eval_coco_${d}.py"
  wd="$EXP_DIR/${d}_dithub_work_dir"
  if [ ! -f "$cfg" ] || [ ! -f "$zcfg" ]; then
    echo "[$(ts)] [${d}] SKIP: config 無し" | tee -a "$LOG"; FAILED+=("${d} config"); continue
  fi

  echo "[$(ts)] [${d}] train 開始 -> $wd" | tee -a "$LOG"
  PORT=$PORT bash tools/dist_train.sh "$cfg" "$GPUS" --work-dir "$wd" 2>&1 | tee -a "$LOG"
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "[$(ts)] [${d}] FAIL (train)" | tee -a "$LOG"; FAILED+=("${d} train"); continue
  fi

  ep=$(best_spec_epoch "$wd")
  best="$wd/epoch_${ep}.pth"
  if [ -z "$ep" ] || [ ! -f "$best" ]; then
    echo "[$(ts)] [${d}] FAIL: specialization 期の best ckpt 無し (ep=${ep:-none})" | tee -a "$LOG"
    FAILED+=("${d} best_ckpt"); continue
  fi
  echo "[$(ts)] [${d}] spec 期 best = epoch_${ep} / ZCOCO 評価 -> ${d}_dithub_zcoco" | tee -a "$LOG"
  PORT=$PORT bash tools/dist_test.sh "$zcfg" "$best" "$GPUS" \
    --work-dir "$EXP_DIR/${d}_dithub_zcoco" 2>&1 | tee -a "$LOG"
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "[$(ts)] [${d}] FAIL (zcoco)" | tee -a "$LOG"; FAILED+=("${d} zcoco")
  fi

  # ディスク節約: best と epoch_20 以外の ckpt を削除
  for f in "$wd"/epoch_*.pth; do
    b=$(basename "$f")
    if [ "$b" != "epoch_${ep}.pth" ] && [ "$b" != "epoch_20.pth" ]; then rm -f "$f"; fi
  done
done

echo "[$(ts)] exp_021 (標準版) 完了" | tee -a "$LOG"
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "=== 失敗 ${#FAILED[@]} 件 ===" | tee -a "$LOG"
  printf '  - %s\n' "${FAILED[@]}" | tee -a "$LOG"
  exit 1
fi
echo "=== 標準版 全ドメイン 正常終了 ===" | tee -a "$LOG"
