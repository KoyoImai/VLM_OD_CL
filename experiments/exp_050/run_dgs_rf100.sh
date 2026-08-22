#!/usr/bin/env bash
# =============================================================================
# exp_050: DGS の RF100 6 ドメイン逐次ドライバ（design.md の承認後に実行）
#
#   0. DTG 用特徴が無ければ抽出（θ0・全量・約 1〜1.5 h。決定的）
#   各ドメイン t（1..6）で:
#     1. DTG: adaptive_task_mapping で task_id_mapping.yaml を更新
#     2. **併合（既存グループ入り）が出たら停止**（design §2.3。ODVG 用 stage2 は
#        未実装のため、追加実装の承認を得てから再開する）
#     3. stage1 で学習（20 epoch・lr 1e-4・dn 有効・AMP 無し）。ckpt は epoch_20 のみ
#     4. 評価: 学習済みドメイン 1..t（per-domain・ルーティング付き）＋ ZCOCO（ood 200）
#
#   再開: epoch_20.pth のあるドメインは学習をスキップ。評価はディレクトリ有無で
#   スキップ（json の無い評価ディレクトリは消してから再投入すること）。
#
# 使い方: bash experiments/exp_050/run_dgs_rf100.sh    # GPUS=4 が既定
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

EXP=experiments/exp_050
CFG="$EXP/configs"
WD="$EXP/work_dirs"
EVALDIR="$EXP/eval"
MAPPING="$WD/task_id_mapping.yaml"
mkdir -p "$WD" "$EVALDIR" "$EXP/stats"

ORDER=(underwater electromagnetic videogames aerial microscopic documents)

echo "==================== exp_050 DGS RF100 ===================="
echo "GPUS=$GPUS / θ0: $THETA0"

# 0. 特徴抽出（未抽出なら。GPU 1 枚で足りる）
if [ ! -d "$EXP/feats/documents/train" ]; then
  echo "-------------------- DTG 用特徴を抽出 --------------------"
  CUDA_VISIBLE_DEVICES=0 python "$EXP/extract_feats_rf100.py" --theta0 "$THETA0"
fi

prev="$THETA0"

for i in "${!ORDER[@]}"; do
  t=$((i + 1))
  dom="${ORDER[$i]}"
  seen=$(IFS=,; echo "${ORDER[*]:0:$t}")
  TWD="$WD/$dom"
  ckpt="$TWD/epoch_20.pth"
  printf -v tt '%02d' "$t"

  echo "==================== [t=$t/6] $dom ===================="
  if [ -f "$ckpt" ]; then
    echo "[t=$t] $ckpt があるため学習をスキップ"
  else
    # 1. DTG
    python projects/dgs_cl/tools/adaptive_task_mapping.py \
      --config "$CFG/dgs_stage1_t${tt}_${dom}.py" --task_id $i \
      --seen_tasks "$seen" --num_tasks ${#ORDER[@]} --work_dirs "$WD"

    # 2. stage 判定（併合が出たら停止。design §2.3）
    last_group=$(tail -n 1 "$MAPPING" | cut -d':' -f2 | tr -d ' ')
    count=$(grep -cE ": *${last_group}\$" "$MAPPING")
    echo "[t=$t] group=$last_group（所属タスク数 $count）"
    if (( count > 1 )); then
      echo "[t=$t] 既存グループへの併合が発生。ODVG 用 stage2 は未実装のため停止する"
      echo "        （design.md §2.3。追加実装の承認後に再開）"
      exit 2
    fi
    echo "[t=$t] stage1 で学習"

    # 3. 学習
    bash tools/dist_train.sh "$CFG/dgs_stage1_t${tt}_${dom}.py" "$GPUS" \
      --work-dir "$TWD" --cfg-options "load_from=$prev"
  fi
  prev="$ckpt"

  # 4. 評価（学習済みドメイン per-domain ＋ ZCOCO）
  EOPTS=("model.task_id=$i" "model.seen_tasks=$seen"
         "model.domain_predictor_cfg.task_id_mapping_path=$MAPPING")
  for k in $(seq 0 $i); do
    d="${ORDER[$k]}"
    [ -d "$EVALDIR/t${tt}_on_${d}" ] || \
      bash tools/dist_test.sh "$CFG/eval_dgs_${d}.py" "$prev" "$GPUS" \
        --work-dir "$EVALDIR/t${tt}_on_${d}" --cfg-options "${EOPTS[@]}"
  done
  [ -d "$EVALDIR/t${tt}_zcoco" ] || \
    bash tools/dist_test.sh "$CFG/eval_dgs_zcoco.py" "$prev" "$GPUS" \
      --work-dir "$EVALDIR/t${tt}_zcoco" --cfg-options "${EOPTS[@]}"
done

echo "==================== exp_050 DGS RF100 完了（最終: $prev）===================="
