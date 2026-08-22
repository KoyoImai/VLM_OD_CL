#!/usr/bin/env bash
# =============================================================================
# exp_049: DGS の ODinW-13（IVLOD）逐次ドライバ（公式 IVLOD_distn.sh の本環境版）
#          design.md の承認範囲は実装検証まで。本走はユーザー確認後に実行する。
#
#   前提: 特徴抽出が済んでいること
#     CUDA_VISIBLE_DEVICES=0 python experiments/exp_049/extract_feats_odinw13.py
#
#   各タスク t（1..13）で:
#     1. DTG: adaptive_task_mapping で task_id_mapping.yaml を更新
#     2. mapping の末尾グループの出現回数で stage1（新グループ）/ stage2（既存）を選択
#     3. 学習（12 epoch）。load_from は θ_{t-1}（t=1 は θ0 の実パス）
#        ※ AMP は本環境で grad_norm=nan（fp16 の勾配 nan）を確認したため既定で外している。
#          公式は --amp。扱いは design の但し書きとユーザー判断（2026-08-21 検証）
#     4. 評価: eval_after_t{t}（学習済みタスク per-dataset）＋ ZCOCO
#
#   再開: epoch_12.pth のあるタスクは学習をスキップ、評価はディレクトリ有無で
#   スキップ（json の無いディレクトリは消してから再投入すること）。
#
# 使い方: bash experiments/exp_049/run_dgs_odinw13.sh    # GPUS=4 が既定
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

EXP=experiments/exp_049
CFG="$EXP/configs"
WD="$EXP/work_dirs"
EVALDIR="$EXP/eval"
MAPPING="$WD/task_id_mapping.yaml"
mkdir -p "$WD" "$EVALDIR" "$EXP/stats"

# 公式順（design.md §2.1）
ORDER=(AerialMaritimeDrone Aquarium CottontailRabbits EgoHands NorthAmericaMushroom
       Packages PascalVOC pistols pothole Raccoon
       ShellfishOpenImages thermalDogsAndPeople VehiclesOpenImages)

echo "==================== exp_049 DGS ODinW-13 ===================="
echo "GPUS=$GPUS / θ0: $THETA0"
[ -d "$EXP/feats/AerialMaritimeDrone/train" ] || {
  echo "特徴が未抽出です: python experiments/exp_049/extract_feats_odinw13.py"; exit 1; }

prev="$THETA0"

for i in "${!ORDER[@]}"; do
  t=$((i + 1))
  task="${ORDER[$i]}"
  seen=$(IFS=,; echo "${ORDER[*]:0:$t}")
  TWD="$WD/$task"
  ckpt="$TWD/epoch_12.pth"

  echo "==================== [t=$t/13] $task ===================="
  if [ -f "$ckpt" ]; then
    echo "[t=$t] $ckpt があるため学習をスキップ"
  else
    # 1. DTG（task_id_mapping.yaml の更新。統計が無ければ feats から推定される）
    printf -v tt0 '%02d' "$t"
    python projects/dgs_cl/tools/adaptive_task_mapping.py \
      --config "$CFG/dgs_stage1_t${tt0}_${task}.py" --task_id $i \
      --seen_tasks "$seen" --num_tasks ${#ORDER[@]} --work_dirs "$WD"

    # 2. stage の選択（公式と同じ: 末尾グループの出現回数 > 1 なら stage2）
    last_group=$(tail -n 1 "$MAPPING" | cut -d':' -f2 | tr -d ' ')
    count=$(grep -cE ": *${last_group}\$" "$MAPPING")
    echo "[t=$t] group=$last_group（所属タスク数 $count）"
    if (( count > 1 )); then stage=stage2; else stage=stage1; fi
    echo "[t=$t] $stage で学習"

    # 3. 学習（タスク別 config にパス・task_id・seen_tasks は焼き込み済み。
    #    tools/dist_train.sh は引数を非引用転送するため、空白を含む値は
    #    --cfg-options で渡せない。渡すのは load_from だけ）
    printf -v tt0 '%02d' "$t"
    cfg="$CFG/dgs_${stage}_t${tt0}_${task}.py"
    bash tools/dist_train.sh "$cfg" "$GPUS" \
      --work-dir "$TWD" --cfg-options "load_from=$prev"
  fi
  prev="$ckpt"

  # 4. 評価（学習済みタスク per-dataset ＋ ZCOCO）
  EOPTS=("model.domain_predictor_cfg.task_id_mapping_path=$MAPPING")
  printf -v tt '%02d' "$t"
  [ -d "$EVALDIR/t${tt}_odinw" ] || \
    bash tools/dist_test.sh "$CFG/eval_after_t${tt}.py" "$prev" "$GPUS" \
      --work-dir "$EVALDIR/t${tt}_odinw" --cfg-options "${EOPTS[@]}"
  [ -d "$EVALDIR/t${tt}_zcoco" ] || \
    bash tools/dist_test.sh "$CFG/eval_zcoco_dgs.py" "$prev" "$GPUS" \
      --work-dir "$EVALDIR/t${tt}_zcoco" --cfg-options "${EOPTS[@]}" \
      "model.task_id=$i" "model.seen_tasks=$seen"
done

echo "==================== exp_049 DGS 完了（最終: $prev）===================="
