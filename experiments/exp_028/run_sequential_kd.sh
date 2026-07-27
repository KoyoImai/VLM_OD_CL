#!/usr/bin/env bash
# =============================================================================
# exp_028 / exp_029 逐次ドライバ: 特徴蒸留 + リプレイ
#   使い方: bash experiments/exp_028/run_sequential_kd.sh <EXP> <COND> <FORM> <KD>
#     EXP  : exp_028 | exp_029 | exp_030 | exp_031
#     COND : condA | condB
#     FORM : l2 | cos
#     KD   : A | B | D | E
#   例:  bash experiments/exp_028/run_sequential_kd.sh exp_028 condA l2 A
#
#   各ドメイン t で:
#     1. 教師 θ^T を決める（t=1 は θ0、t>=2 は前ドメインの last）
#     2. 学習（load_from = θ^T、teacher_ckpt = θ^T、loss_weight = λ）
#     3. 現ドメイン + 全過去ドメイン + ZCOCO を評価
#     4. この last を次ドメインの θ^T にして次へ
#
#   λ は既定で 1.0 固定（2026-07-27 決定。design §4.5）。
#     LAMBDA=<値>  ... 固定値を変える
#     CALIBRATE=1  ... 参照ドリフト方式で校正する（§4.5 後半・参考。exp_027 の ckpt が必要）
#
#   ※ 実行は design.md の承認後（行動原理3）。既存ファイルは無変更。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."   # experiments/exp_0NN/ -> リポジトリルート

EXP="${1:?EXP を指定してください（exp_028|exp_029|exp_030|exp_031）}"
COND="${2:?COND を指定してください（condA|condB）}"
FORM="${3:?FORM を指定してください（l2|cos）}"
KD="${4:?KD を指定してください（A|B|D|E）}"

GPUS=4
export PORT="${PORT:-29660}"
CFG="experiments/$EXP/configs"
EVALCFG=experiments/exp_023/configs      # eval_{domain}.py を流用
OUT="experiments/$EXP"
ZCOCO_CFG=configs/mm_grounding_dino/eval_base_coco.py
TAG="kd${KD}_${COND}_${FORM}"

# リプレイのみの参照実験（λ 校正の参照学生 θ^ref）
case "$COND" in
  condA) REFDIR=experiments/exp_027/fullft_replay ;;
  condB) REFDIR=experiments/exp_027/condB_replay ;;
  *) echo "unknown COND: $COND"; exit 1 ;;
esac

# θ0。THETA0 が未設定なら torch hub キャッシュの既定位置。
THETA0="${THETA0:-$HOME/.cache/torch/hub/checkpoints/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth}"
[ -f "$THETA0" ] || { echo "ERROR: θ0 が見つかりません: $THETA0"; exit 1; }

INITCFG_OPT=(model.backbone.init_cfg=None)

ORDER=(underwater electromagnetic videogames)
PAST=()
teacher="$THETA0"          # t=1 の教師は θ0

for dom in "${ORDER[@]}"; do
  WD="$OUT/${TAG}_${dom}_work_dir"
  echo "==================== [$EXP][$TAG][$dom] ===================="
  echo "教師 θ^T = $teacher"

  # ---- λ の決定 -------------------------------------------------------------
  # 既定は λ = 1.0 固定（2026-07-27 決定。design §4.5）。
  # CALIBRATE=1 を指定した場合のみ、参照ドリフト方式で校正する（§4.5 後半・参考）。
  if [ "${CALIBRATE:-0}" = "1" ]; then
    reflast="$REFDIR/${dom}_work_dir/last_checkpoint"
    if [ ! -f "$reflast" ]; then
      echo "ERROR: λ 校正の参照 last_checkpoint がありません: $reflast"
      echo "       exp_027（$COND）の $dom が未完了です。"
      exit 1
    fi
    ref="$(cat "$reflast")"
    if [ ! -f "$ref" ]; then
      echo "ERROR: λ 校正の参照 ckpt が見つかりません: $ref"
      exit 1
    fi
    calib="$OUT/calibration/${TAG}_${dom}.json"
    echo "-------------------- [$dom] λ の校正 --------------------"
    python experiments/exp_028/calibrate_lambda.py \
      --config "$CFG/${TAG}_${dom}.py" \
      --student "$ref" --teacher "$teacher" --out "$calib"
    lam="$(python -c "import json;print(json.load(open('$calib'))['lambda_for_this_config'])")"
    echo "[$dom] λ = $lam （校正値）"
  else
    lam="${LAMBDA:-1.0}"
    echo "[$dom] λ = $lam （固定値。校正は行わない）"
  fi

  # ---- 学習 -----------------------------------------------------------------
  bash tools/dist_train.sh "$CFG/${TAG}_${dom}.py" "$GPUS" --work-dir "$WD" \
    --cfg-options load_from="$teacher" model.teacher_ckpt="$teacher" \
                  model.kd.loss_weight="$lam" "${INITCFG_OPT[@]}"

  if [ ! -f "$WD/last_checkpoint" ]; then
    echo "ERROR: 学習が last_checkpoint を残していません: $WD"
    echo "       学習が正常終了したかを確認してください。"
    exit 1
  fi
  last="$(cat "$WD/last_checkpoint")"
  echo "[$EXP][$TAG][$dom] last ckpt: $last"

  # ---- 評価 -----------------------------------------------------------------
  for evd in "$dom" "${PAST[@]}"; do
    echo "-------------------- [$TAG][$dom] 評価 on $evd --------------------"
    bash tools/dist_test.sh "$EVALCFG/eval_${evd}.py" "$last" "$GPUS" \
      --work-dir "$OUT/${TAG}_eval_after_${dom}/on_${evd}"
  done
  echo "-------------------- [$TAG][$dom] ZCOCO --------------------"
  bash tools/dist_test.sh "$ZCOCO_CFG" "$last" "$GPUS" \
    --work-dir "$OUT/${TAG}_eval_after_${dom}/zcoco"

  teacher="$last"          # 次ドメインの教師 = この last
  PAST+=("$dom")
done

echo "==================== $EXP $TAG 完了 ===================="
