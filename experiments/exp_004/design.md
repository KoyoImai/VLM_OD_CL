# exp_004: underwater fine-tune（backbone・言語エンコーダ 凍結）— 凍結範囲 ablation

## 目的
exp_003（backbone・言語を 0.1× で学習する「全体 fine-tune」）に対し、
**backbone と言語エンコーダを完全凍結**した場合の適応・忘却を測る。
他の設定はすべて exp_003 と同一にし、**「凍結範囲」のみを変数**とする ablation。

検証したい問い:
- 本体凍結により **破滅的忘却（COCO 低下）をどれだけ抑制**できるか。
- その代償として **適応性能（underwater mAP）がどれだけ犠牲**になるか。

## exp_003 との差分（唯一の変更点）
| 項目 | exp_003 | exp_004（本実験） |
|---|---|---|
| backbone lr_mult | 0.1（学習） | **0.0（凍結）** |
| language_model lr_mult | 0.1（学習） | **0.0（凍結）** |
| 学習対象 | head/enc/dec/neck + backbone/言語(0.1×) | **head/enc/dec/neck のみ** |

それ以外（lr=1e-4 / 実効バッチ64=per-GPU8×4GPU×累積2 / 20epochs milestone[15] /
AdamW wd=1e-4 / clip_grad / データ・拡張 / 評価プロトコル (800,1333)）は **exp_003 と完全同一**。

## 実験設定
- config: `experiments/exp_004/configs/underwater_finetune_frozen.py`
  - exp_003 config（`../../exp_003/configs/underwater_finetune.py`）を継承し、paramwise の
    backbone・language_model を lr_mult=0.0 に上書きするのみ。
  - 検証済み: 実効64 / lr1e-4 / 20ep[15] / eval(800,1333) / num_classes=28。
- 初期重み: 事前学習 `...grit9m_v3det...pth`（継承）
- 実行: **GPU 4 枚 分散学習**（`tools/dist_train.sh ... 4`）

## 実行コマンド
```bash
CKPT_DIR=./experiments/exp_004/underwater_work_dir

# 1) 学習（backbone・言語 凍結）
bash tools/dist_train.sh \
  experiments/exp_004/configs/underwater_finetune_frozen.py \
  4 --work-dir $CKPT_DIR

# 2) 適応評価（best ckpt, underwater）
BEST=$(ls $CKPT_DIR/best_coco_bbox_mAP_epoch_*.pth | tail -1)
bash tools/dist_test.sh \
  configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_underwater.py \
  "$BEST" 4 --work-dir ./experiments/exp_004/underwater_eval

# 3) 忘却評価（同 best ckpt, COCO）
bash tools/dist_test.sh \
  configs/mm_grounding_dino/eval_base_coco.py \
  "$BEST" 4 --work-dir ./experiments/exp_004/coco_forgetting_eval
```

## 比較の基準値
| 指標 | zero-shot (exp_001) | 全体FT (exp_003) | 凍結FT (exp_004) |
|---|---:|---:|---|
| underwater mAP（適応） | 0.051 | 0.340 | 本実験で取得 |
| COCO mAP（忘却/ZCOCO） | 0.504 | 0.438 | 本実験で取得 |

- 予想: 凍結により COCO 低下は小さく（忘却抑制）、underwater 適応は exp_003 より低めになる可能性。

## 期待される成果物
- `experiments/exp_004/underwater_work_dir/` … 学習ログ・best ckpt
- `experiments/exp_004/underwater_eval/`, `coco_forgetting_eval/` … 評価ログ
- `experiments/exp_004/results/finetune_frozen_summary.md` … 適応／忘却の数値表（exp_003 比較）
- `experiments/exp_004/outputs/notes.md` … 考察・次の問い

## 想定される懸念点
1. **メモリ**: 凍結により backbone/言語の勾配・optimizer state が不要になるため、exp_003 より
   **VRAM はむしろ減る**見込み（OOM リスクは exp_003 以下）。
2. **収束**: 学習対象が減るため、同 lr・同 epoch で underwater の適応が頭打ちになる可能性。
   学習中 val 推移で確認する（必要なら別途 lr 調整は別実験で）。
3. **COCO 評価のロード警告**: exp_003 同様、underwater(28) で学習した重みを COCO(80) 評価にロード。
   grounding head はテキスト対照ベースで動作する想定。

## 成功基準
- 学習が完走し、underwater 適応 mAP と COCO 忘却 mAP を取得すること。
- exp_003（全体FT）と比較し、**凍結による「忘却抑制量」と「適応の犠牲」を定量化**すること。
- 結果から凍結範囲の妥当性を考察し、次の問い（部分凍結、アダプタ手法）を提示すること。
