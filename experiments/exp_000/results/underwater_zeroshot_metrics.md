# exp_000 underwater zero-shot 評価結果

- 日時: 2026-06-14
- モデル: MM-Grounding DINO Swin-T（事前学習 obj365+goldg+grit9m+v3det、fine-tune なし）
- データ: rf100_domain/underwater/valid（3,576 枚 / 28 クラス / 16,927 アノテ）
- ログ: `experiments/exp_000/underwater_work_dir/20260614_050847/`

## COCO 指標（bbox）

| 指標 | 値 |
|---|---:|
| mAP (IoU=0.50:0.95) | **0.051** |
| mAP_50 | 0.075 |
| mAP_75 | 0.052 |
| mAP_small | 0.009 |
| mAP_medium | 0.049 |
| mAP_large | 0.082 |
| AR@100 | 0.207 |
| AR@300 | 0.209 |
| AR@1000 | 0.209 |
| AR_small | 0.105 |
| AR_medium | 0.203 |
| AR_large | 0.382 |

copypaste: `0.051 0.075 0.052 0.009 0.049 0.082`
