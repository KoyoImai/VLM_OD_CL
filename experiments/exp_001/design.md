# exp_001: COCO2017 + RF100 各ドメインの zero-shot 性能の確認

## 目的
fine-tune を行わず、事前学習済み MM-Grounding DINO（Swin-T）を
**COCO2017（汎用ドメイン）と RF100 各ドメイン**でそのまま評価し、
継続学習の **出発点（lower bound）一覧表** を作成する。

- COCO2017 = 事前学習で重視される汎用知識の指標（ZiRa の **ZCOCO** に相当）。
  今後の継続学習で「破滅的忘却によりここがどれだけ下がるか」を測る基準。
- RF100 各ドメイン = ドメイン適応の出発点。fine-tune による伸び（適応余地）を測る基準。

exp_000（underwater 単体 zero-shot, mAP=0.051）を本実験の一部として位置づけ、対象を全ドメイン＋COCO に拡張する。

## 評価対象（COCO + 6 ドメイン = 7 評価）
real_world は方針により除外。**underwater も含め全ドメインを本実験で再評価する**
（exp_000 は単一 GPU 実行だったため、exp_001 では分散推論で統一条件のもと取り直す）。

| 対象 | クラス数 | valid 画像 | BERTトークン数 | max_text_len | 備考 |
|---|---:|---:|---:|---:|---|
| **COCO2017** | 80 | 5,000 | <256 | 256 | base pretrain config を使用 |
| underwater | 28 | 3,576 | 111 | 256 | exp_001 で再評価 |
| aerial | 22 | 1,940 | 77 | 256 | |
| videogames | 87 | 2,219 | 265 | **512** | config で設定済（256超） |
| microscopic | 28 | 2,529 | 131 | 256 | |
| documents | 59 | 4,597 | 207 | 256 | 256 内に収まる（確認済） |
| electromagnetic | 39 | 7,314 | 140 | 256 | |

→ 本実験で実行するのは **COCO + 6 ドメイン = 7 評価**（すべて分散推論で実行）。
exp_000 の underwater(mAP=0.051) は、再評価値との一致確認（健全性チェック）にも用いる。

## 実験設定
- **モデル/重み**: MM-Grounding DINO Swin-T 事前学習
  `grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth`
- **入力解像度**: 既定 `FixScaleResize (800, 1333)`（exp_000 と統一）。
- **実行方式**: **GPU 4 枚を使用した分散推論**（`tools/dist_test.sh ... 4`）で実行する。
  各 GPU に画像を分割して評価するため、単一 GPU より大幅に高速化される。
  評価指標（COCO mAP）は全プロセスの結果を集約して算出されるため、単一 GPU と同一の値になる。
- **指標**: COCO mAP（bbox_mAP, mAP_50/75, s/m/l, AR）。
- **プロンプト**: 各データセットのクラス名を `.` 区切りで連結（`custom_entities`, `return_classes=True`）。

## 実行コマンド
すべて GPU 4 枚の分散推論（`bash tools/dist_test.sh <config> <checkpoint> 4 --work-dir <dir>`）で実行する。

### COCO2017（base pretrain config をそのまま使用）
`_base_/datasets/coco_detection.py` が既に `/workspace/kouyou/datasets/coco2017/` の
val2017 + instances_val2017.json を指しているため、追加設定不要。
```bash
bash tools/dist_test.sh \
  configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py \
  https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth \
  4 \
  --work-dir ./experiments/exp_001/coco_work_dir
```

### RF100 各ドメイン（underwater / aerial / videogames / microscopic / documents / electromagnetic）
```bash
bash tools/dist_test.sh \
  configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_<domain>.py \
  https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth \
  4 \
  --work-dir ./experiments/exp_001/<domain>_work_dir
```

## 期待される成果物
- `experiments/exp_001/<target>_work_dir/` … 各評価ログ
- `experiments/exp_001/results/zeroshot_summary.md` … COCO + 7ドメインの mAP 一覧表（lower bound 表）
- `experiments/exp_001/outputs/notes.md` … 考察・次の問い

## 想定される懸念点
1. **base pretrain config の train_dataloader が objects365 を参照**（存在しない）。`test.py` は test 系のみ
   構築するため評価には影響しない想定。エラーが出れば `--cfg-options` で test_dataloader を明示。
2. **COCO のクラス名・プロンプト**: CocoDataset 既定の 80 クラス metainfo を使用。zero-shot 値が
   文献（MM-GDINO の ZCOCO ≈ 50.6）と大きく乖離しないか健全性チェックする。
3. **videogames(265 tokens)** は `max_text_len=512` 設定済みであることを実行前に再確認。
4. 評価時間: electromagnetic(7,314) と documents(4,597) は画像数が多く時間がかかる。

## 成功基準
- COCO + 6 ドメインの評価が分散推論で完走し、**7 行の lower bound 表**が揃うこと。
- underwater の再評価値が exp_000(mAP=0.051) と一致すること（分散推論の健全性チェック）。
- COCO zero-shot が文献値（≈50.6）と整合（健全性チェック通過）。
- ドメイン間で zero-shot 性能差を比較し、「事前学習知識が効くドメイン/効かないドメイン」を特定、
  次の問い（fine-tune 優先度、プロンプト改善余地など）を提示すること。
