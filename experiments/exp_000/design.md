# exp_000: 事前学習済み MM-Grounding DINO の zero-shot 評価（underwater）

## 目的
fine-tune を一切行わず、事前学習済み MM-Grounding DINO（Swin-T）を underwater ドメインの
valid セットでそのまま評価し、**継続学習の出発点（lower bound）** となる zero-shot 性能を把握する。

この値は今後の各実験で「学習によってどれだけ性能が向上したか」「忘却でどこまで戻ったか」を
測る基準となる。1 ドメイン（underwater）で評価・集計の流れを確立することも兼ねる。

## 背景・位置づけ
- 本研究の課題のひとつは「事前学習データに乏しいドメインでの検出性能低下」。
  その低下幅を定量化するには、まず fine-tune 前の素の性能を知る必要がある。
- underwater を最初の対象に選ぶ理由：クラス数が 28 と中規模で `max_text_len=256`（既定）に収まり、
  トークン上限の問題が起きないため、純粋な zero-shot 性能を素直に測れる。

## 実験設定
- **モデル**: MM-Grounding DINO Swin-T（事前学習: obj365 + goldg + grit9m + v3det）
  - チェックポイント: config の `load_from` URL
    `grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth`
- **config**: `configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_underwater.py`
  - 学習は行わず、評価パイプライン（val/test_dataloader, evaluator）のみ使用。
  - テキストプロンプト = underwater の 28 クラス名（config の `metainfo.classes`、`return_classes=True`）。
- **データ**: `/workspace/kouyou/datasets/rf100_domain/underwater/valid/`
  - valid 画像 3,576 枚 / アノテ 16,927 / クラス数 28
- **評価指標**: COCO mAP（bbox_mAP, mAP_50, mAP_75, mAP_s/m/l, AR）

## 実行コマンド
```bash
python tools/test.py \
  configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_underwater.py \
  https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth \
  --work-dir ./experiments/exp_000/underwater_work_dir
```

## 期待される成果物
- `experiments/exp_000/underwater_work_dir/` … 評価ログ（`*.log`, `vis_data/`）
- `experiments/exp_000/results/` … COCO mAP の集計（手動コピーまたは整理）
- `experiments/exp_000/outputs/notes.md` … 結果メモ・考察

## 想定される懸念点・確認事項
1. **チェックポイントのダウンロード**: 初回は URL から自動ダウンロードされる（要ネットワーク）。
   失敗する場合は事前に手動取得してローカルパスを渡す。
2. **num_classes の不整合**: config は `bbox_head.num_classes=28`。Grounding DINO の分類はテキスト
   対照に基づくため zero-shot 評価に影響しない想定だが、チェックポイントロード時の警告を確認する。
3. **クラス名とアノテーションの対応**: valid の `_annotations.coco.json` のカテゴリ順と config の
   `class_name` の順序が一致しているか（mAP が極端に低い場合はここを疑う）。
4. **GPU メモリ / 実行時間**: valid 3,576 枚の推論。バッチ等は既定のまま 1 GPU で実行。

## 成功基準（このタスクとしての完了条件）
- 評価がエラーなく完走し、COCO mAP 一式がログに出力されること。
- zero-shot mAP を記録し、`notes.md` に「fine-tune 前の underwater 性能」として残すこと。
- 結果をもとに次の問い（例: ドメイン特有クラスでの低下要因、fine-tune による改善余地）を提示すること。
