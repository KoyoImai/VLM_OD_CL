# 事前学習済みVision Language Modelを用いた物体検出の継続学習

## プロジェクト概要
事前学習済みVision Language Model (VLM) は汎用的な特徴表現を事前学習で獲得しており，COCO2017などの物体検出データセットで優れた性能を達成する．
しかし，既存のアプローチは主に二つの課題が残されていると考えられる．

- 事前学習データセットに含まれない，あるいはデータ数が少ないドメインのデータに対する知識が不足し，検出性能が低下する，

- 新しいドメインの知識を学習すると，事前学習した汎用的な知識や過去に学習したドメインの知識を消失する破滅的忘却が発生する．

これらの課題に対処する「事前学習済みVLMを対象とした，ドメインが増加する物体検出の継続学習」は研究されていない．
そこで，本プロジェクトでは，これらの課題に対処することを目的とする．

## 行動原理
1. 実験を実行する前に必ず `experiments/exp_NNN/design.md` を作成し，ユーザーの承認を得る．

2. 承認なしに実験を実行しない．

3. 実験結果を報告する際は，必ず次の問いや改善案を複数提案する．

4. 新しい論文を読んだら `papers/` に要約を残す．


## ディレクトリ構成
- papers/：論文PDF・要約
- experiments/exp_NNN/：実験ごとのディレクトリ
  - design.md：実験設計書（承認ゲート）
  - results/：実験結果
  - outputs/figures/：可視化結果
  - outputs/notes.md：実験メモ
- .claude/commands/：カスタムスラッシュコマンド

## このリポジトリの概要

OpenMMLab **mmdetection**（v3.x）をベースに，Vision Language Model の物体検出継続学習の研究プロジェクトに使用している．
事前学習済み VLM として MM-Grouding DINO を対象としている．
`mmdet/` と `tools/` 配下の上流ライブラリのコードはほぼそのまま使用している．
本研究プロジェクト固有のプログラムについては以下の方向性でまとめる．

- `configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_*.py` ー Robotflow100の各ドメイン毎の学習 config ファイル
- `README.md` — データセット，ドメイン毎の統計，正確な学習コマンドを記した正本．データセットの学習や作業を行う前に**まず読むこと**，上流のドキュメントではなく，このREADMEが信頼できる情報源．
- `experiments/exp_NNN/outputs/*_work_dir` — ドメイン毎の学習出力（チェックポイント，ログ，`vis_data/scalars.json`）を配置．ソースではないので編集はしない．

ライブラリは editable インストール（`pip install -e .`）されているため、`mmdet/` への変更は
再インストールなしで反映される。

## 中核となるアーキテクチャ（学習実行の仕組み）

mmdetection は mmengine の `Registry` を介した**config駆動の構成**を採用している。エントリポイントに
ハードコードは無く、`tools/train.py` は config の辞書だけから `Runner` を組み立てる：

- **config は `_base_` で継承される。** 各 `mm_grounding_dino/*_<domain>.py` は
  `_base_ = 'grounding_dino_swin-t_pretrain_obj365.py'` を設定し、差分（データセット、クラス、
  スケジュール）のみを上書きする。`_base_/` に共通の datasets/models/schedules/runtime がある。
- **config 文字列の `type='CocoDataset'` は実行時に**登録済みクラスへ解決される。新しい構成要素
  （dataset, transform, head）を追加するには、`mmdet/` 配下で登録し、config から文字列で参照する。
- **ドメインごとの config パターン**（`..._videogames.py` を参照）：`data_root`、`class_name`
  タプルと `metainfo`、`model.bbox_head.num_classes`、train/val dataloader の `ann_file` と
  `data_prefix`、evaluator の `ann_file`、スケジュール（`max_epoch`, `param_scheduler`）、
  `load_from`（事前学習済み Grounding DINO チェックポイント）を上書きする。backbone と
  `language_model` は `optim_wrapper.paramwise_cfg.custom_keys`（`lr_mult=0.0`）で凍結される。

### プロジェクト固有の注意点（上流ドキュメントには無い）

- **テキストトークン数の上限。** MM-Grounding DINO の既定は `max_text_len=256`。クラス数の多い
  ドメイン（`videogames` 87、`real_world` 405）はこれを超えて失敗する。本プロジェクトでの対処は
  config で値を上げること：
  `model=dict(bbox_head=dict(num_classes=..., contrastive_cfg=dict(max_text_len=512, ...)))`。
- **RF100 のダミーカテゴリ対策。** RF100 の `_annotations.coco.json` にはダミーカテゴリが含まれ、
  クラス数の不一致で学習がクラッシュする。対処（READMEに記載）は `_annotations.coco_fixed.json`
  にコピーしてダミーを削除し、config をその修正済みファイルに向けること。
- ドメイン統合データはリポジトリ外の
  `/workspace/kouyou/datasets/rf100_domain/<domain>/<train|valid>/` にある。

## コマンド

リポジトリのルートで実行する。`<domain>` は
`underwater | aerial | videogames | microscopic | documents | electromagnetic | real_world`
（および `cat`, `smoke`）。

```bash
# 学習（1 GPU）
python tools/train.py configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_<domain>.py --work-dir <domain>_work_dir

# 学習（4 GPU・分散）
bash tools/dist_train.sh configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_<domain>.py 4 --work-dir <domain>_work_dir

# チェックポイントの評価（1 GPU）
python tools/test.py <config> <checkpoint.pth>

# 評価（4 GPU）
bash tools/dist_test.sh <config> <checkpoint.pth> 4

# work-dir 内の最新チェックポイントから再開
python tools/train.py <config> --work-dir <dir> --resume

# 混合精度 / LRの自動スケーリング
python tools/train.py <config> --amp --auto-scale-lr
```

config ファイルを編集せずに任意の値をコマンドラインから上書きする：
```bash
python tools/train.py <config> --cfg-options train_cfg.max_epochs=10 optim_wrapper.optimizer.lr=0.0002
```

推論・予測結果の可視化は `visualize_prediction.ipynb` で行う。

### テストと lint

```bash
pytest tests/                              # 全テスト
pytest tests/test_models/test_detectors/  # サブディレクトリ単位
pytest tests/path/to/test_file.py::test_name

pre-commit run --all-files                 # flake8, yapf, isort など（設定は .pre-commit-config.yaml）
```
