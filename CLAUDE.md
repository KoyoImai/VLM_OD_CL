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

## ディレクトリ・ファイル構成

本リポジトリは OpenMMLab **mmdetection（v3.x）**をベースにしており、`mmdet/` `tools/` `configs/`（大部分）`docs/` `tests/` などは上流のコードをほぼそのまま使用している。以下は、**本研究プロジェクトに関連する主要なディレクトリ・ファイル**を示す。

```
VLM_OD_CL-main/
├── CLAUDE.md                       # 本ファイル。プロジェクトの方針・コマンド・注意点
├── README.md                       # データセット・統計・学習コマンドの正本（まず読む）
├── Dockerfile                      # 環境構築用
│
├── configs/mm_grounding_dino/      # MM-Grounding DINO の config 群（上流ベース）
│   └── grounding_dino_swin-t_finetune_8xb4_20e_{domain}.py # ★研究用：ドメイン毎の学習 config
│                                   #   {domain} = underwater / aerial / videogames /
│                                   #   microscopic / documents / electromagnetic / real_world
│                                   #   （smoke, cat は動作確認用の初期テスト config）
│
├── tools/
│   ├── train.py                    # 学習エントリポイント（上流）
│   └── test.py                     # 評価エントリポイント（上流）
│
├── demo/
│   └── image_demo.py               # 推論・可視化スクリプト（上流）
│
├── experiments/                    # ★研究用：実験ごとのディレクトリを置く
│                                   #   exp_NNN/design.md（実験設計書・承認ゲート）
│                                   #   exp_NNN/{domain}_work_dir（学習出力：ckpt, ログ）
│                                   #   （現状は .gitkeep のみ）
│
├── papers/                         # ★研究用：論文 PDF・要約
│   ├── MM-GroudingDINO.pdf
│   └── ZiRaGroundingDINO.pdf
│
├── setup_and_code_analysis/        # ★研究用：環境構築とコード解析のメモ
│   ├── setup/README.md
│   └── code_analysis/README.md
│
├── visualize_prediction.ipynb      # ★研究用：予測と正解の比較可視化ノートブック
│
├── mmdet/                          # mmdetection 本体（上流。editable install）
└── outputs/                        # 推論スクリプトの既定出力先（vis/ と preds/）
```
★印が、本研究プロジェクト固有のディレクトリ・ファイル。それ以外は上流 mmdetection 由来。

- 学習出力（work_dir）は `experiments/exp_NNN/` 配下に置く方針．
- ドメイン統合済みのデータセットは，リポジトリ外の `/workspace/kouyou/datasets/rf100_domain/{domain}/<train|valid>/` にある．

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
mmdetection は mmengine を介した**config駆動の構成**を採用している，
エントリポイントにハードコードはなく，`tools/train.py` は config の辞書から `Runner` を組み立てる．

- **config は `_base_` で継承される．**
各.  `mm_grounding_dino/*_<domain>.py` は `_base_ = 'grounding_dino_swin-t_pretrain_obj365.py'` を設定し，差分（データセット，クラス，スケジュール）のみを上書きする．
`_base_/` に共通の datasets/models/schedules/runtime が存在する．

- **ドメイン毎の　config　パターン**：



### プロジェクト固有の注意点（上流ドキュメントには無い）

- **テキストトークン数の上限．** MM-Grounding DINO の既定は `max_text_len=256`．クラス数の多い
  ドメイン（`videogames` 87、`real_world` 405）はこれを超えて失敗する．本プロジェクトでの対処は
  config で値を上げること：
  `model=dict(bbox_head=dict(num_classes=..., contrastive_cfg=dict(max_text_len=512, ...)))`．

- ドメイン統合データはリポジトリ外の
  `/workspace/kouyou/datasets/rf100_domain/<domain>/<train|valid>/` にある．

## コマンド

リポジトリのルートで実行する。`<domain>` は
`underwater | aerial | videogames | microscopic | documents | electromagnetic | real_world`．
ただし，`real_world`は基本的に使用しない．

### 学習

```bash
# 学習（1 GPU）
python tools/train.py configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_<domain>.py --work-dir ./experiments/exp_NNN/<domain>_work_dir
```

```bash
# 学習（4 GPU・分散）
bash tools/dist_train.sh configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_<domain>.py 4 --work-dir ./experiments/exp_NNN/<domain>_work_dir
```


### 推論・可視化
指定した画像に対して推論し，予測結果（バウンディングボックス）を可視化する．
Grounding DINO はテキスト（クラス名）を `--texts` で与える必要がある．

**手順 1: 対象ドメインのクラス名を、ピリオド区切りの文字列として取得する**

```
python3 -c "
import json
with open('/workspace/kouyou/datasets/rf100_domain/<ドメイン名>/valid/_annotations.coco.json') as f:
    cats = json.load(f)['categories']
print(' . '.join(c['name'] for c in sorted(cats, key=lambda x: x['id'])) + ' .')
"
```

**手順 2: 取得したクラス名を `--texts` に渡して推論・可視化する**
```bash
python demo/image_demo.py \
  /workspace/kouyou/datasets/rf100_domain/<ドメイン名>/valid/<画像ファイル名>.jpg \
  configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_{domain}.py \
  --weights <work_dir>/best_coco_bbox_mAP_epoch_<エポック数>.pth \
  --texts "<手順1で取得したクラス名>" \
  --custom-entities \
  --out-dir ./experiments/exp_NNN/<ドメイン名>_vis \
  --pred-score-thr 0.3
```
