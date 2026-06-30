# VLM_OD_CL
事前学習済みVision Language Modelを用いた物体検出の継続学習

## ディレクトリ構成
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
└── mmdet/                          # mmdetection 本体（上流。editable install）
```

## Roboflow100データセット」の準備
Roboflow100データセットのgithubリポジトリをクローンする．
```
git clone https://github.com/roboflow-ai/roboflow-100-benchmark.git
cd roboflow-100-benchmark
git submodule update --init --recursive
```
`roboflow`ライブラリをインストールする．
```
pip install roboflow
```
`./scripts/download_datasets.sh`を使用してデータセットをダウンロードする．
保存先は，`/workspace/kouyou/datasets/rf100`とする．
```
chmod 770 ./scripts/download_datasets.sh
./scripts/download_datasets.sh -l <保存したいディレクトリのパス>
```

## Roboflow100データセットでの学習・評価プログラムの実装
Roboflow100データセットを使用した，ドメイン増加の物体検出継続学習を実装する．

### 1データセットでの学習（`rf100/smoke-uvylj`を使用）
まず初めに，Roboflowデータセットに含まれるデータセットの1つを使用して，学習評価プログラムを実装する．
使用するデータセットは`rf100/smoke-uvylj`とする．

#### `configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_smoke.py`の作成
`configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_cat.py`を参考に，`configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_smoke.py`を作成する．

また，それに加えて，Roboflow100データセットに含まれる`_annotations.coco.json`の修正も行う必要がある．
`_annotations.coco.json`には，ダミーカテゴリが含まれるためカテゴリ数（クラス数）が一致せず学習時にエラーが発生する．
そのため，`_annotations.coco_fixed.json`というファイルに内容をコピーし修正，修正した`_annotations.coco_fixed.json`を使用して学習と評価を動かす．

```
Accumulating evaluation results...
DONE (t=0.39s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.758
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.935
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.881
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.290
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.790
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.875
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.894
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.894
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.540
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.918
06/12 06:16:11 - mmengine - INFO - bbox_mAP_copypaste: 0.758 0.935 0.881 -1.000 0.290 0.790
06/12 06:16:11 - mmengine - INFO - Epoch(val) [20][148/148]    coco/bbox_mAP: 0.7580  coco/bbox_mAP_50: 0.9350  coco/bbox_mAP_75: 0.8810  coco/bbox_mAP_s: -1.0000  coco/bbox_mAP_m: 0.2900  coco/bbox_mAP_l: 0.7900  data_time: 0.0033  time: 0.0985
```


## データセット（RF100 ドメイン統合版）

Roboflow100（RF100）の100データセットを，7つのドメインに分類し，ドメインごとに統合したもの．
各ドメイン内で同一クラスは1つに統合している．
元データは640×640にリサイズ済みのCOCO形式．
統合データは `/workspace/kouyou/datasets/rf100_domain/<ドメイン名>/<train|valid>/` に配置．

| ドメイン | train 画像数 | valid 画像数 | クラス数 | train アノテ数 | valid アノテ数 | config ファイル |
|---|---:|---:|---:|---:|---:|---|
| underwater | 12,633 | 3,576 | 28 | 64,362 | 16,927 | `configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_underwater.py` |
| aerial | 6,643 | 1,940 | 22 | 35,840 | 10,283 | `configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_aerial.py` |
| videogames | 8,233 | 2,219 | 87 | 16,757 | 4,613 | `configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_videogames.py` |
| microscopic | 9,576 | 2,529 | 28 | 74,208 | 16,422 | `configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_microscopic.py` |
| documents | 17,866 | 4,597 | 59 | 126,166 | 26,159 | `configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_documents.py` |
| electromagnetic | 25,398 | 7,314 | 39 | 85,089 | 31,054 | `configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_electromagnetic.py` |
| real world | 78,747 | 21,537 | 405 | 572,953 | 126,656 | `configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_real_world.py` |
| **合計** | **159,096** | **43,712** | **668** | **975,375** | **232,114** | — |

### 補足
 - videogames と real world はクラス数が多く，MM-Grouding DINOが対応可能なテキストトークン数（256）に収まりきらないため，学習と評価がうまくできない可能性がある． `max_text_len=512`に変更して学習を行った場合，videogamesでも学習がうまくいく．

## 各ドメインのデータセットで学習

ドメインごとに分割・統合したデータセットで学習を実行する．
学習は1gpu，あるいは複数gpuでの並列実行が可能．
```
# 1gpuでの学習
python tools/train.py configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_<ドメイン>.py --work-dir <確認用work-dir>
```
```
# 4gpuでの学習
bash tools/dist_train.sh configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_<ドメイン>.py 4 --work-dir <確認用work-dir>
```

### underwater　ドメイン
```
# 1gpuでの学習
python tools/train.py configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_underwater.py --work-dir underwater_work_dir

# 4gpuでの学習
bash tools/dist_train.sh configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_underwater.py 4 --work-dir underwater_work_dir
```

### aerialドメイン
```
# 1gpuでの学習
python tools/train.py configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_aerial.py --work-dir aerial_work_dir

# 4gpuでの学習
bash tools/dist_train.sh configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_aerial.py 4 --work-dir aerial_work_dir
```

### videogamesドメイン
```
# 1gpuでの学習
python tools/train.py configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_videogames.py --work-dir videogames_work_dir

# 4gpuでの学習
bash tools/dist_train.sh configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_videogames.py 4 --work-dir videogames_work_dir
```

### microscopicドメイン
```
# 1gpuでの学習
python tools/train.py configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_microscopic.py --work-dir microscopic_work_dir

# 4gpuでの学習
bash tools/dist_train.sh configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_microscopic.py 4 --work-dir microscopic_work_dir
```

### documentsドメイン
```
# 1gpuでの学習
python tools/train.py configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_documents.py --work-dir documents_work_dir

# 4gpuでの学習
bash tools/dist_train.sh configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_documents.py 4 --work-dir documents_work_dir
```

### electromagneticドメイン
```
# 1gpuでの学習
python tools/train.py configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_electromagnetic.py --work-dir electromagnetic_work_dir

# 4gpuでの学習
bash tools/dist_train.sh configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_electromagnetic.py 4 --work-dir electromagnetic_work_dir
```

### real worldドメイン
```
# 1gpuでの学習
python tools/train.py configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_real_world.py --work-dir real_world_work_dir

# 4gpuでの学習
bash tools/dist_train.sh configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_real_world.py 4 --work-dir real_world_work_dir
```






