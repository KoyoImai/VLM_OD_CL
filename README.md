# VLM_OD_CL
事前学習済みVision Language Modelを用いた物体検出の継続学習

## [構築手順 & コード分析](https://github.com/KoyoImai/VLM_OD_CL/tree/main/setup_and_code_analysis)
```
mmdetection/
├── mmdet/
├── configs/        ：設定ファイル群（モデルxデータセット毎のレシピ）
│   ├── _base_/     ：継承の土台（datasets，models，schedules）
├── tools/          ：実行スクリプト群
│   ├── train.py    ：単一GPU学習
├── demo/
├── docker/
├── docs/
├── projects/
├── requirements/
├── tests/
├── resources/
├── setup.py/
├── requirements.txt
├── model-index.yml
├── dataset-index.yml
├── README.md / README_zh-CN.md
└── .gitignore
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