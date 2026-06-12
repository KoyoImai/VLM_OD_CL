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


