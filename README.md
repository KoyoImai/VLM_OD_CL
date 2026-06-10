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
