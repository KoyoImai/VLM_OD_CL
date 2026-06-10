# [構築手順](https://github.com/KoyoImai/VLM_OD_CL/tree/main/setup_and_code_analysis/setup)
事前学習済みVision Language ModelのMM-Grouding DINOをも対象に，学習と評価を動かす環境を整えることを目的とします．
Dockerコンテナ内でのライブラリのバージョン修正，事前学習済み重みを用いたCOCOのゼロショットやcat_datasetを用いたファインチューニング，物体検出結果の可視化などのデモも行います．

# [コード分析](https://github.com/KoyoImai/VLM_OD_CL/tree/main/setup_and_code_analysis/code_analysis)
```
mmdetection/
├── mmdet/                      ★ライブラリ本体（pip install -e . でインストールされる中核）
│   ├── apis/                    推論用の高レベルAPI（inference_detector, DetInferencer等）
│   ├── configs/                 Python形式の純粋なconfig定義（新しいconfig記法）
│   ├── datasets/                データセットクラス（CocoDataset等）とデータ変換（transforms）
│   ├── engine/                  学習・実行を支えるフック、オプティマイザ、スケジューラ等
│   ├── evaluation/              評価指標（COCO mAP等のmetric）
│   ├── models/                  ★モデルの全構成要素（下に詳細）
│   ├── structures/              データ構造（検出結果、bbox、マスク等の入れ物）
│   ├── testing/                 テスト用ユーティリティ
│   ├── utils/                   各種ユーティリティ（setup_env等。train.pyが使う関数群）
│   └── visualization/           検出結果の可視化（描画）
│
├── configs/                    ★設定ファイル群（モデル×データセットごとのレシピ）
│   ├── _base_/                  継承の土台（datasets, models, schedules, default_runtime）
│   ├── grounding_dino/          オリジナル系 Grounding DINO のconfig
│   ├── mm_grounding_dino/       ★あなたが使用中のMM-Grounding DINOのconfig群
│   ├── glip/                    GLIP のconfig（当初目標に含まれる手法）
│   ├── dino/                    DINO（MM-Grounding DINOの基盤となる検出器）のconfig
│   └── （他100以上の手法別ディレクトリ）
│
├── tools/                      ★実行スクリプト群
│   ├── train.py                 ★単一GPU学習（今分析中のファイル）
│   ├── test.py                  単一GPU評価（COCOゼロショット評価で使用）
│   ├── dist_train.sh            マルチGPU分散学習（複数GPU学習で使用）
│   ├── dist_test.sh             マルチGPU分散評価
│   ├── slurm_train.sh / slurm_test.sh   SLURMクラスタ用
│   ├── dataset_converters/      ★データ形式変換スクリプト（Roboflow対応で参考になる）
│   ├── model_converters/        モデル重みの変換
│   ├── analysis_tools/          学習ログ解析、性能分析
│   ├── deployment/              モデルのデプロイ（ONNX等）
│   └── misc/                    その他補助スクリプト
│
├── demo/                       デモ用スクリプトと画像（image_demo.py, demo.jpg等）
├── docker/                     Docker関連ファイル
├── docs/                       ドキュメント（en / zh_cn）
├── projects/                   実験的・コミュニティ提供の拡張プロジェクト
├── requirements/               依存パッケージ定義（multimodal.txt等。環境構築で使用）
├── tests/                      ユニットテスト
├── resources/                  README用画像等のリソース
│
├── setup.py                    パッケージインストール定義（pip install -e . が読む）
├── requirements.txt            基本依存パッケージ
├── model-index.yml             全モデルのインデックス
├── dataset-index.yml           データセットのインデックス
├── README.md / README_zh-CN.md ドキュメント
└── .gitignore                  Git除外設定（*.pth, data/ 等を除外。GitHub登録時に確認済み）
```