# exp_023.5 design — クラスタ移行のデバッグ（本番前の挙動確認）

## 位置付け
exp_024／exp_025 をクラスタで本格実行する前に、**クラスタ側で本環境と同じ挙動が再現されるか**を確認するデバッグ実験。
新しい研究上の問いを立てる実験ではなく、`README_cluster.md` の運用（`.sif`・`--bind` パス間接化・local_cache ステージング・config 解決・`dist_train`／`dist_test`・ZCOCO）が意図通り動くことの実装検証（役割1）。

## 目的
1. クラスタのプラミング（環境・バインド・パス解決・分散学習・評価）が端から端まで動くことを確認する。
2. 既知 good の exp_023 config を流用し、クラスタ固有の問題（パス・バインド・環境差）を、config やモデルの問題から切り分ける。
3. 本環境の既知値と照合し、クラスタが同等の結果を再現することを確認する。

## 切り分けの方針
新規 config を一切作らず、**本環境で完了済み・値が分かっている** `fullft_replay_underwater.py`（および 1 epoch 版 `fullft_replay_underwater_debug.py`）をそのままクラスタで走らせる。
これにより、失敗が出た場合に「クラスタ／バインド／環境の問題」であって「config やモデルの問題ではない」と断定できる。

## 使用する既存 config（すべて無変更・流用）
- 学習（プラミング）: `experiments/exp_023/configs/fullft_replay_underwater_debug.py`（1 epoch）
- 学習（パリティ）: `experiments/exp_023/configs/fullft_replay_underwater.py`（20 epoch。本環境の既知値と対応）
- 評価（適応）: `experiments/exp_023/configs/eval_underwater.py`
- 評価（ZCOCO / COCO ゼロショット）: `configs/mm_grounding_dino/eval_base_coco.py`（ゼロショットは test.py の ckpt 引数に θ0 を渡すだけ）

## 前提: θ0 のオフライン可用性
base config の `load_from` は openmmlab の URL で、計算ノードがインターネット非接続だと取得に失敗する（学習・ゼロショット評価の両方に影響）。
対処は config 無変更で行う: θ0 の `.pth` をホームに置き、(a) torch hub のキャッシュ（URL キーで参照される場所）に配置してURLロードをオフライン解決させる、または (b) `--cfg-options load_from=<path>` / test.py の ckpt 引数でパスを直接渡す。Tier 1 でこのロードが通ることを先に確認する。

## 参照アンカー（本環境の実測値）
- **θ0 ゼロショット COCO: mAP 0.504**（`experiments/exp_001/coco_work_dir` 実測、`exp_001/outputs/notes.md`）。学習不要・θ0同一・評価決定的なので、**1エポック方針でも数値パリティの本命アンカー**。
- underwater fullft_replay（**20 epoch**）: 適応 mAP 0.344 / ZCOCO 0.412（`experiments/exp_023/eval_after_underwater/`）。**これは20エポック到達値であり、下記 1 epoch 実行とは比較しない**（参考として記載）。
- **1エポックの本環境アンカーは無い**: `debug_fullft_replay_underwater_work_dir` の実行は iter 840/2273 で停止・checkpoint 未保存。iter 数（2273）も world_size 依存で 20epoch フル実行（790, 4GPU）と食い違うため、iter 数・1epoch mAP はアンカーにしない。

## 手順

### Tier 1: プラミング確認（インタラクティブジョブ, 数分, `*_interactive` パーティション ≤2h）
`.sif` を起動し、以下をコンテナ内で確認する。
1. `import mmdet, mmcv` が通り、`torch.cuda.device_count()==4`。
2. `--bind` したパスが見える: `/workspace/kouyou/mmdetection`, `/workspace/kouyou/datasets/rf100_domain`, `/workspace/kouyou/datasets/o365v1_stage`（入れ子 bind）, `/workspace/kouyou/datasets/coco2017`。
3. クラスタ側 COCO val の件数が本環境と一致: `instances_val2017.json` = 5000画像 / 80カテゴリ / 36781アノテーション。
4. `Config.fromfile('.../fullft_replay_underwater.py')` がビルドでき、dataloader の各ソースのパスが bind 下で解決する（ファイル存在確認）。
5. dataloader が 1 バッチ yield できる（現ドメイン4＋参照2 の混合が組める）。
6. **θ0 がオフラインでロードできる**（上記「前提」の (a) or (b)）。ここが通らないと以降すべて動かない。

### Tier 2: 評価・学習（バッチジョブ, 案A の sbatch.sh / train_val.sh）
0. **COCO ゼロショット（学習不要・最速の評価パリティ）**: θ0 を `eval_base_coco.py` で評価 → 本環境の **0.504** と照合。θ0ロード・ZCOCOバインド（`/dataset01/MSCOCO`）・評価経路を、学習コストゼロで一気に検証する。
1. **プラミング学習（1 epoch）**: `fullft_replay_underwater_debug.py`（1 epoch）を実行 → クラッシュせず完走し、**epoch 末で checkpoint が保存される**ことを確認。iter 数は world_size 依存（本環境debug=2273 は参考、クラスタ4GPUでは異なる）。損失の桁が本環境序盤ログ（loss 8〜10 台）と大きくずれないか程度のサニティを見る。
2. **適応評価が回る**: 保存 ckpt を `eval_underwater.py` と `eval_base_coco.py` で評価 → 数値が出力される（適応・ZCOCO 両経路が通る）。値の一致確認には使わない（1epoch mAP はノイズが大きく、本環境アンカーも無い）。
3. **1エポックの環境パリティ（後日）**: 現在は本環境で別学習が稼働中のため、GPU が空いた後に**本環境でも同じ 1 epoch（`fullft_replay_underwater_debug.py`）を完走・評価**し、クラスタの 1 epoch 結果と照合する。これにより本環境↔クラスタの環境パリティを確認する。
   - 公平な比較の条件: 本環境アンカーは**クラスタと同じ world_size（4GPU）**で取る（1epoch値は実効バッチ=batch×world_size 依存のため）。`deterministic=False` ゆえ完全一致ではなく「近い値か」を見る。
   - それまでの間の数値パリティは Tier 2-0（COCO ゼロショット 0.504、学習不要）で担保する。

## 確認の考え方（研究判定ではなくエンジニアリング・サニティ）
- Tier 1・Tier 2-1・2-2 は「エラーなく端まで通るか」の二値確認。
- Tier 2-0（COCO ゼロショット）は学習を挟まないので、本環境の **0.504** と**ほぼ一致**するはず（θ0 が同一・評価が決定的）。乖離があれば θ0 ロード差・COCO データ差・評価設定差を疑う。これが数値パリティの本命。
- Tier 2-1/2-2（1 epoch 学習・評価）は「エラーなく端まで通り、checkpoint と評価数値が出るか」の二値確認。数値の一致は見ない（本環境に1epochアンカーが無く、`deterministic=False` で1epoch値のノイズも大きい）。

## コスト（概算, 4×A6000 Ada 1ノード）
- Tier 1: 数分（インタラクティブ）
- Tier 2-0（COCO ゼロショット, 学習なし・val 5000枚評価のみ）: 数分
- Tier 2-1（1 epoch 学習）＋2-2（評価）: 短時間（20 epoch フルは行わない）

## 成果物の置き場所
- 学習出力: `experiments/exp_023.5/debug_underwater_work_dir/`（プラミング）, `experiments/exp_023.5/parity_underwater_work_dir/`（パリティ）
- 評価出力: `experiments/exp_023.5/eval_*/`
- クラスタ実行スクリプト（案A・作成済み）: `experiments/exp_023.5/{sbatch.sh, train_val.sh, tier1_plumbing.sh}`
  - `sbatch.sh`（器）: Slurm＋rf100ステージング＋Singularity起動。`RUN_STAGE=all|zeroshot|train|eval` で範囲指定。
  - `train_val.sh`（中身）: Tier2-0 ゼロショット／2-1 1ep学習／2-2 評価。既存 config を無変更で呼ぶ。
  - `tier1_plumbing.sh`: Tier1 の import・bind・COCO件数・config解決・θ0 存在をインタラクティブに確認。

## 制約の遵守
- 既存プログラム・config は無変更。exp_023.5 が新規に足すのは design.md・クラスタ実行スクリプト・検証メモのみ。
- `*_work_dir` 配下は編集しない。
