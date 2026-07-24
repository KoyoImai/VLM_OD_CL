# exp_024: リプレイ無し逐次ファインチューニング（条件A＝全モジュール）

- 作成日: 2026-07-24
- 状態: **承認待ち（未承認・未実行）**。承認前に実験は実行しない（行動原理3）。
- 元となる実験ノート: [[../../experiment_notes/note04]]（実験1＝比較ベースラインの整備）。
- 上位計画: [[../../experiment_notes/experiments_plan]] 実験1 ／ [[../replay_experiment_plan]] 計画4。
- 実行環境: **MPRG GPU クラスタ**（運用は [[../../README_cluster]]、移行検証は [[../exp_023.5/design]]）。
- **着手条件**: [[../exp_023.5/design]] の Tier2（ゼロショット→1エポック学習→評価）が完了し、クラスタで学習・評価が端まで通ることを確認してから実装・実行に入る。exp_023.5 が通らない場合、クラスタ実行という前提自体が成立しないため、本実験の設計も見直しが必要になる。
- **環境の役割分担**: 本環境（GPU4枚）は exp_023 の完遂（`condA_replay` の3ドメイン、DitHub＋リプレイのデバッグ後の実行）に使用し、**本実験はクラスタで実行**する。両者を並行させて全体の実行時間を短縮する。

---

## 0. 位置付け・目的・仮説

- **位置付け**: 実験1（比較ベースラインの整備）のうち、exp_023 で埋まらない残りの一部。exp_023 が「リプレイ**有り**」を揃えたのに対し、本実験は同一条件の「リプレイ**無し**」を取る。
- **目的**: 全モジュール学習（条件A）で、リプレイを付けない素朴な逐次ファインチューニングを実行し、**リプレイの有無だけが違う対照**を作る。これにより exp_023 の `fullft_replay`（条件A＋リプレイ）と1対1で比較でき、リプレイが忘却抑制に効くかを分離して見られる。
- **仮説**: リプレイ無しの逐次学習では、過去ドメインおよび ZCOCO の性能が大きく低下する（exp_005/006 で確認済みの忘却が、新 regime でも同様に生じる）。適応 mAP は exp_023 のリプレイ有りと大きくは変わらない（リプレイは主に保持側に効く）。**判定は実測値で行う**。

## 1. 手法の定義（1条件のみ）

| 項目 | 内容 |
|---|---|
| 学習範囲 | **条件A＝全モジュール学習**（backbone / language_model を含む全て。`paramwise_cfg` は backbone・language が lr_mult=0.1、他 1.0） |
| リプレイ | **無し**（バッファを一切混ぜない。現在ドメインのみ） |
| 逐次 | あり（3ドメインを順に学習し、重みのみ引き継ぐ） |

- **条件A/B の呼称は note04 の定義に従う**: 条件A＝全モジュール、条件B＝特徴抽出+融合。
  注意: exp_023 の `condA_replay_*.py` はファイル名に反して**条件B**（特徴抽出+融合）を実装している。本実験では取り違えを避けるため、条件A の config を `fullft_replayfree_*` と命名する。

## 2. ドメイン順序

- **underwater → electromagnetic → videogames**（3ドメイン・単一順序）。exp_023 と同一。

## 3. テキスト入力（負例サンプリング）

- exp_023 と同一の新 regime に揃える: 全ドメインを **OD-ODVG 形式**に変換済みのものを使い、`RandomSamplingNegPos`（正例＋同一ドメイン内からサンプリングした負例）で学習する。負例はドメイン別 label_map に閉じる。
- パラメータ（事前学習の設定を引き継ぐ）: **num_sample_negative=85 / full_sampling_prob=0.5 / max_tokens=256**。
- `num_classes` は全手法で **256**（pretrain 継承・非上書き）。DN 経路の `label_embedding` を θ0 から正しくロードさせ、手法間で初期化条件を揃えるため（exp_023 design §1.1(b) と同一の判断）。

## 4. バッチ構成

- **現在ドメインのみ 4/GPU（`batch_size=4`）**。note04 の「バッチの内訳（リプレイフリー）：現在ドメイン4」に一致。
- exp_023 のリプレイ有りは同じ現在4にバッファ2を足して batch=6 としているため、**現在ドメインの露出は両者で一致**し、差分はリプレイの有無だけになる。
- 4GPU 分散（実効バッチ 16）。

## 5. 逐次学習スケジュール

- 各ドメイン: **epoch=20、milestones=[15]（gamma=0.1）、lr=1e-4、AdamW（wd=1e-4）**。全ドメイン共通。
- **各ドメインで LR スケジュール・optimizer 内部状態をリセット**。重みのみ前ドメインの last を `load_from` で引き継ぐ（t=1 は θ0）。
- 適応 mAP は **last**（epoch_20）を採用。
- **seed=0** 固定。

## 6. 評価・判定基準

- 各ドメイン学習後に測定: (a) 現ドメインの適応 mAP、(b) COCO2017 ゼロショット検出（ZCOCO、`eval_base_coco.py` 経由）、(c) **過去に学習した各ドメインごと**の検出性能。
- **忘却はドメインごとに個別に計算**し、全ドメイン平均は取らない（note04・replay_experiment_plan §0 の決定）。
- 本実験は比較ベースラインの整備が目的のため、合否判定ではなく**実測値の収集**を行う。判定（適応の可否など）はユーザーが実験ノートで行う。

## 7. 実装（承認後に着手、実行前に検証）

- **新規 config**（既存ファイルは無変更）:
  - `experiments/exp_024/configs/fullft_replayfree_base.py` — 事前学習 config を継承。θ0 の `load_from`、lr=1e-4、20epoch/milestone[15]、seed=0、`custom_imports`（`exp023_np_compat`：`RandomSamplingNegPos` が使う `np.long` の互換シム）、in-training val 無効。
  - `experiments/exp_024/configs/fullft_replayfree_{underwater,electromagnetic,videogames}.py` — 現在ドメインの ODVG データセットのみを `train_dataloader`（batch_size=4）に設定。`ConcatDataset`／`MultiSourceSampler` は使わない。
  - 評価 config は exp_023 の `eval_{domain}.py` / `eval_base_coco.py` を流用（無変更）。
- **backbone の `init_cfg` を無効化する**（`model.backbone.init_cfg=None`。2026-07-24 決定）。継承元の事前学習 config は ImageNet 事前学習 Swin-T（`swin_tiny_patch4_window7_224.pth`, 110MB）を `init_weights()` で読み込むが、その直後の `load_from`（t=1 は θ0、t≥2 は前ドメイン last）が backbone の **187 パラメータを全て上書き**するため、最終的な重みには影響しない。無効化により毎ジョブの不要なダウンロード・読み込みが省ける。
  - **検証済み（2026-07-24）**: `init_cfg` 有効・無効の2通りでモデルを構築し、`init_weights()` → θ0 ロードまで実行して全 **908 テンソルを比較 → 完全一致（差分 0）**。前ドメイン ckpt（`epoch_20.pth`）も backbone **187/187** を被覆することを確認済み。
- **逐次ドライバ**: `experiments/exp_024/run_sequential_fullft_replayfree.sh`（t=1→2→3、前ドメイン last を `load_from`、各ドメイン後に 現ドメイン＋過去ドメイン＋ZCOCO を評価）。
- **クラスタ実行**: `experiments/exp_024/{sbatch.sh, train_val.sh}`（案A の器／中身。exp_023.5 で検証済みの bind 構造を踏襲）。
- **実装検証（実行前）**: (a) 負例が同一ドメインに閉じるか、(b) リプレイ無しで `batch_size=4` が意図通りか、(c) 全モジュールに勾配が流れるか（条件A の確認）、(d) θ0 からの重みロードで欠損キーが無いか。

## 8. この実験で言えないこと（限界）

- 単一順序のみのため、順序依存の頑健性は言えない。
- テキスト方式が新 regime のため、旧 regime の過去実験（exp_003/005/006、全クラス連結）とは直接比較できない。
- 提案手法（蒸留）の有効性は扱わない（実験2）。

## 9. 成果物

- `experiments/exp_024/{domain}_work_dir/`（ckpt・ログ）、評価出力、`experiments/exp_024/results/`（事実記録、行動原理8）。
- クラスタ実行時の成果物はホーム側に残り、本環境へ回収する（[[../../README_cluster]] §7）。
