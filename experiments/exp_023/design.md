# exp_023: リプレイ付き比較手法の逐次学習（リプレイまでのスコープ）

- 作成日: 2026-07-22
- 状態: **承認待ち（未承認・未実行）**。承認前に実験は実行しない（行動原理3）。
- 元となる決定: [[../replay_experiment_plan]]（「リプレイまで」＝計画0〜4）。詳細な決定日付は同ファイル参照。
- 注: 実験ノート `experiment_notes/note04.md` は現在空。下記「位置付け・目的・仮説」は本会話の決定から仮に起こしたもので、**ユーザーの確認・修正が必要**（研究フレーミングの文章はユーザーが記述）。

---

## 0. 位置付け・目的・仮説（要ユーザー確認）

- **位置付け**: リプレイを主機構とする継続学習の中心実験に向けた、**比較相手（ベースライン群）の整備**。提案手法（蒸留ベースのリプレイ学習）は本スコープ外・後段。
- **目的**: 既存手法・単純手法に**リプレイを付けた**版を、事前学習から遠いドメインが増える逐次設定で学習・評価し、後段の提案手法と比較できる土台を作る。
- **仮説（暫定）**: 本スコープは主に整備であり、強い仮説検証ではない。判定は各手法の適応・ZCOCO・過去ドメイン保持の実測値で行う。

## 1. 比較手法（4つ、すべてリプレイ付き）

| 手法 | 学習範囲 | リプレイ |
|---|---|---|
| ZiRa ＋ リプレイ | ZiRa 固有構造 | あり |
| DitHub ＋ リプレイ | DitHub 固有構造（LoRA 辞書） | あり（下記の特則） |
| full finetuning ＋ リプレイ（＝素朴リプレイ） | 全モジュール | あり |
| 条件A ＋ リプレイ | extraction + fusion（downstream 凍結） | あり |

- **full finetuning ＋ リプレイ と 素朴リプレイ は同一**（全モジュール学習 ＋ 素朴なバッファ混合）。

- **DitHub の特則**: 参照データにのみ含まれるクラス（学習ドメインに無い）は**凍結モデルのみ**を通し、他ドメインと共通で LoRA モジュールがあるクラスのみ **LoRA** を通す（DitHub の名前ベースのモジュール選択に従う）。
- joint（全ドメイン同時学習・オラクル上限）は**仮採用**——余裕があれば実行。

### 1.1 追記（2026-07-22）: リプレイフリー版の追加・num_classes・論文準拠ハイパラ

**(a) ZiRa/DitHub はリプレイフリー版も動かす.** 上表の「＋リプレイ」4手法に加え、ZiRa と DitHub は**リプレイ無し**版も比較に含める（同一手法での「リプレイ有無」比較を可能にするため）。比較セルは full FT＋replay ／ ZiRa＋replay ／ DitHub＋replay ／ 条件A＋replay ／ **ZiRa replay-free** ／ **DitHub replay-free** の計6つ。

**(b) num_classes は全手法で 256 に統一（label_embedding の形状不一致回避）.** MM-GDINO の `num_classes` は denoising 経路の `dn_query_generator.label_embedding`（[num_classes,256]）のサイズを決める。finetune テンプレートの `num_classes=len(class_name)`（例28）では θ0 の [256,256] とサイズ不一致でロードされず**ランダム初期化**される（過去 exp_020/021/022 はこの状態。ZiRa/DitHub では凍結もされランダム固定）。label_embedding は学習時専用（推論では `if self.training:` ガードで不使用）なので**評価 mAP には無影響**だが、学習 loss 絶対値が変わり手法間で DN 初期化条件が食い違う交絡になる。対処として exp_023 は全 config を **num_classes=256**（pretrain 継承・非上書き）とし θ0 から正しくロードさせる。分類はテキスト対比なので num_classes をドメインクラス数に合わせる必要はない。

**(c) ZiRa/DitHub は論文（公式コード）準拠のスケジュール・学習率で動かす（リプレイ有無に関わらず）.** §5 の共通スケジュール（20 epoch・lr 1e-4）は full FT＋replay・条件A に適用し、ZiRa/DitHub は各論文値を用いる（出典: `papers/{ZiRa,DitHub}_implementation_notes.md`、既存 official config）:
  - ZiRa: IterBasedTrainLoop **2000 iter**、AdamW **lr 1e-3 / wd 1e-4**、iter800 で lr×0.1、LLRB のみ lr×0.2、λ=0.1。
  - DitHub: IterBasedTrainLoop **3000 iter**、AdamW **lr 1e-3 / wd 1e-2**、iter1200 で lr×0.1、warmup1500＋specialization1500（iter1500 切替）。
  - dn は有効のまま（他基準線と条件を揃え、スケジュール変数のみ論文準拠にする。exp_020/021 official 版と同じ判断。上記(b)により label_embedding は正しくロードされる）。ckpt は last に統一（§5）。

**(d) 「リプレイの有無以外は可能な限り揃える」（batch 方針）.** ZiRa/DitHub の replay-free と ＋replay で現在ドメインの学習を一致させ、差分をリプレイ（バッファ追加）だけにする。4GPU・現在4/GPU（batch_size=4）を両者共通とし、＋replay 側がそこにバッファ2を足す（batch→6）。iter 数は論文値（batch2 前提）だが実効バッチが大きい分サンプル露出は論文より多い（batch 統一を優先した帰結）。

## 2. ドメイン順序

- **underwater → electromagnetic → videogames**（3ドメイン・単一順序）。

## 3. テキスト入力（負例サンプリング）

- 全ドメイン・参照を **OD-ODVG 形式**に変換し、`RandomSamplingNegPos`（正例＋同一ドメインからサンプリングした負例）で学習。負例はドメインごとの label_map に閉じる（同一ドメイン限定）。
- パラメータ（既存＝事前学習の設定を引き継ぐ）: **num_sample_negative=85 / full_sampling_prob=0.5 / max_tokens=256**。
- 帰結: キャプションは常に 256 トークン以内に収まり、videogames（87クラス）も収まる（`max_text_len=512` 不要、conastive_cfg も 256 と整合）。**現行 finetune（全クラス連結）とは学習条件が変わる**ため、既存 exp_020/021 はそのままは使えない。

## 4. バッファと混合

- **バッファサイズ（ハイパラ、初期）**: 参照データ 1,000、各学習済みドメイン 500。
- **バッファ充填**: 各ドメインの train から**ランダム**に保存。**全比較手法で共通**（同一サンプル）。選択はモデル非依存のため、**学習とは別プロセスの独立スクリプトで学習前に一括事前選択**し固定ファイル化する（学習の RNG に影響させない）。
- **バッファ取り出し**: ランダム。
- **混合比率（ハイパラ、初期）**: 現在ドメイン : (過去＋参照バッファ) ＝ **1 : 0.5**。
- 参照データ: **Objects365v1**（第一・使用予定）／V3Det（第二）。OD 形式。
- 探索範囲（混合比率・バッファサイズ）は**結果を見てから決定**（初期値のみ確定）。

## 5. 逐次学習スケジュール

- 各ドメインの学習設定は**既存の単一ドメイン設定を全ドメインで使用**——epoch=20、milestones=[15]（gamma=0.1）、lr=1e-4。
- **各ドメインで LR スケジュールをリセット、optimizer の内部状態もリセット**。重みのみ前ドメインの last を引き継ぐ。
- 適応 mAP は **last** を採用。
- seed=0 固定。

## 6. 評価・判定基準

- 各ドメイン学習後に測定: (a) 現ドメインの適応 mAP、(b) COCO2017 ゼロショット検出（ZCOCO、`eval_base_coco.py` 経由）、(c) 過去に学習した各ドメインごとの検出性能。
- **忘却は個々のドメインごとに計算**し、**全ドメイン平均は取らない**。
- 本スコープは整備が主目的のため、合否判定ではなく実測値の収集・比較を行う。

## 7. 前提タスク・実装（承認後に着手、実行前に検証）

- 各ドメイン＋参照の OD-ODVG 変換、**ドメイン別 label_map** 作成。
- 参照データ（Objects365v1）取得 ＋ **COCO2017-val とのハッシュ照合**（画像重複が無いことの確認）。
- リプレイバッファ ＋ 混合サンプラーの実装、ZiRa/DitHub へのリプレイ組み込み（DitHub は上記特則）。
- **実装検証**: (a) 同一ドメイン負例が閉じるか、(b) 混合バッチが loss() の `tokens_positive` 経路で正しく処理されるか、(c) 単一ドメイン学習の数値が現行（全クラス連結）とどれだけ変わるか、(d) 勾配疎通。

## 8. この実験で言えないこと（限界）

- 提案手法（蒸留）の有効性はここでは扱わない（後段）。
- 単一順序のみのため、順序依存の頑健性は言えない。
- テキスト方式を変えたため、過去実験（全クラス連結）との直接比較はできない（新 regime で全比較手法を取り直す）。

## 9. 未確定・後段（本 design の対象外）

- 蒸留対象・蒸留データ・蒸留の学習ベース（条件A か full FT か）、t 増加時の配分則（計画5・6）。
- 混合比率・バッファサイズの探索範囲、忘却指標の具体式、計算コスト見積り。

## 10. 成果物

- `experiments/exp_023/{手法}_{ドメイン}_work_dir`、ZCOCO 出力、`experiments/exp_023/results/`（事実記録、行動原理8）。

## 11. 実施記録（データ準備・実装）

実施した手順を時系列で記録（後から何をしたか把握できるように）。詳細な入手手順は [[objects365v1_download]]、実装計画は [[implementation_plan]] を参照。

### 11.1 参照データ Objects365 v1 の取得・解凍（2026-07-22）
- 入手元: OpenDataLab `OpenDataLab/Objects365_v1`（**v2 でなく v1**。no-auth ミラーは v2 のみだったため認証取得）。認証は GitHub 連携アカウントのため **openxlab の AK/SK ログイン**（`openxlab login`）を使用。
- 取得: `openxlab dataset get -r "OpenDataLab/Objects365_v1" -t /workspace/kouyou/datasets/` → `OpenDataLab___Objects365_v1/raw/Objects365_v1.tar.gz`（54G）。
- 解凍: 外側 tar → `o365v1_stage/Objects365_v1/2019-08-02/`（`objects365_train.json` 1.7G ＋ 画像 zip 群）。画像 zip を全解凍 → **train 608,606枚**・val 30,000・test 99,010。
- 環境注意: opendatalab/openxlab は共有パッケージをダウングレードする（urllib3/setuptools 等）。**復元はユーザー側で完了済み**。核心 ML スタック（torch/mmengine/mmdet/mmcv）は無変更。

### 11.2 `coco2odvg.py` に generic モード追加（RF100 対応）
- 既存 `coco2odvg.py` の `-d` は coco/o365v1/o365v2/v3det 専用（各データセットの ID 対応がハードコード）で RF100 ドメインを変換できないため、**入力 JSON の `categories` から動的に対応表を作る `generic` モード**を追加（`dump_generic_label_map` 関数、`generic` 分岐、argparse に `generic`＋`--map-name`）。既存モードは無変更。
- 検証: 全6ドメインで categories 連番・placeholder無し・**config の class_name と順序込み一致**・画像数一致・**bbox の xywh→xyxy 厳密一致**を確認。

### 11.3 OD-ODVG 変換
- 参照: `coco2odvg.py -d o365v1` で `objects365_train.json`（全608,606枚）→ `objects365_train_od.json`（973M、365クラス）＋ `o365v1_label_map.json`。既知欠損3枚スキップ。
- RF100 全6ドメイン: `coco2odvg.py -d generic` で各 train を変換し、**各ドメインディレクトリ内**に `<domain>_train_od.json` ＋ `<domain>_label_map.json` を生成（underwater 28/aerial 22/videogames 87/microscopic 28/documents 59/electromagnetic 39 クラス）。train のみ（評価は val・負例サンプリング不要）。

### 11.4 バッファ選択ツールと事前選択（2026-07-22、seed=0）
- ツール: `tools/select_replay_buffer.py`（**実験横断で再利用可**の汎用ツール）。ODVG から seed 固定でランダムに k 件を選び、(1) ODVG サブセット（学習が読む実体）と (2) manifest（seed・件数・元ファイル・選択画像ファイル名一覧）を出力。2パス方式でメモリ効率化（大きい参照ファイル対応）。
- 事前選択（**seed=0**、学習とは別プロセスのため学習 RNG に影響なし、全手法共通の固定バッファ）: `experiments/exp_023/buffer/` に
  - `reference_o365v1_1000.odvg.json`（参照1,000）
  - `<domain>_500.odvg.json`（6ドメイン各500。exp_023 で未使用のドメイン含め全6ドメイン分を選択）
  - 各 `*.manifest.json` 付き。
- 検証: seed=0 で再実行し選択が完全一致（**再現性**）、manifest に seed・選択ファイル名を記録、ODVG 形式が正しいことを確認。

### 11.5 参照バッファの COCO2017-val 照合（pHash、2026-07-22）
- 目的: 参照1,000枚に COCO2017-val 画像が紛れていない＝リプレイしても ZCOCO のゼロショット性が保たれるか確認。
- 方法: **知覚ハッシュ（pHash、64bit）**を自前実装（scipy DCT。imagehash 不使用で環境無汚染）。参照1,000枚と COCO2017-val 5,000枚の pHash を計算し、各参照の最近傍 COCO-val との Hamming 距離を算出。
- 結果: **距離 ≤5（同一画像の閾値）は 0件**。最小距離10の1件のみ ≤10 だが、二次確認（サイズ・アスペクト比相違、NCC=0.304）で**別画像と確定**。分布は最小10・中央値16。
- 判定: **参照バッファに COCO2017-val の重複なし。ZCOCO 妥当性は保たれる。**

### 11.6 full finetuning + リプレイ の config 部品（2026-07-22）
**方針: 既存プログラムは一切変更せず、新規ファイル追加のみで実装（ユーザー厳守指示）。**

- **numpy 互換シム（既存無変更）**: 既存 `RandomSamplingNegPos` が使う `np.long` は numpy>=1.24 で削除（現環境 1.26）。新規モジュール `exp023_np_compat.py`（リポジトリルート、editable install で import 可）で `np.long=np.int64` を復元し、config の `custom_imports` から読み込む。config 内の生 import は mmengine が lazy-import 構文と誤認して弾くため、`custom_imports` 方式を採用。
- **共通ベース** `experiments/exp_023/configs/fullft_replay_base.py`: 事前学習 config を継承（ODVG 対応モデル num_classes=256、`RandomSamplingNegPos` 入り train_pipeline）。full FT は事前学習の backbone/language lr_mult=0.1（＝全モジュール学習）をそのまま使い、lr=1e-4・20 epoch・milestone[15]・seed=0。in-training val は無効（評価はドライバ）。`load_from`=θ0（t=1 用、t>=2 はドライバが前 last に差し替え）。
- **ドメイン差分 config**（各 ODVGDataset＋負例サンプリング、`MultiSourceSampler` で混合）:
  - `fullft_replay_underwater.py`（t=1）: [現在, 参照] の2ソース、source_ratio [2,1]（現在4・参照2）。
  - `fullft_replay_electromagnetic.py`（t=2）: [現在, 参照, 過去(underwater)] の3ソース、[4,1,1]（現在4・参照1・過去1）。
  - `fullft_replay_videogames.py`（t=3）: 過去2ドメインを入れ子 `ConcatDataset` で1つの「過去プール」に束ね [現在, 参照, 過去プール]、[4,1,1]（現在4・参照1・過去1）。入れ子は metainfo の `cumulative_sizes` 衝突を `ignore_keys=['cumulative_sizes']`（mmdet 既存パラメータ）で回避。
  - **batch_size は全ドメイン 6 に統一**（2026-07-22 決定）。現在ドメインのバッチが全て 4 になり、過去実験（batch4）と整合。
- 現在:バッファ=2:1、バッファ側は「参照＋過去が毎バッチ必ず1件以上」（解釈B。過去は複数なら1プールから一様）。
- **build 検証（3ドメインとも合格）**: config 構造（full FT・スケジュール・比率）、numpy 互換（np.long 復元・負例サンプリング動作）、**負例が同一ドメインに閉じる**（各ソースが自ドメインのクラスでキャプション生成）、**混合比率**（現在:参照:過去＝意図通り、参照・過去が毎バッチ必ず入る）、過去プールが両ドメインから一様に引かれる、を確認。

### 11.7 評価 config・逐次ドライバ・スモークテスト（2026-07-22）
- **評価 config**（各ドメイン valid、既存 finetune 継承・既存無変更）: `eval_{underwater,electromagnetic,videogames}.py`。ODVG 学習済み ckpt に合わせ num_classes=256・max_text_len=256 に上書き。videogames のみ 87クラス=265トークンで256を超えるため `chunked_size=40`（分割推論）。ZCOCO は既存 `eval_base_coco.py`（num_classes=256）を流用。
- **逐次ドライバ** `run_sequential_fullft_replay.sh`（新規）: underwater→electromagnetic→videogames を順に、前ドメインの last（`cat work_dir/last_checkpoint`）から `load_from` で学習 → 現＋全過去ドメイン＋ZCOCO を評価 → 次へ。bash 構文 OK。
- **batch_size 全ドメイン 6 に統一**（現在ドメインのバッチ=4、過去実験の batch4 と整合）。
- **スモークテスト（design 7節(d) 勾配疎通、GPU0・1バッチ forward/backward）**: loss=19.28（有限・健全）。勾配疎通=backbone 175/175・language 197/197・neck 16/16・encoder 276/276・decoder 174/174・その他 9/9（bbox_head 44/49 は1バッチで全クラス未活性のため正常）。**backbone/language にも勾配 → full FT が正しく動作**。学習パイプラインは正常に動く。

### 11.8 リプレイ混合の実装箇所（既存 mmdet の組み合わせ）
バッファ混合は新規コードでなく既存コンポーネントの config 合成で実現している（参照用に明記）。
- **バッファをデータに含める**: config の `ConcatDataset`。`reference_buffer`・`past_*` を、バッファサブセットファイル（`experiments/exp_023/buffer/*.odvg.json`）を指す `ODVGDataset` として `datasets=[現在, 参照, 過去…]` に束ねる。t=3 の過去は入れ子 `ConcatDataset`（過去プール）で1ソース化。
- **毎バッチ 現在・参照・過去 を必ず含める**: `MultiSourceSampler`（`mmdet/datasets/samplers/multi_source_sampler.py`、既存）。config で `batch_size=6, source_ratio=[4,1,1]` を宣言 → `__init__` が `num_per_source=[4,1,1]`（現在は余りを吸収）を算出し、`__iter__` が毎バッチ各ソースから正確にその件数を取り出す。`_infinite_indices` が各ソースを無限巡回するため 500/1000 件の小バッファも毎バッチ供給される。参照・過去を別ソースにして各々 ratio≥1 を与えることで「参照 AND 過去を毎バッチ」を保証（解釈B）。t=3 の過去枠1件はプール全体から一様（両過去ドメインの各1枚ではない）。

### 11.9 label_embedding 形状不一致の発見と num_classes=256 統一（2026-07-22）
- 発見: full FT＋replay のデバッグ学習で loss が過去実験（54–65）より大幅に小さい（14.6）ことから調査。過去 exp_020/021/022 のログに `size mismatch for dn_query_generator.label_embedding.weight: [256,256]→[28,256]` があり、num_classes=domain のため DN の label_embedding がランダム初期化されていたと判明。exp_023（num_classes=256）は θ0 から loads=True。
- 機構確認: label_embedding は DINO の CDN（denoising）専用で `if self.training:`（`grounding_dino.py`）により推論時は経路ごとスキップ → **評価 mAP に無影響**。ZiRa/DitHub は凍結だが終盤 dn_loss_cls が 0.03–0.27 まで低下＝DN タスクは解けており「汚染」は証拠で支持されず。ただし手法間で DN 初期化条件が食い違う交絡を排すため num_classes=256 に統一（§1.1(b)）。
- パラメータ数: label_embedding = 256×256 = 65,536（全体約1.7億の0.04%、推論非使用）。

### 11.10 ZiRa/DitHub リプレイフリー・論文準拠 config（2026-07-22）
既存の ZiRa/DitHub 検出器・層・neck・hook（exp_020/021 実装）は**無変更で流用**。config の `custom_imports` で読み込むのみ。mmdet/ への新規追加・変更なし。
- **方法別ベース**（新規）: `zira_replayfree_base.py` / `dithub_replayfree_base.py`。fullft_replay_base（pretrain 継承・num_classes=256・θ0・dn有効）を土台に、検出器差し替え＋論文準拠スケジュール/lr（§1.1(c)）。IterBasedTrainLoop・ckpt=last・in-training val なし。
- **ドメイン差分**（新規、各3本）: `{zira,dithub}_replayfree_{underwater,electromagnetic,videogames}.py`。ODVG 単一ソース（現在ドメインのみ、同一ドメイン負例）、InfiniteSampler、batch_size=4。DitHub は `dithub_classes` を各 label_map の index 順で付与（28/39/87）。
- **build 検証（6本合格）**: num_classes=256・label_embedding [256,256] loads=True・凍結、学習対象（ZiRa=neck ZiRa枝＋text_feat_map 2.60%、DitHub=クラス別 LoRA でクラス数比例 12.77/16.65/30.21%）、IterBasedTrainLoop・iter数・lr・wd・decay・InfiniteSampler・DitHub phase(warmup_iters=1500)・dithub_classes 数、を確認。

### 11.11 ZiRa・DitHub のハイパラを論文準拠に変更し、リプレイ有り版を追加（2026-07-22）

これまで ZiRa・DitHub の学習設定はプロジェクト共通値（20 epoch・学習率 1e-4）を使っていたが、公式コード準拠の値に変更した。変更の意図は §1.1(c) に記した。ZiRa は 2000 iteration・学習率 1e-3・weight decay 1e-4 で学習し、iteration 800 で学習率を 1/10 に落とし、LLRB のみ学習率を 0.2 倍にする。DitHub は 3000 iteration・学習率 1e-3・weight decay 1e-2 で学習し、iteration 1200 で学習率を 1/10 に落とし、iteration 1500 で warmup から specialization へ切り替える。これらの共通設定は方法ごとのベース config（`zira_replayfree_base.py`・`dithub_replayfree_base.py`）にまとめ、ドメイン別 config はデータだけを差し替える形にした。学習ループは iteration 基準（IterBasedTrainLoop）、サンプラは InfiniteSampler、batch size は 4、denoising は有効のまま、保存する checkpoint は last である。

これに加えて、ZiRa・DitHub の「リプレイ有り」版（`zira_replay_{domain}.py`・`dithub_replay_{domain}.py`）を新規作成した。これらはリプレイ無し版と同じ手法・同じスケジュールのまま、full finetuning + リプレイと同一の混合（ConcatDataset で現在ドメイン・参照・過去ドメインを束ね、MultiSourceSampler で毎バッチに必ず各ソースを含める）を載せたものである。batch size は 6 とし、内訳を現在ドメイン 4・バッファ 2 とした。これはリプレイ無し版の「現在ドメイン 4 枚/GPU」と現在ドメインの投入枚数を一致させ、リプレイ有無の差を「バッファ 2 枚を足すかどうか」だけに絞るためである。

DitHub のリプレイ有り版は、既存コードのままでは学習が停止する問題があったため、新規サブクラスで対処した。DitHub は specialization フェーズで画像ごとに GT クラスを 1 つ選び、そのクラス専用の LoRA（`per_class_lora_A`）を適用するが、この参照処理（`dithub_layers.py` の `_delta_per_sample`）にはライブラリ非登録クラスを弾く分岐が無い。リプレイ有り版では参照（Objects365）や過去ドメインの画像が混ざり、それらの GT クラスは現在ドメインの `dithub_classes` に含まれないため、specialization 中に KeyError で停止してしまう。そこで既存ファイルは一切変更せず、新規ファイル `exp023_dithub_replay.py` に、ライブラリ非登録クラスの画像では per_class_lora_A の代わりに warmup 用の共通 A（`warmup_lora_a`）を使うよう `_delta_per_sample` のみを上書きした `DitHubReplayLinear` と、それをモデルに注入する `DitHubReplayGroundingDINO` を定義した。非登録クラスを warmup 経路へ回す方針はユーザーが選択したものである。注入は、親クラスが生成した `DitHubLinear` インスタンスの `__class__` を差し替える方式で行う（新しいパラメータを増やさず挙動だけを変えるサブクラスなので安全）。検証では、非登録クラスを含む forward がクラッシュせず出力を返すこと、全 109 層が置換されること、対照として親クラスに戻すと同条件でクラッシュすることを確認した。

### 11.12 DitHub を逐次で正しく評価できるようにする修正（2026-07-22）

DitHub はクラスごとに別々の LoRA を持ち、評価時はプロンプトに含まれるクラスのうちライブラリにモジュールがあるものだけに LoRA を適用する（無いクラスは事前学習の重みのまま＝ゼロショット）。ところが 1 回の学習が保存する checkpoint には現在ドメインのクラス分の LoRA しか入らない。そのため逐次学習でドメインを進めると、過去ドメインを評価したときにその LoRA が checkpoint に存在せず、過去ドメインの検出性能が事前学習のゼロショット水準まで落ちてしまい、DitHub の忘却を実際より過大に見積もってしまう。これを避けるには、全ドメインの LoRA を蓄積した「ライブラリ」を保持して評価する必要があり、これは DitHub 本来の設計そのものである。実装方針はユーザーが選んだ公式忠実版（クラス名が重複するクラスに公式の式3 のマージを効かせる）とした。以下はすべて既存コードを変更せず、新規ファイルで実装した。

第一に、warmup から specialization への切替時に、現在ドメインのクラスのうち過去ドメインでも学習済みのクラスへ、公式の式3（新しい warmup 用 A と過去の A を λ_A=0.3 で混ぜる fetch+merge）を適用するフックを `exp023_dithub_seq.py`（`DitHubSeqPhaseHook`）に作った。既存の切替フックはこの「学習済みクラス集合」を渡さないため、過去クラスの A まで warmup で上書きしてしまう。ドメイン間のクラス重複を実測したところ、重複は videogames（t=3）の person と car の 2 つだけで、underwater・electromagnetic には重複が無かったので、videogames の config にだけこの 2 クラスを渡す。単体テストで、学習済みクラスの A が「0.3×warmup + 0.7×過去」になり、未学習クラスは warmup のコピーになることを確認した。

第二に、ドメイン境界でのライブラリ更新には既存の `merge_dithub.py` をそのまま用いる。これは公式の式4 に相当する共有 B の融合と、過去ドメインの A の引き継ぎ（和集合）を checkpoint に対して行い、全ドメインの LoRA を蓄積した「ライブラリ checkpoint」を作るスクリプトである。

第三に、過去ドメインを評価するための config を新規作成した（`dithub_eval_{underwater,electromagnetic,videogames,zcoco}.py`）。ここでは `dithub_classes` を 3 ドメインの全クラスの和集合（152 クラス）にしてモジュールの受け皿を全て確保し、育てたライブラリ checkpoint を漏れなく読み込めるようにした。評価時は既存のフィルタにより、プロンプトに含まれ、かつライブラリに存在するクラスにのみ LoRA が当たる（過去ドメインは学習済みの A で評価され、ZCOCO では person・car など重複クラスだけに適応が効き、他は事前学習のまま）。

最後に、これらをつなぐ逐次ドライバ `run_sequential_dithub.sh` を作った（引数で replayfree / replay を切り替える）。各ドメインで学習し、`merge_dithub.py` でライブラリを更新し、全クラス評価モデルにそのライブラリを読ませて現在ドメイン・全過去ドメイン・ZCOCO を評価する流れである。

### 11.13 ZiRa を逐次で正しく動かすための修正（rep_merge のバグ修正、2026-07-22）

ZiRa の実装を検証する過程で、既存コードにバグを見つけたため修正した（ユーザー承認済みの、既存無変更原則の例外）。ZiRa は各タスクの学習後に、高学習率の枝（HLRB）を低学習率の枝（LLRB）へ融合し、HLRB を初期状態に戻す操作（Rep+）を行うことで、LLRB に過去タスクの適応を蓄積していく手法である。ところがモデル内の融合メソッド `rep_merge()`（`zira_layers.py` の ZiRaLinear・ZiRaConvRDB）は、LLRB へ融合した直後に `init_rdb()` を呼んでおり、この `init_rdb()` は HLRB だけでなく LLRB も 0 に初期化してしまう。つまり融合したばかりの過去適応がその場で消えてしまう。exp_020 は単一ドメインで rep_merge を一度も呼んでいなかったため、このバグは表面化していなかった。修正として、融合後は HLRB と scaling だけを初期化し、LLRB は融合結果を保持するようにした。修正後、Rep+ の前後でモデルの出力が一致すること（＝適応が失われないこと）を確認した。

一方、checkpoint に対して同じ Rep+ を行うスクリプト `merge_hlrb.py` の方は LLRB を初期化しておらず、正しく動くことを確認した（融合後に出力が保存され、LLRB が「元の LLRB + scaling×HLRB」になり、HLRB と scaling が初期値へ戻る）。逐次ドライバではこの checkpoint 版を用いる。

評価用には ZiRa モデルの config を新規作成した（`zira_eval_{underwater,electromagnetic,videogames,zcoco}.py`）。ZiRa の枝（RDB）は全ての入力に一律に適用され、Rep+ によって LLRB に過去の適応が蓄積されるため、DitHub のようなクラス別ライブラリを必要とせず、単一のモデルで現在ドメインも過去ドメインも評価できる。

逐次ドライバ `run_sequential_zira.sh` を新規作成した（replayfree / replay を切り替える）。各ドメインで前ドメインの Rep+ 融合済み checkpoint から学習を開始し、融合前の checkpoint で現在ドメイン・全過去ドメイン・ZCOCO を評価し（ZiRa の forward は Rep+ の前後で数学的に等価なので融合前で構わない）、`merge_hlrb.py` で Rep+ を適用した checkpoint を次ドメインの初期重みとする。

### 11.14 条件A + リプレイの実装（2026-07-22）

条件A は、特徴抽出と融合を担うモジュール（Swin backbone・neck・BERT・text_feat_map・feature enhancer）だけを学習し、下流（decoder・bbox_head など）を凍結する条件で、定義は `experiment_notes/note02.md` にある。この凍結は専用の検出器ではなく optimizer の paramwise 設定（モジュールごとの学習率倍率）で実現できるため、config だけで書ける。学習データの混合・スケジュール・batch size は full finetuning + リプレイと共通なので、`condA_replay_{domain}.py` は `fullft_replay_{domain}.py` を継承し、paramwise の学習率倍率だけを差し替えた。実際に optimizer を組んで各モジュールの実効学習率を確認したところ、backbone と language_model が 1e-5、neck・text_feat_map・encoder・level_embed が 1e-4 で学習され、decoder・bbox_head・dn_query_generator・memory_trans_fc・memory_trans_norm・query_embedding は学習率 0（凍結）となっており、note02 の定義通りであった。条件A は素の GroundingDINO なので、評価には既存の `eval_{domain}.py`・`eval_base_coco.py` をそのまま使える。逐次ドライバ `run_sequential_condA.sh` は full finetuning 版と同型で、単一モデルを前ドメインの重みから継続学習する。

### 11.15 実装状況と残タスク（2026-07-22 時点）

比較6手法（full finetuning + リプレイ、ZiRa のリプレイ無し・有り、DitHub のリプレイ無し・有り、条件A + リプレイ）は、いずれも学習 config・評価 config・逐次ドライバまで実装し、単体・機構レベルの検証を終えた。既存コードへの変更は 2 箇所のみで、いずれもユーザー承認済みである（`zira_layers.py` の rep_merge バグ修正、`coco2odvg.py` の generic 変換モード追加）。それ以外はすべて新規ファイルで、mmdet 内の ZiRa・DitHub 検出器本体は exp_020・exp_021 のものを変更せずに流用している。

残タスクは次の通り。まず design.md の承認を得る（本番実行の前提、行動原理3）。承認後に各手法を逐次実行し、end-to-end の検証を行う（ZiRa の Rep+ による過去適応の蓄積、DitHub のライブラリの成長、過去ドメイン評価の実際の数値は、学習を実際に回さないと確認できない）。あわせて、単一ドメイン学習の数値が現行の全クラス連結方式とどれだけ変わるかを実測する（§7(c)）。最後に、ディスク回収（raw tar・zip・full ODVG）を行う。
