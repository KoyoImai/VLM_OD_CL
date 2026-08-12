# exp_037 設計書: DitHub の ODinW-13 再現（修正版の検証）

作成: 2026-08-10 / 状態: **未承認（承認後に実行）**

関連: [[../exp_034/design]]（ZiRa の ODinW-13 再現。本実験と同一の順序・θ0・評価集合）、
[[../../papers/DitHub_implementation_notes]]（手法の正本）、[[../exp_021/design]]・[[../exp_023/design]]（既存の DitHub 実行）。

---

## 0. 位置付け

exp_021（RF100 単発）と exp_023（RF100 逐次）で DitHub を動かしたが、逐次の結果が
「学習直後のドメインですら単発学習を大きく下回る」という説明のつかない形になっていた
（electromagnetic 0.039 / videogames 0.018 に対し、単発学習は 0.113 / 0.094）。
note04 にも「DitHub は現在バグが発生しており，デバッグは未完了」と記録されている。

2026-08-10 に原因を特定し修正した（§1）。修正後の実装が論文どおりに振る舞うかを、
**論文が主結果を報告している ODinW-13 で確認する**のが本実験である。RF100 は論文が
扱っていないため、実装の正しさをそこで判定することはできない。

本実験は exp_034（ZiRa の ODinW-13 再現）と、θ0・タスク順・データ版・評価集合・
評価プロトコルをすべて揃える。したがって **ZiRa と DitHub の直接比較**にもなる。

---

## 1. 修正した不具合と、公式実装との差分

`https://github.com/chiara-cap/DitHub` を取得し、`lora_pool.py` / `task_memory.py` /
`lora_utils.py` / `main.py` / `test/test_odinw13` を本実装と 1 対 1 で照合した。
修正は 5 件で、いずれも検証済み（§6）。設定値の照合結果は §4.1・§5.1 にある。

### 1.1 【重大】タスク 2 以降で warmup フェーズが実行されない

`DitHubLinear.dithub_per_class_enabled` は persistent buffer なので checkpoint に
保存される。逐次学習で前タスクのライブラリを `load_from` すると 1 が読み込まれ、
そのタスクは **iter 0 から specialization フェーズで始まる**。

結果として `warmup_lora_a` は勾配を受けず、iter = max_iters/2 の切替で全クラスの A が
「前タスクの warmup_lora_a」で上書きされ、それまでの 1500 iter の学習が破棄される。

実測の裏づけ（exp_023 replay-free）:

| | iter 1400 | 1500 | 1550 | 1600 |
|---|---:|---:|---:|---:|
| t=1 underwater | 11.82 | 10.97 | 10.60 | 12.53 |
| t=2 electromagnetic | 10.53 | 10.12 | **21.32** | 17.02 |
| t=3 videogames | 8.16 | 6.94 | **18.80** | 14.36 |

さらに `warmup_lora_a` が t=1 の ckpt と t=2 の ckpt で **109 層すべてビット一致**
（タスク 2 の 3000 iter で一度も勾配が流れていない）。

修正: `DitHubPhaseHook.before_train` で学習開始時に進捗からフェーズを決め直す
（新規タスク → WARMUP、途中再開 → 進捗どおり）。

### 1.2 【重大】単一クラスのタスクで warmup を行ってしまう

公式 `LinearPool.forward` の条件は
`do_warmup_a = self.training and num_classes_task > 1 and not self.per_class` であり、
**クラスが 1 つしかないタスクでは warmup を使わず、iter 0 からそのクラスの A を学習する**。
対応して `enable_per_class` も、単一クラスかつ未学習のときは A に触れない
（`_class_counter > 0` のときだけ式3を適用する）。

本実装はクラス数によらず warmup を使い、切替で必ず warmup A をコピーしていた。
**ODinW-13 は 13 タスク中 6 タスク（pistols / CottontailRabbits / Raccoon / Packages /
pothole / EgoHands）が単一クラス**なので、この差は再現に直接効く。RF100 の 3 ドメインは
すべて多クラスなので exp_021 / exp_023 には影響しない。

修正: `DitHubLinear` に `single_class_task`（= `len(class_keys) == 1`）を持たせ、
forward と `enable_per_class` を公式と同じ条件にした。

### 1.3 未学習クラスのランダム A が評価時の合成に混ざる

評価用モデルは全タスクのクラス分のスロットを確保するため、まだ学習していない
クラスの `per_class_lora_A` は kaiming 初期値のまま残る。本実装の評価 forward は
「スロットが存在するか」でしか絞っていなかったため、それが平均に入っていた。

実測: exp_023 の t=1 時点の ZCOCO は θ0 と mAP_75 だけ 0.0010 ずれる
（θ0 0.5520 に対し 0.5510。COCO の person / car / dog / bicycle / horse の
スロットが存在し、未学習の乱数 A が混入したため）。

公式は 13 タスクすべてを学習してから一括評価するのでこの状況が起きない。本実験は
各タスク後に評価するため、**ライブラリに保存済みのクラスだけを合成に使う**ように修正した
（checkpoint に含まれる `per_class_lora_A` のキー集合をライブラリとみなす）。
最終時点（t=13）では両者は一致する。

### 1.4 dn（contrastive denoising）が有効のままだった

公式は build 時に `dn_number=0`（`groundingdino_dt.py:703`）。exp_023 は他基準線と
条件を揃えるため dn を有効にしていた。本実験は公式に合わせて無効化する。
機構は exp_034 で用意した `use_dn=False` + `NoDNGroundingDINOHead` をそのまま使う。

### 1.5 【重要】warmup_lora_a がタスクをまたいで引き継がれていた

公式はタスクごとに **θ0 からモデルを作り直す**（`main.py:302` `load_model`、
`train.init_checkpoint` は空文字なので `resume_or_load` は何も読まない）。
LoRA の状態は checkpoint ではなく `TaskMemory` が運ぶが、`LinearPool.__init__` が
復元するのは `per_class_lora_A`（`get_modules`）と `shared_lora_b`（`get_memory`）だけで、
**`warmup_lora_a` は復元されず毎タスク kaiming で初期化される**。

本実装は前タスクのライブラリ checkpoint を `load_from` で読むため、`warmup_lora_a` も
そのまま引き継いでいた。タスク t≥2 の warmup の出発点が公式と変わり、そこからコピー・
融合される各クラスの A も変わる。

修正: `DitHubPhaseHook.before_train` で `runner.iter == 0`（タスクの先頭）のときだけ
`warmup_lora_a` を kaiming で引き直す。途中再開では引き継ぐ。

なお `per_class_lora_A` は公式でも `save_modules` が全クラスのキーを保存するため
タスクをまたいで持続する。本実装はタスクごとに当該タスクのクラスだけを確保するので、
新規クラスは各タスクで kaiming、既出クラスはライブラリから復元、となり公式と一致する。

### 1.6 その他（修正済みだが挙動は変わらないもの）

- `DecomposedMHA` の射影を batch-first に直してから Linear に通すようにした。
  `_delta_per_sample` が形状から先頭次元を推定しているため、系列長 = バッチサイズの
  ときに seq-first を batch-first と誤認しうる潜在的な危険を消すのが目的。出力は同一。
- `canonical_key` に括弧内除去を追加し、学習側（pipeline の `clean_name` 通過後）と
  評価側（metainfo の生のクラス名）でキーが一致するようにした。ODinW-13 と RF100 の
  クラス名に括弧は無いので実挙動は変わらない。

### 1.7 差分が残る点（意図的、または再現不能）

| 項目 | 公式 | 本実装 | 理由 |
|---|---|---|---|
| ベースモデル | GroundingDINO-T (O365+GoldG+Cap4M) | MM-Grounding DINO-T | 本プロジェクトの対象。θ0 が違うので絶対値は一致しない（exp_034 と同じ事情） |
| クラススロット | 全 13 タスク分を最初から確保 | タスクごとに当該タスク分のみ | B=0 と `enable_per_class` の対象が現タスク限定なので学習は等価。評価は §1.3 の絞り込みで一致 |
| 式3 の発火条件 | クラス単位のカウンタ（specialization 中に実際に選ばれた回数）>0 | 「過去タスクに出現したクラス」を config で列挙 | 過去タスクで一度も選ばれなかったクラスがあると差が出る。ODinW-13 では該当は稀 |
| タスク順 | `--seed 3` で glob 順をシャッフル | exp_034 と同じ seed 42 の順 | 公式の順は glob 順依存で再現不能。ZiRa と揃える方が本プロジェクトには有用 |
| seed 数 | 3 seed 平均 | 1 本 | コスト（§7）。exp_034 も 1 本 |

---

### 1.8 RF100 側との役割分担（2026-08-10 決定）

DitHub を RF100 で走らせる場合は、バッチサイズ・プロンプト生成（ODVG +
`RandomSamplingNegPos`）などをリプレイ・蒸留の比較対象と揃える。手法間の比較を
成立させることが目的なので、公式 DitHub の設定（total batch 2、全クラス固定
キャプション）には合わせない。

**本実験（exp_037）はその逆で、論文の再現ができているかの確認に専念する。**
したがって §4 のとおり公式の設定に揃え、RF100 とは条件を共有しない。両者は
目的が違うので、値を直接比べない。

## 2. 目的と問い

**問い**: 修正後の DitHub 実装は、論文が報告する挙動を ODinW-13 で再現するか。

**仮説**: 再現する。すなわち (a) ODinW-13 の平均 mAP が θ0 から大きく上がり、
(b) ZCOCO はほぼ保たれ、(c) 同一条件で走らせた ZiRa（exp_034）より平均 mAP が高い。

論文の該当値（3 seed 平均、GDINO-T ベース）:

| | ZCOCO | ODinW-13 Avg |
|---|---:|---:|
| ゼロショット GD | 47.41 | 46.80 |
| ZiRa | 46.26 | 57.98 |
| **DitHub** | **47.01** | **62.19** |

DitHub は θ0 比で Avg +15.39、ZCOCO −0.40。ZiRa は同表で Avg +11.18、ZCOCO −1.15。

---

## 3. 判定材料

**最終的な判定はユーザーが行う。** 本実験が提示する材料は次の 5 つ。

1. **適応**: ODinW-13 の平均 mAP（t=13 時点）と θ0 からの変化。論文の +15.39 と比較する。
2. **汎用知識の保持**: ZCOCO の t=13 時点の値と θ0（0.5040）からの変化。論文は −0.40 pt。
3. **ZiRa との相対**: exp_034（θ0・タスク順・データ・評価集合が完全に同一）の
   Avg 0.5942 / ZCOCO 0.4980 との比較。論文の順位（DitHub > ZiRa）が再現するか。
4. **修正の直接確認**: 各タスクの iter 1500 前後で損失が跳ねないこと（§1.1 の症状の消失）と、
   単一クラスの 6 タスクで warmup が走っていないこと（§1.2）。
5. **タスク別の対照**: θ0 / 学習直後 / 最終の 3 点をタスクごとに記録し、適応と忘却に分ける。

数値の閾値は設けない。

---

## 4. 条件

### 4.1 手法のハイパーパラメータ（公式実装と照合済み）

| 項目 | 公式の値 | 出典 | 本実験 |
|---|---|---|---|
| LoRA rank r | 16 | `GroundingDINO_SwinT_OGC_dt_dithub.py:57` | 16 |
| lora_alpha | 8（scaling = 0.5） | 同 :58 | 8 |
| lora_dropout | 0.0 | 同 :59 | 0.0 |
| 挿入対象 | out_features ≥ 128 の Linear、`transformer.dec* / bert / backbone / feat_map` を除外 | `get_lora_modules` | 同一（109 層。§6 で照合） |
| optimizer | AdamW lr 1e-3 / wd 1e-2 | `main.py:306-307` + model cfg | 同一 |
| iter / タスク | 3000（300 iter/epoch × 10） | `test/test_odinw13/for_train/*.py` | 3000 |
| lr 減衰 | iter 1200 で ×0.1 | `modified_coco_scheduler(10, 4, base_steps=300)` | MultiStepLR milestones=[1200] |
| batch | total 2 | 同 `total_batch_size = 2` | 2（1 GPU） |
| フェーズ切替 | `iter == int(max_iter/2)` = 1500 | `main.py:192-194` | warmup_iters=1500 |
| grad clip | max_norm 0.1 / L2 | 同 config | 同一 |
| λ_A（式3） | 0.3 | `lora_pool.py` | 0.3 |
| λ_B（式4） | 0.7 | `task_memory.merge_B` | 0.7 |
| dn | 0 | `groundingdino_dt.py:703` | 無効（§1.4） |
| 追加損失 | なし | — | なし |
| lr の linear warmup | **なし**（`warmup_epochs=0` → `warmup_length=0`） | `coco_schedule.py:91-125` | なし |
| AdamW betas | (0.9, 0.999) | `common/optim.py` | 同一（mmdet 既定） |
| wd の除外 | norm 層のみ wd=0（LoRA は全て行列なので該当なし） | `get_default_optimizer_params(weight_decay_norm=0.0)` | 該当なし |
| AMP / EMA | 無効 | `common/train.py`（`amp.enabled=False`, `model_ema.enabled=False`） | 無効 |
| max_text_len | 256 | `..._dt_dithub.py:33` | 256 |
| 学習の augmentation | 50% で Flip+多スケール resize、50% で Flip+resize(400/500/600)+RandomCrop(384,600)+多スケール resize | `data/odinw/*.py` の `augmentation` / `augmentation_with_crop` | 同一（exp_034 の train_pipeline） |
| 評価の resize | short 800 / max 1333 | 同 test mapper | 同一 |
| θ0 の再読込 | 毎タスク θ0 から作り直し、LoRA は TaskMemory が運ぶ | `main.py:302`, `train.init_checkpoint=""` | 前タスクのライブラリ ckpt を `load_from`（等価。§1.5 で warmup_lora_a だけ揃えた） |
| seed | `--seed 3`（タスク順のシャッフルにのみ影響） | `train_dithub.sh` | 0（プロジェクト規約）。順は §4.2 で固定 |

`num_classes` は 256 のまま（exp_034 と同じ。13 タスクで ckpt の形を一定に保つため。
dn を無効にするので `dn_query_generator.label_embedding` は参照されない）。

`--lora-r` の argparse 既定は 8 だが、実際には model config の `lora_r = 16` が
`apply_lora` に渡る（`main.py:225`）。`--lora-lr` / `--lora-weight-decay` の既定は
1e-3 / 1e-2 で、`train_dithub.sh` は上書きしていない。タスク config 側の
`optimizer.weight_decay = 1e-4` は `do_train` が後から 1e-2 で上書きする
（`main.py:306`）ので、**実効 wd は 1e-2** である。

### 4.2 データとタスク順

exp_034 の `odinw_official_tasks.py` をそのまま使う（公式 ZiRa 実装と同一の版・split。
評価は test split、PascalVOC のみ valid）。ODinW-13 のアノテーションは本環境に
全 13 タスク分が存在することを確認済み。

タスク順（seed 42。exp_034 と同一）:

```
pistols → PascalVOC → CottontailRabbits → Raccoon → VehiclesOpenImages → Packages
→ thermalDogsAndPeople → pothole → EgoHands → NorthAmericaMushrooms
→ AerialMaritimeDrone → Aquarium → ShellfishOpenImages
```

クラスは延べ 50、生のクラス名でのユニークは 46、`canonical_key` 正規化後（小文字化 +
非英数字を `_` に）は **43**（`Bus`/`bus`、`Car`/`car`、`CoW`/`cow` が同じスロットになる）。
評価モデルはこの 43 スロットを確保する。過去タスクと重複するクラス（= 式3 の対象）は
t=5 の Bus / Car、t=7 の dog / person、t=10 の CoW、t=11 の boat / car。
単一クラスのタスクは t=1, 3, 4, 6, 8, 9 の 6 つ。

### 4.3 逐次の手続き（タスク t ごと）

1. **学習**: `load_from` = 前タスクの成長ライブラリ（t=1 は θ0）。3000 iter。
   iter 1500 で `DitHubSeqPhaseHook` が specialization へ切替。現タスクのクラスのうち
   過去タスクで学習済みのものには式3を適用する。
2. **ライブラリ更新**: `experiments/exp_021/merge_dithub.py` で式4（B の融合、λ_B=0.7）と
   過去クラス A の和集合を取る。t=1 は学習 ckpt をそのまま採用（公式も最初のタスクでは
   式4 を実行しない）。
3. **評価**: 46 クラス分のスロットを持つ評価用モデルでライブラリ ckpt を読み、
   **学習済みの全タスク**と **ZCOCO** を評価する。

### 4.4 実行環境

本環境の A100 40GB **1 枚**。並列実行はしない。`encoder_cp=0`（activation checkpointing
なし。batch 2 なら不要で、出力は数学的に同一）。

---

## 5. 評価プロトコル

exp_034 と完全に同一にする。

- **タスク評価**: 公式と同じ評価集合（test split、PascalVOC のみ valid）。プロンプトは
  当該タスクのクラス名のみ。公式は `use_add_names=True` だが、DitHub の経路では
  `learned_classes` が空のままで実質無効なので、追加名は付けない（コード確認済み）。
- **ZCOCO**: COCO2017-val 5000 枚、80 クラス。
- **測定点**: 各タスク終了後に「学習済みタスク全部 + ZCOCO」。全 91 + 13 = 104 回。
- **θ0 の起点**: exp_034 で測定済みの値をそのまま使う（同一 ckpt・同一評価集合）。
  Avg 0.4971、ZCOCO 0.5040。**再測定はしない**。

記録は `experiments/exp_037/results/summary.md` に事実のみ（行動原理8）。

### 5.1 評価側で公式と一致しない点（本プロジェクト共通の定数）

いずれも exp_034（ZiRa）と共通なので、**ZiRa との比較には影響しない**。論文の絶対値との
比較にのみ効く。

| 項目 | 公式 | 本実験 | 影響 |
|---|---|---|---|
| 画像あたりの出力ボックス数 | 200（`select_box_nums_for_evaluation`） | 300（mmdet `test_cfg.max_per_img`） | AP に効きうる |
| COCO 評価の maxDets | 100（detectron2 COCOEvaluator の既定） | 1000（mmdet `CocoMetric` が `proposal_nums=(100,300,1000)` を `params.maxDets` に入れ、`stats[0]` を採る） | 検出数が多いぶん AP がわずかに高く出る方向 |
| クラス名の区切り | `"."`（`".".join(names) + "."`） | `". "`（mmdet `_special_tokens`） | tokenizer 上はほぼ同一 |

### 5.2 データ版の確認状況

公式 DitHub のリポジトリには ODinW のデータセット登録ファイルが含まれておらず、
13 タスク全部の版・split をコード上で直接確認することはできなかった。確認できたのは次のとおり。

- **コードで確認できたもの**: AerialMaritimeDrone = `tiled`、PascalVOC = 学習 `train` /
  評価 `valid`、VehiclesOpenImages = `416x416`、thermalDogsAndPeople = `train`/`test`、
  EgoHands = `generic`（`tools/overlapped_classes_dataset.py` と
  `config/configs/common/data/odinw/*.py` のファイル名）。**すべて exp_034 の表と一致**。
- **確認できなかったもの**: Aquarium / CottontailRabbits / NorthAmericaMushrooms /
  Packages / Raccoon / ShellfishOpenImages / pistols / pothole の版。

ただし DitHub のダウンロードスクリプトは ZiRa と同じ GLIP の `odinw_35` を落としており
（`tools/download_odinw.py`）、コードベースも ZiRa と同系譜である。したがって exp_034 が
ZiRa 公式から起こした表をそのまま使う。**この点は仮定であり、断定はできない。**

---

## 6. 実装と実行前検証

### 6.1 変更したファイル

| ファイル | 変更 |
|---|---|
| `mmdet/models/layers/dithub_layers.py` | `set_per_class_enabled` / `reinit_warmup_a` 追加、`single_class_task`、`library_class_keys` と `_load_from_state_dict`、`canonical_key` の括弧除去、`DecomposedMHA` の batch-first 射影 |
| `mmdet/models/detectors/dithub_grounding_dino.py` | `set_phase` / `reinit_warmup_a` 追加、`use_dn` 引数 |
| `mmdet/engine/hooks/dithub_phase_hook.py` | `before_train` 追加（フェーズ初期化 + warmup_lora_a の引き直し） |

いずれも本プロジェクトが追加したファイルであり、上流 mmdetection のコードには触れていない。
既定値は従来どおりなので、exp_021 の単発学習の挙動は変わらない。
**exp_023 の DitHub 逐次結果は、この修正の前に得られたものとして扱う**（再実行はしない）。

### 6.2 済んでいる検証

- `experiments/exp_021/check_dithub_setup.py`（既存 10 項目）: 修正後も **10/10 OK**。
  対象層 109（期待どおり）、attention 分解の等価性 差 0.000e+00、θ0 等価性 差 0.000e+00。
- `experiments/exp_037/check_dithub_phase_fix.py`（新規 8 項目）: **8/8 OK**。
  ライブラリ ckpt が enabled=1 を持つこと、load_from で混入すること、修正後は WARMUP に
  戻ること、途中再開では引き継ぐこと、実データ 1 iteration で warmup 期の勾配が
  warmup_lora_a に 109/109 層流れ per_class_A には流れないこと、修正前の経路では
  warmup_lora_a に 0/109 層しか流れないこと、切替の連続性（差 4.8e-07）、
  タスク先頭で warmup_lora_a が 109/109 層引き直され途中再開では 109/109 層保持されること。

### 6.3 実行前検証（`check_dithub_odinw13_setup.py`、9 項目・**9/9 OK**）

2026-08-10 実施。

| # | 項目 | 実測 |
|---|---|---|
| 1 | 単一クラスタスクの forward | pistols で per-class 経路との差 0.000e+00 / warmup 経路との差 3.905e-01 → per-class を使用 |
| 1b | 単一クラスの `enable_per_class` | 未学習の単一クラスで A が warmup で上書きされない |
| 2 | 多クラスタスクの warmup | PascalVOC で warmup 経路との差 0.000e+00 / per-class 経路との差 4.276e-01、切替で全 20 クラスが warmup をコピー |
| 3 | 評価時のライブラリ絞り込み | ライブラリ外のみのプロンプトで出力変化 0.000e+00、ライブラリ内で 5.257e-03。スロット 43 |
| 4 | dn 無効 | 実データで `model.loss()` を通し損失 21 項、`dn_*` は 0 項（dn 有効なら 39 項） |
| 5 | θ0 ロード | 形状不一致 0、真に欠落したキー なし。48 キーは ckpt の `in_proj`/`out_proj` から分解復元され、数値一致を確認 |
| 6 | データ版と実ファイル | 13 タスクの train / eval アノテーションすべて存在、テーブルと config の不一致なし |
| 7 | 式3 の `trained_classes` | 逐次順から決まる重複と完全一致（t=5 Bus/Car、t=7 dog/person、t=10 CoW、t=11 boat/car） |
| 8 | optimizer | 学習対象 327 個＝`lora_` を含むパラメータ数と一致、lr 0.001 / wd 0.01 の単一設定 |

---

## 7. コスト

学習は 3000 iter × 13 タスク。exp_034（ZiRa、2000 iter × 13、評価込みで 9.4 時間）を
基準に、iter 数 1.5 倍と LoRA 109 層のオーバヘッドを見込んで **12〜15 時間**（1 GPU）。
内訳の目安は学習 6〜8 時間、評価 104 回で 6〜7 時間。

ディスクは LoRA のみ保存ではなく検出器全体の ckpt を保存するため、13 タスク × 約 1.2 GB
＋ ライブラリ 13 本で **約 30 GB**。

---

## 8. リスクと caveat

- ベースモデルが違うため、論文の絶対値（Avg 62.19 / ZCOCO 47.01）とは一致しない。
  見るのは **θ0 からの変化量と、ZiRa との相対**である。
- 1 seed のみ。論文は 3 seed 平均。タスク順も公式と同一である保証はない。
  exp_034 と同順にしてあるので、ZiRa との比較には順序の交絡がない。
- 式3 の発火条件が公式のカウンタ方式と厳密には一致しない（§1.7）。
- 修正 §1.3 は公式に無い挙動である（公式は最終時点でしか評価しないため必要が無い）。
  最終時点では両者は一致するので、論文値との比較には影響しない。中間時点の値は
  「ライブラリに登録済みのクラスだけを使った場合」の値である。

---

### 6.4 実装済みの資産（2026-08-10、実行前に用意したもの）

| ファイル | 内容 |
|---|---|
| `configs/dithub_odinw13_base.py` | 共通ベース（公式準拠のスケジュール・dn 無効・COCO 形式 pipeline・batch 2） |
| `gen_configs.py` | 以下の生成器。タスク定義は exp_034 の `odinw_official_tasks.py` を共有する |
| `configs/dithub_odinw13_<task>.py` × 13 | 学習 config（`dithub_classes` と `trained_classes` を含む） |
| `configs/dithub_odinw13_eval.py` | 全 13 タスクの評価 config（43 スロットの DitHub モデル） |
| `configs/zcoco_eval.py` | ZCOCO 評価 config（同上） |
| `configs/_eval_after_t{01..13}.py` | 学習済みタスクだけを評価する部分集合 |
| `run_sequential_dithub_odinw13.sh` | 逐次ドライバ（学習 → 式4 ライブラリ更新 → 評価。再開可能） |
| `check_dithub_odinw13_setup.py` | §6.3 の 9 項目 |
| `check_dithub_phase_fix.py` | §1.1 / §1.5 の回帰テスト 8 項目 |

---

## 9. 承認をお願いする範囲

- §4 の条件で ODinW-13 を 13 タスク逐次学習し、§5 のプロトコルで評価すること。
- 実行は本環境の GPU 1 枚、並列なし。所要 12〜15 時間。
- 実装・検証は済んでおり（§6.3 が 9/9 OK、§6.4 の資産が揃っている）、**残るのは実行のみ**。
  実行手順は [[README]] に置いた。
