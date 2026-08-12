# exp_038 設計書: 素の LoRA（特徴抽出部＋融合部）の ODinW-13 逐次学習

作成 2026-08-11。**実行前のユーザー承認ゲート**（行動原理1・3）。
関連: [[../exp_034/design]]（ZiRa 再現）、[[../exp_037/design]]（DitHub 再現）、
[[../exp_008/design]]（素の LoRA の実装確立と sweep）、[[../../experiment_notes/note09]]。

## 0. 位置付け

exp_034 と exp_037 で、ODinW-13 の同一条件（同じ θ0・同じタスク順・同じデータ版・同じ評価
プロトコル）における ZiRa と DitHub の値が揃った。両者はいずれも「どこに何を足すか」を設計した
手法であり、その効果を測るには**同じ土俵に立つ素の LoRA の値**が要る。

exp_008 は素の LoRA を RF100 の単一ドメインで測ったもので、逐次学習は行っていない。本実験は
素の LoRA を ODinW-13 の 13 タスク逐次に載せ、ZiRa・DitHub と直接比較できる形にする。

## 1. 目的

素の LoRA を、事前学習済み VLM の**特徴抽出部（Swin / BERT）と融合部（feature enhancer）
および text_feat_map** に挿入し、ODinW-13 を逐次学習したときの適応と忘却を測る。

**仮説と判定基準は実験ノート（note09）でユーザーが確定する。** 本書は測るものと条件を定める。

## 2. 条件

### 2.1 LoRA の挿入箇所（232 層）

`r=16 / alpha=8`（scaling = 0.5）で、学習パラメータは **5,671,936 / 178,649,629 = 3.17%**。
下表は実機で構築して数えた値。

| 構成要素 | 層数 | 対象 |
|---|---:|---|
| image backbone（Swin-T） | 51 | `attn.w_msa.{qkv, proj}` 24、FFN 24、`downsample.reduction` 3 |
| text backbone（BERT） | 72 | 12層 × `{query, key, value, attention.output.dense, intermediate.dense, output.dense}` |
| feature enhancer（encoder） | 108 | 画像 self-attn 24、画像 FFN 12、テキスト self-attn 24、テキスト FFN 12、融合層 36 |
| `text_feat_map` | 1 | 768→256 |

テキスト self-attention の 24 層は `nn.MultiheadAttention` を q/k/v/o の独立 `nn.Linear` に
分解して挿入する（`lora.decompose_mha`）。分解は数学的に等価な書き換えで、推論はビット単位で
一致することを実測確認済み（§6.2）。

**対象外**: decoder、bbox_head、`memory_trans_fc`、`query_embedding`、`dn_query_generator`
（挿入可能だが本条件では入れない）。neck（Conv2d 4個）と `level_embed`（Parameter）は
`nn.Linear` でないため LoRA を掛けられない。

この集合は note04 の条件B（特徴抽出部＋融合部）から、LoRA を掛けられない neck と `level_embed`
を除いたものに一致する。exp_025 の条件B（`lr_mult=0` による凍結）と対応づけて読める。

LoRA 以外の全パラメータは `requires_grad=False`。optimizer には LoRA のみ登録する
（`TrainableParamsConstructor`）。

### 2.2 学習設定（DitHub 準拠＝ exp_037 と同一）

| 項目 | 値 | 根拠 |
|---|---|---|
| iteration | 3000 / タスク | exp_037 と同一 |
| batch size | 2（1 GPU） | 同上 |
| optimizer | AdamW lr 1e-3 / weight_decay 1e-2 | 同上 |
| lr schedule | `MultiStepLR` milestones=[1200], gamma=0.1, warmup なし | 同上 |
| grad clip | max_norm 0.1（L2） | 同上 |
| dn | 無効（`use_dn=False` ＋ `NoDNGroundingDINOHead`） | 公式 ZiRa / DitHub がともに dn_number=0 |
| num_classes | 256 | 13 タスクで ckpt の形を一定に保つ |
| train pipeline | COCO 形式（`RandomSamplingNegPos` なし） | exp_034 / exp_037 と同一 |
| AMP / EMA | なし | 同上 |
| seed | 0 | プロジェクト規約 |

### 2.3 逐次学習の方式（各タスク後に base へ焼き込む）

```
t=1: θ0     + LoRA(B=0) を 3000 iter 学習 → merge_lora → θ1
t=2: θ1     + LoRA(B=0) を 3000 iter 学習 → merge_lora → θ2
...
t=13: θ12   + LoRA(B=0) を 3000 iter 学習 → merge_lora → θ13
```

各タスクの学習開始時に LoRA は B=0 から引き直す（＝ΔW=0、直前の θ と厳密に等価な状態から
始まる）。マージ時に分解した MHA は `in_proj_weight` へ再融合するので、θt は常に plain な
GroundingDINO 構造であり、評価は plain config でそのまま行える。

パラメータ数は増えない一方で base が毎タスク更新されるため、**ゼロショット保護の機構は無い**。
ZCOCO の劣化が ZiRa / DitHub より大きく出ることは設計上あり得る。これは条件の帰結であって
失敗ではない。

### 2.4 データと逐次順

exp_034 / exp_037 と完全に同一。タスク定義は `experiments/exp_034/odinw_official_tasks.py`
を import して共有し、複製しない。逐次順は seed 42 で決まる次の並び。

```
pistols → PascalVOC → CottontailRabbits → Raccoon → VehiclesOpenImages → Packages
→ thermalDogsAndPeople → pothole → EgoHands → NorthAmericaMushrooms
→ AerialMaritimeDrone → Aquarium → ShellfishOpenImages
```

## 3. 判定材料

**最終的な判定はユーザーが行う。** 数値の閾値は設けない。以下を材料として揃える。

0. **起点（θ0）のゼロショット性能**: ODinW-13 の 13 タスクと ZCOCO を、本実験で生成した
   plain 評価 config で測る（t=0）。exp_034 の実測値（タスク別は同 design §5.1、Avg 0.4971、
   ZCOCO 0.5040）と一致することを確認し、評価系が 3 実験で同一であることの裏付けとする。
1. **適応**: 13 タスクの最終 mAP の平均（Avg）と、θ0 からの変化。タスク別の内訳も出す。
2. **忘却**: 各タスクの「学習直後」と「t=13 時点」の差。タスク別に出し、平均化した値も併記する。
3. **ゼロショット保持**: ZCOCO の t=0〜13 の推移。
4. **ZiRa / DitHub との相対**: 同一条件で測った exp_034（Avg 0.5942 / ZCOCO 0.4980）と
   exp_037（Avg 0.6466 / ZCOCO 0.4980）との差。起点は θ0（Avg 0.4971 / ZCOCO 0.5040）。
5. **13×13 の推移表**: 各時点における学習済みタスクの mAP（exp_037 §4 と同形式）。

## 4. 手順

**t=0**: 学習の前に θ0 を評価する。ODinW-13 の 13 タスク全部と ZCOCO を、本実験で生成した
plain 評価 config で測り、exp_034 の実測値と照合する（§3 の判定材料0）。

その後、各タスク t について次を繰り返す。

1. 学習。`load_from` は θ_{t-1}（t=1 は θ0）。
2. `python projects/lora_cl/merge_lora.py <last ckpt> <θt>`。LoRA を base へ畳み、
   分解した MHA を `in_proj` へ再融合する。
3. 評価。θt に対して「学習済みタスク 1..t」と ZCOCO を測る。

```bash
cd /workspace/kouyou/mmdetection

# 実行前検証（全項目 OK でなければ学習を始めない）
python experiments/exp_038/check_lora_odinw13_setup.py
python projects/lora_cl/check_lora_setup.py
python projects/lora_cl/check_mha_decompose.py
python projects/lora_cl/check_backbone_lora.py

# 逐次学習と評価（GPU 1 枚・約 17 時間・途中で止まっても同じコマンドで再開）
GPU=0 nohup bash experiments/exp_038/run_sequential_lora_odinw13.sh 42 \
  > experiments/exp_038/run_s42.log 2>&1 &
```

## 5. 評価プロトコル

exp_034 / exp_037 と完全に同一の評価集合・プロトコルを使う。評価対象が plain な ckpt に
なるため、model 部分だけ plain GroundingDINO にした評価 config を新たに生成する
（データセット定義・評価器・`max_per_img` などは exp_034 の `odinw13_official_eval.py` と
同一の内容を `gen_configs.py` が出力する）。

この評価 config は本実験で新規に作るものなので、**t=0 で θ0 を測って exp_034 の値と一致するか
を確認する**（§4）。exp_034 の θ0 評価は ZiRa 検出器（初期状態で θ0 と等価）で行われたため、
plain 検出器での測定は評価系の独立な検算になる。一致しなければ学習に進まない。

ZCOCO は COCO2017 val 5000 枚のゼロショット評価。exp_034 / exp_037 と同一の config を使う。

評価側の既知の差（`max_per_img` 300 対 200、`maxDets` 1000 対 100、クラス区切りが `'. '` 対
`'.'`）は exp_034 / exp_037 と共通の定数であり、3 実験を横に比較する分には影響しない。

## 6. 実装

### 6.1 既に実装済み（本セッションで対応）

| 項目 | 場所 | 状態 |
|---|---|---|
| `init_weights` 後の B=0 復元 | `projects/lora_cl/grounding_dino_lora.py` | 検証 10/10 |
| MHA の q/k/v/o 分解（`decompose_mha`） | 同上（`DecomposedMHA` を import） | 検証 10/10 |
| Swin / BERT の対象化（`exclude_components`） | 同上 | 検証 10/10 |
| dn の無効化（`use_dn=False`） | 同上 | 本設計で追加。§6.3 で検証する |
| マージ時の MHA 再融合 | `projects/lora_cl/merge_lora.py` | 検証済み（plain へ missing/unexpected 0 でロード） |

mmdet 本体は変更していない。

### 6.2 等価性の実測（既存）

| 比較 | 結果 |
|---|---|
| `DecomposedMHA` vs `nn.MultiheadAttention`（GPU・推論） | encoder 出力・検出スコア・bbox すべて `max|Δ| = 0` |
| 同（学習モード） | 損失の相対差 1.1e-5、勾配の相対差 1.4e-3（float32 の演算順序差。同一モデル2回の対照は差 0） |
| B=0 の LoRA モデル vs plain GroundingDINO | `max|Δscore| = 0` |
| マージ後 ckpt → plain GroundingDINO | missing / unexpected ともに 0、検出スコア一致 |

### 6.3 これから作るもの

| 成果物 | 内容 |
|---|---|
| `experiments/exp_038/configs/lora_odinw13_base.py` | 共通ベース（手書き）。exp_037 の base と同じスケジュールで model を `GroundingDINOLoRA` に差し替える |
| `experiments/exp_038/gen_configs.py` | 13 タスクの学習 config、plain 評価 config、ZCOCO 評価 config、13 個の部分集合評価 config を生成 |
| `experiments/exp_038/run_sequential_lora_odinw13.sh` | 学習 → merge → 評価 を 13 回。再開可能 |
| `experiments/exp_038/check_lora_odinw13_setup.py` | 実行前検証。232 層の内訳、θ0 等価性、dn が切れていること、merge → plain の往復、実データ 1 step |

### 6.4 実行前に確認する項目

1. LoRA 232 層の内訳が §2.1 と一致
2. θ0 ロード後に `max|ΔW| = 0`（B=0 が保たれている）
3. `dn_loss_*` が損失に出ないこと（dn が無効）
4. 学習 → merge → plain ロードの往復で missing/unexpected 0
5. 実データ 1 step で base 不変・LoRA に勾配

## 7. コスト

実測（`pistols` の実データ、batch 2、A100 40GB 1 枚）で **1 iteration = 0.756 s**、
ピーク VRAM 7.29 GiB。

| 項目 | 見積もり |
|---|---|
| t=0 の θ0 評価（ODinW 13 タスク ＋ ZCOCO） | 約 1 h |
| 学習 | 0.63 h/タスク × 13 = **8.2 h** |
| 評価（ODinW 累積 13 回 ＋ ZCOCO 13 回） | 約 8.3 h（exp_037 の実績から） |
| 合計 | **約 18 h**（GPU 1 枚、逐次・並列なし） |
| ディスク | 学習 ckpt 13 本 ＋ θ1..θ13 の 13 本で約 30 GB |

## 8. リスク

- **ZCOCO の劣化**が ZiRa / DitHub より大きくなる可能性がある。§2.3 のとおり保護機構が無い
  条件なので、そうなっても条件の帰結として記録する。
- **テキスト self-attention の q/k は勾配が小さい**（DitHub 実装での実測で q≈1e-4 / k≈8e-4 に
  対し v/o≈1e-1）。バッチによっては float32 で 0 に落ちる。24 層のうち q/k の 12 層は
  ほとんど動かない可能性がある。実装の不具合ではないことは DitHub 側の実測で確認済み。
- **Swin / BERT に LoRA を入れるのは本プロジェクトで初めて**。exp_008 は encoder/decoder/
  bbox_head のみだった。実行前検証（§6.4）で勾配疎通を確認してから始める。

## 9. 承認をお願いする範囲

§2 の条件（挿入箇所 232 層、DitHub 準拠のスケジュール、各タスク後にマージする逐次方式、
r=16 / alpha=8）で、§4 の手順により θ0 のゼロショット評価（t=0）と ODinW-13 の 13 タスクの
逐次学習・評価を行うこと。所要は GPU 1 枚で約 18 時間。

**2026-08-11 承認・実行済み。結果は §11 と `results/summary.md`。**

---

# 追加条件B: BERT を凍結（2026-08-12 追記・承認待ち）

## 10.1 動機

条件A（232 層）は ODinW-13 Avg 0.4971 → 0.1595、ZCOCO 0.5040 → 0.0020 と崩壊した。
実装の不具合ではないことは検証済み（学習中の非 LoRA パラメータは base 932 キーが θ0 と
厳密一致、マージは独立再計算で 908 キーすべて差 0、マージ前後の ZCOCO が 0.2470 で一致）。

t=1 時点の測定では、更新量そのものは過大ではなかった（encoder 側の ‖ΔW‖/‖W‖ は中央値
0.0291 で、ZCOCO を保った DitHub の 0.0268 とほぼ同じ）。一方で構成要素別に見ると
Swin が中央 0.0697、BERT が中央 0.0494（最大 0.1343）と特徴抽出部が最も動いており、
COCO 80 クラス名のテキスト特徴は 1 タスクで中央 29% ずれ、クラス間の平均 cos 類似が
0.5175 → 0.5861 に上がって識別性が落ちていた。

本条件は、テキスト backbone を適応対象から外したときに何が変わるかを見る。

## 10.2 条件（条件A との差分は 1 点のみ）

| | 条件A（実行済み） | 条件B（本追記） |
|---|---:|---:|
| image backbone（Swin-T） | 51 | 51 |
| text backbone（BERT） | **72** | **0（凍結）** |
| feature enhancer（encoder） | 108 | 108 |
| `text_feat_map` | 1 | 1 |
| **合計** | **232** | **160** |

config 上の差分は `lora.include` から `'language_model'` を外すことだけ。
`exclude_components=[]` のままだと BERT が候補に残るため、`exclude_components=['language_model']`
を明示して既定除外に戻す。

その他（r=16 / alpha=8、3000 iter、batch 2、AdamW lr 1e-3 / wd 1e-2、iter 1200 で ×0.1、
grad clip 0.1(L2)、dn 無効、num_classes 256、COCO 形式 pipeline、seed 0、タスク順 seed 42、
各タスク後に base へマージ、評価集合・評価プロトコル）は条件A と完全に同一。

## 10.3 判定材料

条件A と同じ 5 項目（適応 / 忘却 / ZCOCO 推移 / ZiRa・DitHub との相対 / 13×13 の推移表）に
加えて、**条件A との差**を並べる。θ0 の評価（t=0）は条件A で実測済みなので再測定しない。

## 10.4 コスト

条件A の実績が 13 時間（学習 ＋ 評価、GPU 1 枚）。LoRA 層数は減るが backward は Swin を
通るため学習時間はほぼ変わらない見込みで、**約 13 時間**。ディスクは約 30 GB。

## 10.5 事前に述べておく懸念

BERT を外しても Swin の適応は残る。t=1 の実測で Swin の ‖ΔW‖/‖W‖ は中央 0.0697 と
BERT（0.0494）より大きく、画像側の表現も相応に動く。したがって**崩壊が止まる保証はない**。
また base へ係数 1.0 で焼き込む点（§2.3、融合係数による減衰なし）と、適用範囲を絞る機構が
無い点は条件A と変わらない。本条件が切り分けるのは「テキスト backbone の寄与」だけである。

## 10.6 承認をお願いする範囲

§10.2 の条件（挿入箇所 160 層、他は条件A と同一）で、§4 の手順により ODinW-13 の 13 タスクを
逐次学習・評価すること（t=0 は再測定しない）。所要は GPU 1 枚で約 13 時間。

**2026-08-12 承認・実行中。**

---

# 追加条件C: 学習率 1e-4（2026-08-12 追記・承認待ち）

## 11.1 動機

条件A の lr は 1e-3 で、これは exp_037（DitHub 準拠）に揃えた値である。一方、本プロジェクトの
RF100 側のリプレイ（exp_027）・蒸留（exp_028）は AdamW lr 1e-4 / wd 1e-4 を使い、さらに
`paramwise_cfg` で `backbone`（Swin）と `language_model`（BERT）に `lr_mult=0.1` を掛けて
実効 1e-5 に抑えている。条件A は LoRA を `TrainableParamsConstructor` でフラットに登録する
ため `lr_mult` が効かず、Swin / BERT を含む全 LoRA が一律 1e-3 で動く。特徴抽出部に限れば
実効学習率は **1e-5 対 1e-3 で 100 倍**の開きがある。

exp_008 では、RF100 単一ドメインの素の LoRA（encoder+decoder+bbox_head の 143 層）で
lr=1e-4 と 1e-3 を比較しており、1e-4 では適応 0.218（frozen FT 0.316 に届かず）／COCO 0.417、
1e-3 では適応 0.315 ／ COCO 0.315 だった。lr がトレードオフ点を動かす主要な軸であることは
この時点で分かっている。

本条件は、**lr だけ**を 1e-3 → 1e-4 に変えて条件A と対比する。

## 11.2 条件（条件A との差分は lr のみ）

| | 条件A（実行済み） | 条件C（本追記） |
|---|---|---|
| lr | 1e-3 | **1e-4** |
| lr schedule | milestones=[1200] gamma=0.1（→1e-4） | milestones=[1200] gamma=0.1（**→1e-5**） |
| 挿入箇所 | 232 層（Swin 51 / BERT 72 / encoder 108 / text_feat_map 1） | 同一 |
| weight_decay | 1e-2 | **1e-2 のまま** |
| その他 | 3000 iter, batch 2, clip 0.1, dn 無効, num_classes 256, seed 0, 各タスク後にマージ | 同一 |

weight_decay を 1e-4 に揃えない理由は、2 要因を同時に動かすと lr の効果を切り分けられなく
なるため。`lr_mult` による Swin / BERT の抑制も本条件では行わない（それは別の軸であり、
必要なら条件D として立てる）。

## 11.3 判定材料

条件A との差を並べる（lr 以外同一なので差分は lr に帰属する）。項目は §3 と同じ 5 つ。
θ0 の評価（t=0）は条件A で実測済みなので再測定しない。

## 11.4 コスト

条件A と同一構成なので **約 13 時間**（GPU 1 枚）。ディスク約 30 GB。
条件B と並行実行する場合は別 GPU を使う（本環境は A100 40GB × 4、条件B が GPU 0 を使用中）。

## 11.5 事前に述べておく懸念

exp_008 の実測では lr=1e-4 の素の LoRA は適応が frozen FT 水準に届かなかった。ODinW-13 でも
**「崩壊は止まるが適応もしない」**という結果になり得る。その場合、条件C は「lr が崩壊の主因
である」ことの証拠にはなるが、実用的な設定を与えるものにはならない。

また base へ係数 1.0 で焼き込む点（§2.3）と、適用範囲を絞る機構が無い点は条件A・B と変わらない。

## 11.6 承認をお願いする範囲

§11.2 の条件（lr=1e-4、他は条件A と同一の 232 層）で、§4 の手順により ODinW-13 の 13 タスクを
逐次学習・評価すること（t=0 は再測定しない）。所要は GPU 1 枚で約 13 時間。
