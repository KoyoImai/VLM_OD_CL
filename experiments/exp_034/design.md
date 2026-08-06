# exp_034: ZiRa の ODinW-13 再現（実装の動作確認）

- 作成日: 2026-08-03
- 状態: **承認待ち（未承認・未実行）**。承認前に実験は実行しない（行動原理3）。
- 論文: `papers/ZiRaGroundingDINO.pdf`、読解記録: `papers/ZiRa_implementation_notes.md`
- 実行環境: **本環境**（A100 40GB × 4。学習は1枚のみ使用。§7）。クラスタは使わない。

---

## 0. 位置付け・目的

exp_023 で ZiRa を RF100 の3ドメインに適用したが、**実装が正しく動作しているかを論文の土俵で
確かめたことがない**。本実験は ZiRa 本来のベンチマークである ODinW-13 で逐次学習・評価し、
**既存実装の妥当性を確認する**。新しい知見を得る実験ではない。

## 1. ベースモデルが論文と異なる（判定に直結するので最初に読む）

論文と公式コードの出発点は **Grounding DINO Swin-T（O365+GoldG+Cap4M）**（実装ノート §6）。
本実験は **MM-Grounding DINO の θ0（O365,GoldG,GRIT,V3Det）** を使う（2026-08-03 決定）。

| | ZiRa 論文 | **本実験** |
|---|---|---|
| ベースモデル | GDINO-T (Cap4M) | **MM-GDINO-T = θ0** |
| ZCOCO（ゼロショット） | 47.37 | **50.4**（exp_001 実測 0.504） |
| ODinW-13 平均（ゼロショット） | **46.70**（Tab.1 "Original Model" 行の13タスク平均。2026-08-05 に本文から再確認） | **53.30**（2026-08-03 に本環境で実測。上流公表値 0.533 と一致） |

**注意**: 論文の 46.70 と本実験の 53.30 は**ベースモデルだけでなく評価集合も違う**（§6.5(3)）。
同じ GDINO-T でも、論文の評価集合では Avg 46.70、mmdet の評価集合では 51.4（上流公表値）になる。
差はタスクによって大きく、NorthAmericaMushrooms 25.39 対 50.7、CottontailRabbits 64.72 対 79.9、
Packages 54.49 対 68.7。**論文の絶対値との比較は評価集合を揃えない限り成立しない。**

**論文の絶対値（ZCOCO 46.06 / Avg 59.73）とは直接比較できない。**出発点が違うので、絶対値の
一致を再現の判定基準にはできない。判定はゼロショットからの**相対変化**で行う（§5）。

## 2. 実験設定（公式コード準拠に戻す）

exp_023 の ZiRa config は他の比較手法と条件を揃えるため2点を意図的に変えていた。
**本実験では公式に戻す**（2026-08-03 指定）。

| 項目 | 公式（実装ノート §6） | exp_023 | **exp_034** |
|---|---|---|---|
| **dn（denoising）** | **無効** | 有効 | **無効**（実装コストあり。§6.1） |
| **batch size** | **2** | 4/GPU × 4GPU = 16 | **2（合計）＝1 GPU × 2** |
| optimizer | AdamW, lr 1e-3, wd 1e-4 | 同 | 同 |
| スケジュール | タスクあたり **2000 iter**、iter 800 で lr×0.1 | 同 | 同 |
| LLRB の学習率 | lr × η、**η = 0.2**（実効 2e-4） | 同 | 同 |
| ZiL 重み | **λ = 0.1** | 同 | 同 |
| grad clip | max_norm 0.1, norm_type 2 | 同 | 同 |
| スケール s 初期値 | 0.1、HLRB 初期値 1e-8、LLRB 初期値 0 | 同 | 同 |
| 学習対象 | RDB のみ（base は凍結） | 同 | 同 |
| Rep+ | 各タスク終了後に `W_llrb ← W_llrb + s·W_hlrb`、HLRB→1e-8、s→0.1 | 同 | 同 |

- **batch 2 は「合計 2」と解釈**する。4 GPU に分けると合計 8 になり公式と乖離する。
  したがって**学習は 1 GPU（単一プロセス）で回す**。
- **`num_classes` は 256 のまま**とする。dn を無効化するので DN の `label_embedding` は
  使われなくなるが、13タスクを逐次で読み継ぐためにチェックポイントの形を全タスクで
  一定にする必要がある（タスクごとのクラス数 1〜20 に合わせると次タスクでロードできない）。

## 3. データ（公式実装と同一の版・split。COCO 形式のまま変換しない）

**2026-08-05 決定: 学習・評価とも論文（公式実装）と同じデータを使う。**
本リポジトリの `configs/mm_grounding_dino/odinw/…odinw13.py` は valid split かつ一部タスクで
別バージョンを使っており、同じ GDINO-T でもゼロショット Avg が 51.4（mmdet）対 46.70（論文）と
5 ポイント近く違う（§6.5(3)）。論文と同じ土俵で判定するため公式側に合わせる。

- 実体 `/workspace/kouyou/datasets/odinw/`（2.3GB）→ `data/odinw`（シンボリックリンク）。
  **公式版の学習13タスク（計 24,883 枚）・評価13タスクとも既に存在し、画像の欠損は 0（2026-08-05 実測）。
  追加ダウンロードは不要。**
- クラス名は公式版と mmdet 版で**全タスク完全に一致**（実測）。したがって `class_name` は
  odinw13 config のものを流用でき、モデルに与えるプロンプトは変わらない。
- **学習**: 各タスクの train split、`CocoDataset` + `return_classes=True`、`filter_cfg` は
  `dict(filter_empty_gt=False, min_size=32)`（公式の `filter_empty=False` に対応。`min_size=32` で
  落ちる画像は13タスクとも 0 枚）。
- **pipeline**: RF100 用 finetune config と同じ構成。事前学習 pipeline から
  **`RandomSamplingNegPos`（ODVG 専用）と `FilterAnnotations` を外す**。
  負例サンプリングは行わず、そのタスクの全クラス名がそのままプロンプトになる。
  最大クラス数は PascalVOC の 20 で `max_text_len=256` に十分収まる（最長でも 51 トークン）。
- **評価**: 各時点で学習済みタスク分だけを `MultiDatasetsEvaluator` に渡す（§5）。ZCOCO も毎時点。

### 使用するデータ（2026-08-05 に実測）

| タスク | 使うディレクトリ | train 画像 | box | クラス | 4000枚投入時の周回 | eval 画像 |
|---|---|---:|---:|---:|---:|---:|
| AerialMaritimeDrone | `AerialMaritimeDrone/tiled/` | 371 | 1,237 | 5 | 10.8 | 32 (test) |
| Aquarium | `Aquarium/Aquarium Combined.v2-raw-1024.coco/` | 448 | 3,324 | 7 | 8.9 | 63 (test) |
| CottontailRabbits | `CottontailRabbits/` | 1,980 | 2,070 | 1 | 2.0 | 10 (test) |
| EgoHands | `EgoHands/generic/` | 3,840 | 12,015 | 1 | 1.0 | 480 (test) |
| NorthAmericaMushrooms | train=`v1-416x416.coco/`, eval=`v2-416x416augmented.coco/` | 41 | 67 | 2 | 97.6 | 246 (**v2 の train**) |
| Packages | `Packages/augmented-v1/` | 95 | 155 | 1 | 42.1 | 3 (test) |
| PascalVOC | `PascalVOC/` | 13,690 | 31,356 | 20 | 0.3 | 3,422 (**valid**) |
| Raccoon | `Raccoon/Raccoon.v38-416x416-resize.coco/` | 150 | 164 | 1 | 26.7 | 17 (test) |
| ShellfishOpenImages | `ShellfishOpenImages/416x416/` | 406 | 858 | 3 | 9.9 | 58 (test) |
| VehiclesOpenImages | `VehiclesOpenImages/416x416/` | 878 | 1,676 | 5 | 4.6 | 126 (test) |
| pistols | `pistols/export/`（画像は直下、ann は `train_/test_` 接頭辞） | 2,377 | 2,728 | 1 | 1.7 | 297 (test) |
| pothole | `pothole/` | 465 | 1,256 | 1 | 8.6 | 67 (test) |
| thermalDogsAndPeople | `thermalDogsAndPeople/` | 142 | 181 | 2 | 28.2 | 20 (test) |

評価集合の合計は 4,841 枚。**2000 iter × batch 2 = 4000 枚**は全タスク共通なので、小さいタスクは
数十周、PascalVOC は 1 周未満になる。これは公式設定どおりで、本実験では変更しない。

NorthAmericaMushrooms の「評価が v2 の train」は公式コードの登録どおり
（`common/data/odinw/NorthAmericaMushrooms.py`）。誤りではないのでそのまま踏襲する。

## 4. 逐次順

論文・公式コードは **ODinW-13 をランダム順で逐次学習（seed 42、3 seed 平均）**（実装ノート §6）。
**順序の生成方法まで公式と一致させることはできない**（実装ノートに順序そのものの記載がなく、
同じ seed でもシャッフル実装が違えば並びが変わる）。本実験では次で固定する。

```python
import random
tasks = [odinw13 config の記載順の13タスク名]
random.Random(seed).shuffle(tasks)
```

**まず seed 42 の1本だけ実行**する。順序依存が疑われたときに 43 / 44 を追加する。

## 5. 評価・判定基準

**各タスク終了後（Rep+ 適用後）に、そこまでに学習済みのタスクと ZCOCO を評価する**
（2026-08-04 指定）。タスク t の後に評価するのは、逐次順で 1〜t 番目のタスクの valid と
COCO2017-val の計 t+1 個である。未学習タスクは評価しない。

- 報告値: **Avg**（最終時点の13タスク mAP の平均）、**ZCOCO**、参考に **hAP**（両者の調和平均。
  論文の指標）。加えて各時点の「学習済みタスクごとの mAP」と「ZCOCO」の推移を記録する。
- 各タスクの mAP は「学習直後」と「その後の各時点」の両方が残るので、**タスク単位の忘却量**が
  そのまま読める（平均化しない）。ZCOCO も毎時点あるので、事前学習知識の減り方が推移で見える。


### 5.1 起点（θ0）のタスク別ゼロショット性能（公式評価集合。判定はこの値からの変化で見る）

2026-08-05 実測（`experiments/exp_034/eval/odinw_theta0/`、1 GPU、`odinw13_official_eval.py`）。
論文 Tab.1 "Original Model"（GDINO-T、同じ評価集合）を併記する。

| タスク | **θ0（MM-GDINO）** | 論文の GDINO-T | 差 |
|---|---:|---:|---:|
| AerialMaritimeDrone | 0.1710 | 0.1910 | -0.020 |
| Aquarium | 0.2650 | 0.2075 | +0.058 |
| CottontailRabbits | 0.7180 | 0.6472 | +0.071 |
| EgoHands | 0.5940 | 0.5696 | +0.024 |
| NorthAmericaMushrooms | 0.2190 | 0.2539 | -0.035 |
| Packages | 0.5780 | 0.5449 | +0.033 |
| PascalVOC | 0.5660 | 0.5480 | +0.018 |
| pistols | 0.6910 | 0.6597 | +0.031 |
| pothole | 0.2820 | 0.2208 | +0.061 |
| Raccoon | 0.6640 | 0.6222 | +0.042 |
| ShellfishOpenImages | 0.5380 | 0.3284 | +0.210 |
| thermalDogsAndPeople | 0.5950 | 0.7062 | -0.111 |
| VehiclesOpenImages | 0.5810 | 0.5715 | +0.010 |
| **平均** | **0.4971** | **0.4670** | **+0.030** |

ZCOCO の θ0 は 0.504（exp_001 実測）、論文の GDINO-T は 0.4737。
ベースモデルが違うので絶対値は一致しないが、mmdet の評価集合での 0.5330 ではなく
論文と同水準（0.4971 対 0.4670）になっており、**評価集合が公式側に切り替わっていることの
裏付けになる**。

適応の余地はタスクによって大きく違う。AerialMaritimeDrone 0.171 / NorthAmericaMushrooms 0.219 /
Aquarium 0.265 / pothole 0.282 は低く伸びしろが大きい。CottontailRabbits 0.718 / pistols 0.691 /
Raccoon 0.664 は既に高い。

### 5.2 参考記録: mmdet の評価集合（valid）での θ0

**判定には使わない**（§3 で公式版に切り替えたため）。評価系そのものが上流と一致することの
確認記録として残す。（2026-08-03 実測、
`experiments/odinw13_setup_check/console.log`。θ0 ＋ odinw13 評価 config）。
上流公表値（`configs/mm_grounding_dino/README.md` の MM-GDINO-T (O365,GoldG,GRIT,V3Det) 列）を
併記する。13タスク中10タスクが完全一致、残り3タスクも差 0.001 で、評価系が正しいことの確認になる。

| タスク | **θ0 mAP（実測）** | mAP_50 | 上流公表値 | 差 |
|---|---:|---:|---:|---:|
| AerialMaritimeDrone | 0.1500 | 0.2910 | 0.151 | -0.001 |
| Aquarium | 0.2830 | 0.4730 | 0.283 | 0.000 |
| CottontailRabbits | 0.7860 | 0.9120 | 0.786 | 0.000 |
| EgoHands | 0.5200 | 0.8250 | 0.519 | +0.001 |
| NorthAmericaMushrooms | 0.7670 | 0.7740 | 0.767 | 0.000 |
| Packages | 0.7060 | 0.8320 | 0.706 | 0.000 |
| PascalVOC | 0.5660 | 0.6680 | 0.566 | 0.000 |
| pistols | 0.7290 | 0.9060 | 0.729 | 0.000 |
| pothole | 0.2430 | 0.3880 | 0.243 | 0.000 |
| Raccoon | 0.5350 | 0.9190 | 0.535 | 0.000 |
| ShellfishOpenImages | 0.4880 | 0.5850 | 0.488 | 0.000 |
| thermalDogsAndPeople | 0.5420 | 0.7180 | 0.542 | 0.000 |
| VehiclesOpenImages | 0.6140 | 0.7530 | 0.615 | -0.001 |
| **平均** | **0.5330** | — | **0.533** | 0.000 |

ZCOCO の θ0 は **0.504**（exp_001 実測）。

適応の余地はタスクによって大きく違う。pothole 0.243 / AerialMaritimeDrone 0.150 /
Aquarium 0.283 は低く伸びしろが大きい一方、CottontailRabbits 0.786 / NorthAmericaMushrooms 0.767 /
pistols 0.729 は既に高く、上げ幅は小さいと見込まれる。Avg の上昇は前者が牽引する可能性が高い。

### 判定

**論文の要点は「Avg が大きく上がる一方で ZCOCO はほとんど下がらない」という非対称**である。
これが再現されるかを見る。

| | 論文（GDINO-T 基準） | **exp_034 の起点（θ0）** |
|---|---|---|
| ゼロショット Avg | 46.70（論文の評価集合） | **53.30**（実測・mmdet の評価集合。タスク別は §5.1） |
| ゼロショット ZCOCO | 47.37 | **50.4**（実測） |
| ZiRa 後 | Avg 59.73（**+8.3**） / ZCOCO 46.06（**−1.3**） | Avg は上昇、ZCOCO はほぼ保持 |

実装に問題がある可能性が高いと考える兆候は次の2つ。

- **Avg がゼロショット 53.30 から上がらない** → 適応が効いていない（RDB が学習されていない等）。
- **ZCOCO が大きく落ちる** → ZiL / Rep+ が効いていない。実装ノート §7 に「ZiL を外すと
  ZCOCO が 46.09 → 39.72 に崩れる」とあり、**ZCOCO の落ち方が ZiL の動作確認になる**。

**数値の閾値は設けない。判定はユーザーが行う。**

## 6. 実装

### 6.1 dn の無効化には**コード追加が必要**（config だけでは無理）

上流の mmdet では dn を config から切れない。確認した事実は次のとおり。

- `mmdet/models/detectors/dino.py:44` は `dn_cfg` が None でも `CdnQueryGenerator(**dn_cfg)` を
  呼ぶ（`if dn_cfg is not None:` は 34〜43 行の assert / 値埋めだけを囲っている）。
  → `model=dict(dn_cfg=None)` は TypeError で落ちる。
- `group_cfg` の `num_dn_queries=0` でも無効化にならない。
  `dino_layers.py` の `get_num_groups` は `num_groups < 1` のとき 1 に引き上げる。
- `mmdet/models/dense_heads/dino_head.py:462` は `dn_meta['num_denoising_queries']` を
  **None チェック（463行）より前に**読む。→ `dn_meta=None` を渡すとここで落ちる（上流のバグ）。

**上流ファイルは無変更**とし、次の3点で対応する（2026-08-04 にユーザー承認、実装済み）。

1. `mmdet/models/layers/nodn_query_generator.py`（新規）
   — `CdnQueryGenerator` を継承し `__call__` だけを差し替えた `NoDNQueryGenerator`。
   長さ0の dn クエリと `attn_mask=None` / `dn_meta=None` を返す。上流 `pre_decoder`
   （`grounding_dino.py:388-393`）は `torch.cat` するだけなので長さ0で dn 無しと等価になる。
   親のパラメータ構成（`label_embedding`）は保持するので state_dict のキーは変わらない。
2. `mmdet/models/dense_heads/nodn_grounding_dino_head.py`（新規）
   — `GroundingDINOHead` を継承した `NoDNGroundingDINOHead`。`split_outputs` の
   None チェック順序だけを直す。`DINOHead.loss_by_feat` は denoising 側が None なら
   dn 損失を飛ばすよう既に書かれているため、修正はこの1メソッドで足りる。
3. `mmdet/models/detectors/zira_grounding_dino.py`（既存の自作ファイルに追記）
   — `use_dn: bool = True` を追加。False のとき `dn_query_generator` を
   `NoDNQueryGenerator` に差し替える。**既定は True なので exp_020 / exp_023 の挙動は不変**。

**検証（2026-08-04、実データ1バッチを `model.loss()` → `backward()` まで通して確認）**
ODinW-13 の Aquarium train（7クラス）を `CocoDataset` で読み、batch 2 で実行した。

| | 損失項数 | dn 損失項 | `dn_query_generator` | `bbox_head` |
|---|---|---|---|---|
| `use_dn=False`（exp_034） | **22**（検出21＋`loss_zil`） | **0** | `NoDNQueryGenerator` | `NoDNGroundingDINOHead` |
| `use_dn=True`（既定＝exp_023） | 40（検出39＋`loss_zil`） | 18 | `CdnQueryGenerator` | `GroundingDINOHead` |

いずれも backward が通り、勾配が付くのは RDB の25パラメータのみ（`use_dn=False` で
`text_feat_map.llrb.weight` の grad norm 22.55、`neck.rdb_extra_convs.0.llrb.weight` 17.96）。

### 6.2 新規に作るもの

- `experiments/exp_034/configs/zira_odinw13_base.py`
  — exp_023 の `zira_replayfree_base.py`（lr 1e-3 / 2000 iter / iter800 減衰 / llrb lr_mult 0.2 /
  λ=0.1 / ckpt last）を継承し、`use_dn=False`、新 head、`batch_size=2`、`randomness=dict(seed=0)` を設定。
- `experiments/exp_034/configs/odinw13_official_eval.py`
  — `configs/mm_grounding_dino/odinw/…odinw13.py` をコピーし、13タスクの `data_root` /
  `ann_file` / `data_prefix` を公式版（§3 の表）に差し替えたもの。`class_name` と
  `MultiDatasetsEvaluator` はそのまま流用する。**コード変更は不要**（2026-08-05 確認）。
- `experiments/exp_034/configs/zira_odinw13_{task}.py`（13本）
  — タスクごとの `data_root` / `ann_file` / `data_prefix` / `metainfo` と COCO 用 train_pipeline。
- `experiments/exp_034/run_sequential_zira_odinw13.sh`
  — exp_023 の `run_sequential_zira.sh` と同型。順序リストを seed 42 で生成した13タスクにし、
  各タスク後に `experiments/exp_020/merge_hlrb.py` で Rep+ → 融合後 ckpt を次タスクの
  `load_from` にする → **学習済みタスク（1〜t 番目）の valid と ZCOCO を評価**（§5）。
  評価用 config は `configs/mm_grounding_dino/odinw/…odinw13.py` の 13 データセット定義から
  学習済み分だけを取り出して `MultiDatasetsEvaluator` に渡す形にする（評価側の設定は流用）。
  exp_023 は `dist_train.sh` / `dist_test.sh`（4 GPU）だったが、**exp_034 は学習・評価とも
  `tools/train.py` / `tools/test.py` を単一プロセス・1 GPU で呼ぶ**（§7）。

### 6.3 流用（無変更）

`mmdet/models/detectors/zira_grounding_dino.py`、`mmdet/models/layers/zira_layers.py`、
`mmdet/models/necks/zira_channel_mapper.py`、`experiments/exp_020/merge_hlrb.py`、
`configs/mm_grounding_dino/odinw/grounding_dino_swin-t_pretrain_odinw13.py`、
`configs/mm_grounding_dino/eval_base_coco.py`。

### 6.4 実行前の実装検証

1. **学習対象が RDB のみ**であること（`requires_grad=True` が hlrb / llrb / scaling だけ）。
   → **確認済み（2026-08-04）**。25パラメータのみに勾配が付く。
2. **dn が無効**であること（`dn_loss_*` と `d*.dn_loss_*` が現れない＝40項→22項）。
   → **確認済み（2026-08-04）**。§6.1 の表。
3. **LLRB の実効 lr が 2e-4**、HLRB が 1e-3 であること（optimizer の param group を直接確認）。
   → config 作成後に確認する。
4. **`loss_zil` が計算され backward が通る**こと。
   → **確認済み（2026-08-04）**。dn 無効・COCO 形式で `loss_zil` を含む22項が計算され
   backward が通った。optimizer step までの確認は config 作成後に行う。
5. **Rep+ が forward を保存する**こと（融合前後で同一入力に対する出力が一致）。
   `merge_hlrb.py` は exp_020 で検証済みだが、dn 無効・COCO 形式でも再確認する。


### 6.5 公式実装との差分精査（2026-08-04、公式リポジトリの実コードと突合）

公式 https://github.com/JarintotionDin/ZiRaGroundingDINO を取得し、ZiRa の実行経路
（`train_odinw13_zira.sh` → `train_multidatasets.py --config-file test_odinw13_softfreeze
--model-config-file groundingdino/config/GroundingDINO_SwinT_OGC_rep.py --num-gpus 2 --seed 42
--shuffle-tasks`）を辿って1項目ずつ確認した。

#### (1) 一致を確認した項目（公式のファイル:行を根拠として明記）

| 項目 | 公式 | 自実装 |
|---|---|---|
| HLRB 初期値 | `zero_value=1e-8`、`nn.init.constant_(self.weight, 1e-8)`（`groundingdino_dual_zero_rep_branch.py:60,69-72,105-108`） | `HLRB_INIT=1e-8`（`zira_layers.py:24,54`） |
| LLRB 初期値 | `freeze_conv/freeze_linear` を 0（同 79-81,111-113） | 0（`zira_layers.py:57-59`） |
| s 初期値 | `lan_scale=vis_scale=0.1`（同 61-62） | `SCALING_INIT=0.1`（`zira_layers.py:25`） |
| 学習時 forward | `branch = s*HLRB(x)`、`out = branch + LLRB(x)`（同 87-90,119-122） | 同一（`zira_layers.py:80-84,134-135`） |
| ZiL | `SmoothL1(reduction='mean')` の2項 `ZiL(branch)+ZiL(out)`（同 85,91-93,117,123-125） | 同一（`zira_layers.py:28-30,83,137`） |
| λ | `loss_adapter_weight=0.1`、conv 側と linear 側に別々に乗算（`GroundingDINO_SwinT_OGC_rep.py` 末尾、同 584-587） | `zil_loss_weight=0.1` を総和に乗算（`zira_grounding_dino.py`）。総和は同一 |
| 挿入位置（視覚） | `input_proj[l][1](input_proj[l][0](x) + adapter(x))`＝GN の前に加算（同 489-495,505-522） | `cm.conv(x)+rdb(x)` → GN（`zira_channel_mapper.py:50-53`）。MM-GDINO の neck は `act_cfg=None` で公式の `Sequential(Conv2d, GroupNorm)` と同構成 |
| 挿入位置（言語） | `feat_map(bert_out) + rep_linear_adapter(bert_out)`（同 460-464） | `ZiRaLinear.forward` の `base(x)+rdb(x)` |
| extra level の入力 | `features[-1].tensors`（射影前のバックボーン出力） | `inputs[-1]`（`zira_channel_mapper.py:65`） |
| Rep+ | `__rep__`: `freeze_w += w*s`、`w←1e-8`、`s←0.1`（同 97-103,129-135）。各タスクの `after_train` で実行され、`model_final.pth` 保存の前（`train_multidatasets.py:221-237`） | `merge_hlrb.py:24-47`。学習後に適用し、融合後 ckpt を次タスクの `load_from` にする |
| 学習対象 | `before_train`: 全凍結 → 名前に `adapter` を含む param のみ解凍（同 721-734。`freeze_model_()` は 737 でコメントアウト＝LLRB も学習される） | `hlrb/llrb/scaling` のみ `requires_grad=True`（25 param） |
| η（LLRB の lr 比） | `lr_factor_func = 0.2 if "freeze" in module_name else 1`（`test_odinw13_softfreeze/for_train/*.py:24`） | `custom_keys={'llrb': dict(lr_mult=0.2)}` |
| optimizer | AdamW, lr 1e-3, wd 1e-4（同 25-26、`common/optim.py`） | 同一 |
| スケジュール | `max_iter=10*200=2000`、`modified_coco_scheduler(10,4,base_steps=200)` → `values=[1.0,0.1] / milestones=[800,2000]`、warmup 長 0（`coco_schedule.py:105-125`） | `MultiStepLR(milestones=[800], gamma=0.1)`、warmup なし |
| batch | `dataloader.train.total_batch_size = 2`（detectron2 では**全 GPU 合計**）。`--num-gpus 2` なので 1 枚/GPU | 1 GPU × 2 = 合計 2 |
| grad clip | `max_norm=0.1, norm_type=2`（同 15-17） | base から継承（`grounding_dino_swin-t_pretrain_obj365.py:219`） |
| dn | `dn_number=0` を build で直接指定（同 799）。forward でも `input_query_bbox=input_query_label=attn_mask=dn_meta=None`（同 533） | `use_dn=False`（§6.1。損失22項を実測確認） |
| AMP / EMA | ともに無効（`common/train.py`） | mmengine `OptimWrapper`（AMP なし）、EMA なし |
| 空アノテーション画像 | `filter_empty=False` | config で `filter_cfg=dict(filter_empty_gt=False, min_size=32)` とする。`min_size=32` で落ちる画像は13タスクとも **0 枚**（実測）なので影響なし |
| データ拡張 | RandomFlip ＋ 多スケール resize、または resize(400/500/600)→RandomCrop(384,600)→多スケール resize（`common/data/odinw/*.py`） | mmdet の GDINO 既定 train_pipeline が同一構成 |

#### (2) 記録済みの意図的な差分（実験の主旨に影響しないと判断したもの）

1. **ベースモデル**（§1）。公式 GDINO-T (Cap4M) 対 MM-GDINO-T。最大の差分。
2. **評価時 forward**。公式は eval で LLRB のみ（同 95,127）、自実装は常に全和。Rep+ 後は
   HLRB=1e-8・s=0.1 なので寄与は 1e-9 オーダーで無視できる。
3. **言語側 HLRB の bias 初期値**。公式 `RepZeroLinear` は `self.weight` のみ 1e-8 にし、
   **bias は nn.Linear 既定の乱数のまま**（同 105-113。conv 側は bias も 1e-8）。自実装は
   bias も 1e-8。公式は eval で HLRB を通さないため乱数 bias が評価に出ないが、自実装は
   全和 forward なので 1e-8 にしないと差が出る。2 と整合させるための選択。
4. **逐次順の生成方法**。公式は `glob.glob(...)` の結果を `random.shuffle`（`train_multidatasets.py:481-484`）。
   glob の順序はファイルシステム依存で再現できない（§4）。
5. **評価の頻度**。公式は全タスク終了後にまとめて評価（同 508-538）。本実験は各タスク後に
   学習済み分と ZCOCO を評価する（§5、2026-08-04 指定）。測定を増やす方向の差分。

#### (3) 新たに判明した差分（判断が必要）

**公式が使う ODinW-13 のサブセット版と split が、本リポジトリの odinw13 評価 config と違う。**

| タスク | 公式の train | 公式の eval | 現行 design の eval |
|---|---|---|---|
| AerialMaritimeDrone | **tiled**/train (371) | **tiled/test** (32) | large/valid (15) |
| Aquarium | 同 (448) | **test** (63) | valid (127) |
| CottontailRabbits | 同 (1,980) | **test** (10) | valid (19) |
| EgoHands | 同 (3,840) | **test** (480) | valid (480) |
| NorthAmericaMushrooms | v1/train (41) | **v2-416x416augmented の train** (246) | v1/valid (5) |
| Packages | **augmented-v1**/train (95) | **augmented-v1/test** (3) | Raw/valid (4) |
| PascalVOC | 同 (13,690) | valid (3,422) | valid (3,422) ＝**一致** |
| Raccoon | **v38-416x416-resize**/train (150) | **v38/test** (17) | v2-raw/valid (29) |
| ShellfishOpenImages | **416x416**/train (406) | **416x416/test** (58) | raw/valid (116) |
| VehiclesOpenImages | 同 (878) | **test** (126) | valid (250) |
| pistols | 同 (2,377) | **test** (297) | val (297) |
| pothole | 同 (465) | **test** (67) | valid (133) |
| thermalDogsAndPeople | 同 (142) | **test** (20) | valid (41) |

**2026-08-05 決定: 学習・評価とも公式（論文）側に合わせる。**確定した内容は §3 の表。
コード変更は不要で、評価 config 1本の差し替えと θ0 の再測定（約14分）で済む。
θ0 の起点値（§5.1 の平均 0.5330）は mmdet の評価集合の値なので、判定には使わない。

### 6.6 学習ハイパーパラメータの三者照合（論文 / 公式コード / exp_034）

論文本文 §4.1（`papers/ZiRaGroundingDINO.pdf`、実装詳細の段落）と公式コードの実値を並べる。

| 項目 | 論文 | 公式コード | **exp_034** | 判定 |
|---|---|---|---|---|
| 学習率 | 1e-3 | `optimizer.lr = 0.001` | 1e-3 | 一致 |
| optimizer | AdamW | `torch.optim.AdamW`, betas (0.9, 0.999) | AdamW, mmengine 既定 betas (0.9, 0.999) | 一致 |
| weight decay | 1e-4 | `optimizer.weight_decay = 1e-4` | 1e-4 | 一致 |
| batch size | 2 | `total_batch_size = 2`（detectron2 は全 GPU 合計） | 2（1 GPU） | 一致 |
| GPU 枚数 | 2×RTX3090 | `--num-gpus 2`（1 枚/GPU） | 1×A100（2 枚/GPU） | 実効バッチ同じ |
| 学習長 | **2 epoch** | **2000 iter 固定**（`max_iter = 10 × 200`） | 2000 iter | 論文とコードが不一致。**コード準拠** |
| lr decay | 1 epoch 後に ×0.1 | iter **800** で ×0.1（`values=[1.0,0.1]`, `milestones=[800,2000]`） | iter 800 で ×0.1 | コード準拠 |
| warmup | 記載なし | `warmup_epochs=0` → warmup 長 0 | なし | 一致 |
| grad clip | 記載なし | `max_norm=0.1, norm_type=2` | 同（base から継承） | 一致 |
| η（LLRB の lr 比） | 「0.1 か 0.2」が最適、**主結果の表は η=0.20**（Tab.7 で 46.06/59.73＝Tab.1 と同値） | `0.2 if "freeze" in module_name`（softfreeze config） | 0.2（`llrb` に lr_mult 0.2） | 一致 |
| λ（ZiL 重み） | hAP 最良は 0.05 だが、**主結果は λ=0.10**（Tab.6 の 0.10 行が 46.06/59.73＝Tab.1 と同値） | `loss_adapter_weight = 0.1` | 0.1 | 一致 |
| RDB 初期化 | 「RDB 全体をゼロで初期化」 | HLRB 1e-8 / LLRB 0 / s 0.1 | コードと同じ | コード準拠 |
| AMP | 記載なし | `amp.enabled=False` | なし（`OptimWrapper`） | 一致 |
| EMA | 記載なし | `model_ema.enabled=False` | なし | 一致 |
| 検出損失の重み | 記載なし | cls 1.0 / bbox 5.0 / giou 2.0（`criterion/__init__.py:23-27`） | cls 1.0 / bbox 5.0 / giou 2.0（`loss_iou` は `DeformableDETRHead` 既定 2.0） | 一致 |
| focal α / γ | 記載なし | 0.25 / 2.0 | 0.25 / 2.0 | 一致 |
| num_queries | 記載なし | 900 | 900 | 一致 |
| max_text_len | 記載なし | 256 | 256 | 一致 |

#### ベース実装の違いから来る差（ZiRa のハイパラではないが学習に効く）

1. **マッチングコストの重み**。公式 ZiRa リポジトリの `build_matcher(args)` は引数を無視して
   `HungarianMatcher()` を返すため、cost は **class 1 / bbox 1 / giou 1**（`matcher/matcher.py:58-64` の既定値）。
   MM-GDINO は **focal 2.0 / L1 5.0 / giou 2.0**（`grounding_dino_swin-t_pretrain_obj365.py:113-118`）。
   base を凍結していてもマッチング結果は RDB の勾配に効く。MM-GDINO 側を変えると本体の学習規約から
   外れるため、**現状は MM-GDINO のまま**とする。
2. **評価時の box 数**。公式 `select_box_nums_for_evaluation=200`、MM-GDINO は `max_per_img=300`。
3. **seed**。公式は `--seed 42` で順序も学習も 42。本実験は順序に 42、学習の `randomness` は
   **seed=0**（プロジェクト規約。2026-08-05 に seed=0 で確定）。

## 7. コスト

**学習・評価とも 1 GPU のみを使い、タスクの並列実行はしない。**以下はすべて 1 GPU 換算。

- 学習: exp_023 の ZiRa 実測（4 GPU × batch 4、約 1.0 s/iter ＝ 1 GPU あたり 4 img/s）から、
  batch 2 では **約 0.5 s/iter → 2000 iter で 15〜20分/タスク**、13タスクで **約 3.5〜4.5 時間**。
- ODinW-13（公式版・学習済み分のみ）: 評価集合は13タスク合計 4,841 枚。seed 42 の順序では
  累積枚数が 297 → 3,719 → 3,729 → 3,746 → 3,872 → 3,875 → 3,895 → 3,962 → 4,442 →
  4,688 → 4,720 → 4,783 → 4,841 で、**13回分の合計は 50,569 枚**。
  ゼロショット実測（2026-08-03、4 GPU、1 rank 1235 iter、平均 0.1357 s/iter）から
  1 GPU で 0.136 s/枚 として **約 1.9 時間 ＋ 起動・集計 13回分 約 0.6 時間 ＝ 約 2.5 時間**。
- ZCOCO を毎タスク後に13回: exp_017 の実測（4 GPU、1 rank 1250 iter ＝ 5,000 枚、0.18 s/iter、
  04:25:54 開始 → 04:29:54 推論終了 → 04:32:08 集計終了）から 1 GPU で
  5,000 × 0.18 ≈ 15分 ＋ 集計・起動 3.5分 ＝ **約 18.5分/回 × 13 ＝ 約 4.0 時間**。
- **合計 約 10.6 時間**（学習 4.0h ＋ ODinW 2.6h ＋ ZCOCO 4.0h）。
- ディスク: 各タスクの last と Rep+ 後で 26 個 × 約 2.1GB ＝ **約 55GB**。

参考: seed 42 では PascalVOC（3,422枚、全体の 69%）が2番目に来るため、累積評価と全13評価の
コスト差は小さい（51,626 枚 対 13×4,938＝64,194 枚、時間差は約 0.5 時間）。全13を毎回
評価すれば未学習タスクの forward transfer も取れるが、そこまでは指定されていないので
累積のみとする。

## 8. この実験で言えないこと

- **ベースモデルが論文と違うため、絶対値の一致では判定できない**（§1）。言えるのは相対変化の
  向きと大きさが整合するかまで。
- **逐次順が論文と同一である保証がない**（§4）。数値差の一部は順序の違いで説明され得る。
- 1 seed だけでは、順序による変動と実装の問題を切り分けられない。
- 学習は非決定的である（同一 seed でも一致しない。`MultiScaleDeformableAttention` の backward が
  atomicAdd を使うため。2026-08-02 に実測）。小さい差の解釈にはこれを考慮する。
- 負例サンプリングを行わない（COCO 形式）ので、**exp_023 の ZiRa 結果（ODVG）とは直接比較できない**。

## 9. 成果物

- `experiments/exp_034/zira_s42_t{01..13}_{task}_work_dir/`（学習ログ・last ckpt）
- `experiments/exp_034/zira_s42_t{01..13}_merged/`（Rep+ 後 ckpt）
- `experiments/exp_034/eval/odinw_after_t{01..13}/`（各時点で学習済みタスクのみ）
- `experiments/exp_034/eval/zcoco_after_t{01..13}/`（各時点の ZCOCO）
- `experiments/exp_034/results/`（事実記録のみ。解釈・考察は書かない＝行動原理8）
