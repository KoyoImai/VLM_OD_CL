# exp_040 設計書: 後半3ドメイン（Aerial → Microscopic → Documents）の逐次学習

作成 2026-08-14。**実行前のユーザー承認ゲート**（行動原理1・3）。
実験ノート: [[../../experiment_notes/note11]]。
前半3ドメインの実験: [[../exp_024/design]]（リプレイフリー）、[[../exp_027/design]]（リプレイ）、
[[../exp_035/design]]（λ=10 蒸留E＋リプレイ）。

## 0. 位置付け

前半3ドメイン（underwater → electromagnetic → videogames）は exp_024 / exp_027 / exp_035 で
測り終えている。本実験はその **3タスク目終了時のパラメータを初期値**として、残る3ドメインを
逐次学習する。6ドメインの逐次という長い系列で、各手法の適応と忘却がどう推移するかを見る。

## 1. 目的

note11 の3条件について、後半3ドメイン（Aerial → Microscopic → Documents）を逐次学習し、
学習済み6ドメインと ZCOCO を評価する。

**仮説と判定基準は実験ノート（note11）でユーザーが確定する。** 本書は測るものと条件を定める。

## 2. 条件

### 2.1 3条件と初期パラメータ

| # | 条件 | 初期パラメータ（t=3 終了時） |
|---:|---|---|
| 1 | リプレイフリー（full FT） | `experiments/exp_024/fullft_replayfree_videogames_work_dir/epoch_20.pth` |
| 2 | リプレイ（full FT） | `experiments/exp_027/fullft_replay_videogames_work_dir/epoch_20.pth` |
| 3 | λ=10 蒸留E ＋ リプレイ | `experiments/exp_035/kdE_condA_l2w100_videogames_work_dir/epoch_20.pth` |

3本とも存在を確認済み（各 work_dir に `epoch_20.pth`、`last_checkpoint` も `epoch_20.pth` を指す）。

note11 の記載は 2026-08-14 に `epoch_20.pth` へ修正済み（3条件とも一致）。

### 2.2 ドメイン順序とデータ

```
（前半: underwater → electromagnetic → videogames）→ Aerial → Microscopic → Documents
                                                      t=4        t=5           t=6
```

| ドメイン | 画像 | box | クラス |
|---|---:|---:|---:|
| aerial | 6,643 | 35,840 | 22 |
| microscopic | 9,576 | 74,208 | 28 |
| documents | 17,866 | 126,166 | 59 |

ODVG 版（`*_train_od.json` ＋ `*_label_map.json`）も3ドメインとも存在を確認済み。

### 2.3 学習設定（前半と完全に同一）

`experiments/exp_023/configs/fullft_replay_base.py` の値。exp_024 / exp_027 / exp_035 が
これを継承しているので、後半も同じ土台に載せる。

| 項目 | 値 |
|---|---|
| データセット | `ODVGDataset` ＋ `RandomSamplingNegPos` |
| num_classes | 256 |
| epoch | 20、`MultiStepLR` milestones=[15]、gamma=0.1 |
| optimizer | AdamW lr 1e-4 / weight_decay 1e-4、`backbone`・`language_model` は lr_mult=0.1 |
| grad clip | max_norm 0.1（L2） |
| batch size | リプレイ有り 6、リプレイ無し 4 |
| dn | 有効 |
| seed | 0 |
| in-training val | 行わない（評価は逐次ドライバ） |

### 2.4 リプレイのバッファ構成

条件2・3 は［現在, 参照, 過去プール］の**3ソース構成**、`source_ratio=[4,1,1]`、batch 6
（現在:バッファ = 2:1）。前半（exp_023 / exp_027 / exp_035）と同じ方式で、2026-08-14 に
この構成のまま進めることを確定した。

| ソース | 中身 | 1バッチあたり |
|---:|---|---:|
| 0 現在ドメイン | 当該タスクの train | 4 枚 |
| 1 参照バッファ | **Objects365 1,000 枚（全タスク共通・固定）** | 1 枚 |
| 2 過去プール | 学習済みドメインの各 500 枚を `ConcatDataset` で1ソースに | 1 枚 |

**Objects365 は過去プールには入れず独立したソースとして保つ。**これにより毎バッチ必ず
Objects365 が 1 枚・過去ドメインが 1 枚という比率が保証される（1ソースにまとめると
o365 が枚数比で過半を占め、過去ドメインが 0 枚のバッチが生じる）。

**過去プールは累積**（2026-08-14 決定）。学習済みドメインをすべて入れ、タスクが進むごとに
追加していく。参照バッファは前半と同じ `reference_o365v1_1000.odvg.json`（Objects365 1,000 枚）
で固定。

| t | ドメイン | 過去プール（各 500 枚） | 参照 |
|---:|---|---|---|
| 4 | aerial | underwater, electromagnetic, videogames | o365 1,000 |
| 5 | microscopic | ＋ aerial（計4ドメイン） | o365 1,000 |
| 6 | documents | ＋ microscopic（計5ドメイン） | o365 1,000 |

バッファ定義は `exp_023/buffer/` に7ドメイン分すべて存在する（実在を確認済み）。

`source_ratio=[4,1,1]` の3ソース構成は前半と同じで、過去プールは複数ドメインの
`ConcatDataset` として1ソースにまとめる（前半 t=3 で underwater 500 ＋ electromagnetic 500 を
1ソースにしていたのと同じ方式）。したがってプール内のドメイン数が増えても、
1バッチあたりの過去画像は 1 枚で一定である。

### 2.5 蒸留の設定（条件3）

exp_035 の `kdE_condA_l2w100` と同一。

| 項目 | 値 |
|---|---|
| 検出器 | `KDGroundingDINO` |
| 蒸留対象 | `['img', 'txt', 'fus']`（画像・テキスト・融合の3特徴） |
| 損失 | `FeatureDistillLoss`（L2） |
| λ | **10.0** |
| `num_buffer_per_batch` | 2 |
| 教師 | θ_{t-1}（直前ドメインの学習結果）。ドライバが `--cfg-options teacher_ckpt=` で渡す |

## 3. 判定材料

**最終的な判定はユーザーが行う。** 数値の閾値は設けない。

1. 各ドメインの適応 mAP（学習直後）と最終値（t=6 時点）、忘却量。
2. ZCOCO の推移（t=3 の値を起点として t=4〜6）。
3. 3条件の対照。
4. 前半3ドメインの推移と接続したときの、6ドメイン通しての傾向。
5. 前半で学習した3ドメイン（underwater / electromagnetic / videogames）が後半でどう変化するか。

## 4. 手順

各ドメイン t（4→5→6）について次を繰り返す。

1. 学習（20 epoch）。`load_from` は θ_{t-1}（t=4 は §2.1 の初期パラメータ）。
   条件3 は `teacher_ckpt` にも θ_{t-1} を渡す。
2. 評価: 学習済み6ドメインのうち t までのものと ZCOCO。

## 5. 実行環境とコスト

**MPRG クラスタで実行する**（exp_027 / exp_028 / exp_039 と同じ運用）。本環境では実行しない。

### 5.1 クラスタ側の前提

exp_039 の README §2 と同じ。`a6000_ada`（A6000 Ada ×4、48GB）、`--exclude=node03`、
Singularity、`/local_cache/${SLURM_JOB_ID}` へのステージング、Objects365 と `/dataset01` の bind。

**exp_036 / exp_039 が動いている場合の同時実行4ジョブ制限に注意。**

### 5.2 所要時間

exp_027（リプレイ有り full FT、batch 6、20 epoch）の実測から、画像数に比例して見積もる。
underwater 12,633 枚で 8.9 h、electromagnetic 25,398 枚で 16.6 h だったので約 0.7 ms/枚/epoch。

| ドメイン | 画像 | リプレイ有り | リプレイ無し |
|---|---:|---:|---:|
| aerial | 6,643 | 約 4.7 h | 約 2.5 h |
| microscopic | 9,576 | 約 6.7 h | 約 3.6 h |
| documents | 17,866 | 約 12.5 h | 約 6.7 h |
| 計 | | **約 24 h** | **約 13 h** |

| 条件 | 学習 | 評価 | 小計 |
|---|---:|---:|---:|
| 1 リプレイフリー | 13 h | 約 6 h | 19 h |
| 2 リプレイ | 24 h | 約 6 h | 30 h |
| 3 蒸留E＋リプレイ（教師の forward 分 増） | 約 30 h | 約 6 h | 36 h |
| **合計** | | | **約 85 GPU 時間** |

評価は t=4 で4ドメイン、t=5 で5、t=6 で6 ＋ 各時点の ZCOCO。

**3ジョブに分割して並列投入する**（2026-08-14 決定）。実時間は約 36 時間（条件3 が律速）。

| ジョブ | 内容 | 見積もり |
|---|---|---:|
| `exp040_replayfree` | 条件1 の3ドメイン | 約 19 h |
| `exp040_replay` | 条件2 の3ドメイン | 約 30 h |
| `exp040_kdE` | 条件3 の3ドメイン | 約 36 h |

`--time` は余裕を見て 72:00:00 とする。クラスタの同時実行は4ジョブまでなので、
exp_039 の稼働状況によっては投入を待つ必要がある（§7）。

## 6. 実装

### 6.1 既にあるもの

`ODVGDataset`、`CurrentEpochMultiSourceSampler`、`KDGroundingDINO`、`FeatureDistillLoss`、
`exp_023/buffer/` の7ドメイン分のバッファ定義、後半3ドメインのデータ（ODVG 版含む）。
評価 config は exp_023 / exp_026 のものを流用できるか要確認（§6.3）。

### 6.2 作るもの

| 成果物 | 内容 |
|---|---|
| `configs/{replayfree,replay,kdE}_{aerial,microscopic,documents}.py` | 学習 config 9本。前半のドメイン config を雛形に、データと過去プールだけ差し替える |
| `configs/eval_{aerial,microscopic,documents}.py` | 後半3ドメインの評価 config（前半3ドメインは exp_023 のものを流用） |
| `run_sequential.sh` | 3条件を回す逐次ドライバ。条件を引数で切り替え |
| `sbatch_*.sh` | クラスタ投入 |
| `check_exp040_setup.py` | 実行前検証 |

### 6.3 実行前に確認する項目

1. 9本の config が build でき、学習設定が前半（exp_024 / exp_027 / exp_035）と一致
2. 初期パラメータ3本がロードでき、`load_from` が正しく解決すること
3. データ経路（ODVGDataset・サンプラ・source_ratio・過去プールの中身）
4. 条件3 の蒸留設定（targets / λ=10 / num_buffer_per_batch=2）と `teacher_ckpt` の受け渡し
5. 評価 config が6ドメイン分そろい、前半3ドメインのものと同一の評価集合を指すこと
6. 実データ 1 step で loss が計算でき、想定外のパラメータが更新されないこと
7. 本環境固有のパスを config に埋め込んでいないこと

**model dict を前半の config と機械的に突き合わせる項目**も入れる（exp_039 で、個別項目の
列挙では取りこぼしを2件続けて出したため）。

## 7. リスク・懸念

- **documents は 17,866 枚と大きい**。前半最大の electromagnetic（25,398 枚）に次ぐ規模で、
  条件3 のリプレイ＋蒸留では 12〜15 h かかる見込み。
- **過去プールが累積するため、1ドメインあたりの露出が後半ほど減る**（§2.4）。1バッチの過去画像は
  常に1枚で、t=4 では3ドメインから、t=6 では5ドメインから一様に引く。古いドメインほど
  相対的に薄まる。これは累積方式の帰結として記録する。
- **前半の3条件の t=3 時点の値**を起点として引き継ぐ必要がある。exp_024 / exp_027 / exp_035 の
  結果記録から取得する（本実験では再測定しない）。
- クラスタの同時実行4ジョブ制限。exp_039 が2ジョブ使っている場合、exp_040 は同時に2本までしか
  投入できない。

## 8. 承認をお願いする範囲

§2 の条件で、3条件について後半3ドメイン（Aerial → Microscopic → Documents）を逐次学習・
評価すること。**実行は MPRG クラスタ**。約 85 GPU 時間。

## 9. 確定事項（2026-08-14）

1. **初期パラメータ**: 3条件とも `epoch_20.pth`（note11 の記載も修正済み）。
2. **過去プール**: 累積。t=4 は underwater + electromagnetic + videogames の各 500 枚
   ＋ Objects365 1,000 枚。t=5 以降も学習済みドメインを追加していく。
3. **ジョブ分割**: 3条件を3ジョブに分けて並列投入。
4. **ソース構成**: ［現在, 参照(Objects365), 過去プール］の3ソースを維持。Objects365 は
   過去プールに統合せず独立ソースのまま、1,000 枚・毎バッチ 1 枚で固定。
