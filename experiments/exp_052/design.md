# exp_052 設計書: 蒸留E × 学習可能モジュールのアブレーション（旧バッチ枠・前半 3 ドメイン）

作成 2026-08-22。**実行前のユーザー承認ゲート**（行動原理1・3）。
関連: [[../exp_035/design]]・[[../exp_028/design]]（本実験の枠 = 旧バッチ [4,1,1]×6 の
蒸留E λ=10。kdE 全モジュール学習が直接の対照）、
[[../exp_015/design]]・[[../exp_016/design]]（モジュール境界の正本）、
[[../exp_017/design]]・[[../exp_018/design]]（凍結学習の方式 = lr_mult 0.0 の前例）。

## 0. 位置付け

蒸留E＋リプレイ（exp_035 系: バッチ [現在4, 汎用1, 過去1]×6、λ=10、蒸留はバッファ
由来 2 枚限定、教師 θ_{t-1}）の学習可能モジュールを、単独モジュール群に制限する
アブレーション。全モジュール学習（exp_035 kdE）とリプレイのみ（exp_027）が既存の
実測対照になる。モジュール境界は exp_015/016 で確立した区分に厳密に揃える。

## 1. 目的

以下の 5 条件を underwater → electromagnetic → videogames の前半 3 ドメインで
逐次学習・評価する（2026-08-22 確定）。
**仮説と判定基準はユーザーがノートで確定する。**

## 2. 条件

### 2.1 共通枠（exp_035 系と同一。差分は学習可能モジュールのみ）

- データ・バッチ: exp_023 の `fullft_replay_{domain}.py` を継承
  （t=1 [現在4, 汎用2]・t≥2 [現在4, 汎用1, 過去1]、batch 6/GPU・4 GPU 合計 24、
  バッファは exp_023 の既存物）
- 蒸留: `KDGroundingDINO`・蒸留E（img+txt+fus）・L2・λ=10・
  `num_buffer_per_batch=2`・教師 θ_{t-1}（t=1 は θ0 の実パス）
- スケジュール: 20 epoch / MultiStepLR [15] / AdamW lr 1e-4・wd 1e-4 / clip 0.1
- dn 有効 / num_classes 256 / seed 0 / AMP 無し
- 逐次方式: epoch_20 を次の load_from に。評価はドライバ（学習済み 1..t ＋ ZCOCO）

### 2.2 学習可能モジュール 5 条件（凍結は paramwise lr_mult=0.0。exp_017/018 の方式）

境界は exp_015 の区分に一致させる。学習側の lr は unfrozen 準拠
（backbone / language_model は lr_mult 0.1 = 実効 1e-5、それ以外は 1.0 = 1e-4）。

| 条件 | 学習するモジュール（custom_keys で lr_mult > 0） | それ以外 |
|---|---|---|
| swinNeck | backbone(0.1)・neck(1.0) | 全て 0.0 |
| bertTfm | language_model(0.1)・text_feat_map(1.0) | 全て 0.0 |
| enhancer | encoder(1.0)（Feature Enhancer 全体） | 全て 0.0 |
| qsel | memory_trans_fc・memory_trans_norm・level_embed・query_embedding（各 1.0）＝ Language-guided Query Selection（exp_015 の qsel 区分） | 全て 0.0 |
| decoder | decoder(1.0)（Cross-Modality Decoder。ref_point_head・decoder.norm を含む） | 全て 0.0 |

- bbox_head（cls/reg branches）と dn_query_generator は**全条件で凍結**
  （exp_017/018 と同じ扱い。cls/reg は exp_015 で別区分＝本アブレーションの対象外）。
- 凍結は lr_mult=0.0 方式（requires_grad は触らない。DDP と教師 deepcopy に安全で、
  exp_017/018 で実績）。

### 2.3 対照（追加実行なし）

- kdE 全モジュール学習: exp_035（t=3 で 0.253 / 0.437 / 0.731 / ZCOCO 0.382）
- リプレイのみ: exp_027
- 蒸留なしの凍結スイープ（単独ドメイン）: exp_017/018（bert_tfm・neckTfm 等）

## 3. 判定材料

**最終的な判定はユーザーが行う。**

1. 各 t（1〜3）の学習済み全ドメイン mAP と ZCOCO。
2. 対 exp_035 kdE（全モジュール学習との差 = 各モジュール単独の適応・保持への寄与）。
3. 対 exp_017/018 の凍結スイープ（蒸留・リプレイの有無で傾向が変わるか）。

## 4. 実装（design 承認後に着手。新規コード無し）

| 成果物 | 内容 |
|---|---|
| `gen_configs.py` | 学習 config 15 本（`{5条件}_{3ドメイン}.py`）。exp_023 の replay config を継承し、model（kdE）と paramwise（凍結）を上書き |
| `run_sequential.sh` | 逐次ドライバ（COND=swinNeck 等。exp_051 のドライバを写す） |
| `sbatch_{cond}.sh` | クラスタ投入 5 本（exp_051 の器を写す。Objects365 bind あり） |
| `check_exp052_setup.py` | 実行前検証: config build・exp_035 kdE 相当との diff・**各条件の学習対象パラメータの実測**（意図したモジュールのみ lr>0）・実データ 1 step |
| `README.md` | クラスタ手順（push/clone/sbatch はユーザー実施） |

## 5. 実行環境とコスト

**MPRG クラスタ 5 ジョブ**（各 4 GPU、a6000_ada）。同時実行 4 本制限により
1 本以上はキュー待ち。見積もりは exp_035 系の前半 3 実測から **各条件 約 30〜45 h**
（`--time=96:00:00`）。ckpt は **last（epoch_20）のみ・optimizer 状態なし**
（直近 2 実験の方針を踏襲。変更希望があれば指定を）。約 0.7 GB × 3 × 5 条件 ≒ 10 GB。

## 6. リスク・懸念

- 実装は既存 `KDGroundingDINO` ＋ paramwise 凍結のみで新規経路なし。
- qsel 条件は学習パラメータが極小（約 0.13M: memory_trans_fc 65.8K＋norm 0.5K＋
  level_embed 1K＋query_embedding 230K → 実測は検証で確認）で、適応がほぼ出ない
  可能性があるが、それ自体が測定対象。
- lr_mult=0.0 凍結は勾配計算自体は走るため、full FT と同等の計算コスト
  （凍結による高速化はない）。

## 7. 承認をお願いする範囲

§2 の 5 条件を前半 3 ドメインで逐次学習・評価すること。実行はクラスタ 5 ジョブ
（push・clone・sbatch はユーザー実施）。本環境で行うのは config 作成と実行前検証まで。
