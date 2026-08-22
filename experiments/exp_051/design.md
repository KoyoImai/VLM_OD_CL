# exp_051 設計書: 蒸留の適用箇所アブレーション（exp_043 枠・6 ドメイン）

作成 2026-08-22。**実行前のユーザー承認ゲート**（行動原理1・3）。
関連: [[../exp_043/design]]（本実験の枠 = バッチ [4,2,2]×8・kdE の正本）、
[[../exp_028/design]]（蒸留 A/B/D/E の定式化）、
[[../exp_035/design]]（旧バッチ [4,1,1]×6 での λ=10 A/B/D/E、note07 の表）。

## 0. 位置付け

exp_043 の蒸留E（img＋txt＋fus の 3 点、λ=10、バッファ由来 4 枚に限定、バッチ
[現在4, 汎用2, 過去2]×8）を適用箇所別に分解する。条件は蒸留 A（img のみ）／
B（txt のみ）／D（fus のみ）の 3 つで、蒸留E（exp_043 kdE）とリプレイのみ
（exp_043 replay）は既存の実測を対照に使う。旧バッチでの同型アブレーション
（exp_035 系、note07）との対応で「適用箇所の効果がバッチ構成に依存するか」も読める。

## 1. 目的

蒸留 A / B / D の 3 条件を、underwater → electromagnetic → videogames → aerial →
microscopic → documents の 6 ドメインで逐次学習・評価する（2026-08-22 確定）。
**仮説と判定基準はユーザーがノートで確定する。**

## 2. 条件

### 2.1 共通枠（exp_043 と完全に同一。差分は kd.targets のみ）

- データ・バッチ: exp_043 の `replay_{domain}.py` を継承
  （t=1 [現在4, 汎用4]、t≥2 [現在4, 汎用2, 過去2]、batch 8/GPU・4 GPU 合計 32、
  バッファは exp_023 の既存物、`CurrentEpochMultiSourceSampler`）
- スケジュール: 20 epoch / MultiStepLR [15] / AdamW lr 1e-4・wd 1e-4 / grad clip 0.1
- dn 有効 / num_classes 256 / seed 0 / AMP 無し
- 蒸留: `KDGroundingDINO`、L2、λ=10（const）、`num_buffer_per_batch=4`
  （バッチ末尾のバッファ由来 4 枚に限定）、教師 θ_{t-1}（t=1 は θ0 の実パス）
- 逐次方式: 各ドメインの epoch_20 を次の load_from に。評価はドライバ
  （学習済み 1..t ＋ ZCOCO、既存の plain 評価 config）

### 2.2 条件（kd.targets のみが差分）

| 条件 | kd.targets | 蒸留点 |
|---|---|---|
| kdA | ['img'] | neck 出力の多スケール画像特徴 |
| kdB | ['txt'] | text_feat_map 出力のテキスト特徴 |
| kdD | ['fus'] | enhancer 出力（memory / memory_text の 0.5 和） |

対照（追加実行なし）: kdE = exp_043 kdE、リプレイのみ = exp_043 replay、
旧バッチの A/B/D/E = exp_035 系（note07）。

## 3. 判定材料

**最終的な判定はユーザーが行う。**

1. 各 t（1〜6）の学習済み全ドメイン mAP と ZCOCO。
2. 対 exp_043 kdE / replay（同一バッチでの適用箇所の寄与分解）。
3. 対 exp_035 系 A/B/D（旧バッチとの適用箇所効果の一貫性。前半 3 の範囲で対応）。

## 4. 実装（design 承認後に着手。新規コード無し）

| 成果物 | 内容 |
|---|---|
| `gen_configs.py` | 学習 config 18 本（`kd{A,B,D}_{6ドメイン}.py`）。exp_043 の `replay_{domain}.py` を継承し model の kd.targets だけ差し替え |
| `run_sequential.sh` | 逐次ドライバ（COND=kdA/kdB/kdD。exp_043 のドライバを写す） |
| `sbatch_kd{A,B,D}.sh` | クラスタ投入 3 本（exp_043 の器を写す。リプレイあり = Objects365 bind 必要） |
| `check_exp051_setup.py` | 実行前検証（config build・exp_043 kdE との機械的 diff が targets のみ・実データ 1 step で対象別 kd 診断値の出方確認） |
| `README.md` | クラスタ手順（push/clone はユーザー実施） |

## 5. 実行環境とコスト

**MPRG クラスタ 3 ジョブ**（各 4 GPU、a6000_ada、`--exclude=node03`）。
exp_050（DGS）等と合わせて同時実行 4 本制限に掛かる分はキュー待ち。
見積もりは exp_043 kdE 実測ベースで **各条件 約 90〜120 h**（`--time=144:00:00`、
exp_043 と同じ）。蒸留 1 点のため kdE よりやや軽い可能性がある。
ckpt は **last（epoch_20）のみ・optimizer 状態なし**（2026-08-22 ユーザー決定。
各条件 約 0.7 GB × 6 ドメイン）。

## 6. リスク・懸念

- 実装は exp_035/043 で 6 ドメイン実績のある `KDGroundingDINO` の targets 切替のみで、
  新規経路は無い。
- ckpt は last のみのためディスクは軽い（3 条件合計 約 13 GB）。

## 7. 承認をお願いする範囲

§2 の 3 条件を 6 ドメインで逐次学習・評価すること。実行はクラスタ 3 ジョブ
（push・clone・sbatch はユーザー実施）。本環境で行うのは config 作成と実行前検証まで。
