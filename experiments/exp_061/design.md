# exp_061 設計書: バッファサイズ感度分析（Ours=蒸留E ＋ ER=リプレイ、クラスタ）

作成 2026-09-11。**実行前のユーザー承認ゲート**（行動原理1・3）。
関連: [[../exp_058/design]]（Ours の同一分析・DGX1。バッファ拡張方式の正本）、
[[../exp_035/design]]・[[../exp_040/design]]（Ours=kdE の枠）、
[[../exp_027/design]]・[[../exp_040/design]]（ER=replay の枠）。

## 0. 位置付け

exp_058 は Ours（蒸留E）のバッファ量感度を DGX1 で測った。本実験は
**同一のバッファ拡張軸を Ours と ER の両手法でクラスタ上で測り直す**。
Ours も再学習するのは、Ours と ER を同一環境・同一バッファファイル・同一 seed で
学習し、手法間の感度差を公平に比較するため（2026-09-11 ユーザー決定）。

## 1. 条件

| 条件 | 参照（O365） | 過去（/ドメイン） | 倍率 |
|---|---|---|---|
| （再利用）base | 1,000 | 500 | 1× |
| b2 | 2,000 | 1,000 | 2× |
| b3 | 3,000 | 1,500 | 3× |
| b4 | 4,000 | 2,000 | 4× |
| b5 | 5,000 | 2,500 | 5× |

- **手法**: Ours（kdE・λ=10・L2・教師 θ_{t-1}）と ER（replay・蒸留なし）。
  バッチ比率・スケジュール・seed 0・num_classes 256 は各手法の既存枠と同一で不変。
  条件間で変えるのは**バッファ json のパスだけ**（exp_058 と同じ rewrite 方式）。
- **base（1×）は再利用**: Ours=exp_035/040 kdE、ER=exp_027/040 replay の既存実測。
  新規学習は b2〜b5 のみ。
- **ドメイン**: 6 通し（underwater → electromagnetic → videogames → aerial →
  microscopic → documents）。評価は各 t の学習済み 1..t ＋ ZCOCO。
- **ckpt**: epoch_20 のみ・optimizer 状態なし。

## 2. バッファ（exp_058 を再利用）

exp_058/buffer/ の入れ子拡張バッファ（参照 2000/3000/4000/5000、過去 各ドメイン
1000/1500/2000/2500、seed 0、manifest 付き、包含鎖）を**そのまま使う**。
Ours と ER で同一ファイルを参照するため、バッファ内容は手法間で完全一致する。
新規作成は不要。

## 3. 実装（承認後）

| 成果物 | 内容 |
|---|---|
| `gen_configs.py` | Ours=exp_035/040 kdE・ER=exp_027/040 replay を解決し、train_dataloader の |
|  | バッファ ann_file を exp_058/buffer の拡張版に差し替え。2 手法 × 4 サイズ × 6 = 48 本 |
| `run_sequential.sh` | `METHOD=ours|er COND=b2..b5` で 6 ドメイン逐次学習＋評価。既存 ckpt/eval はスキップ |
| `sbatch_*.sh` | 手法×条件でジョブ分割（リプレイに O365 bind 必要。exp_040/058 と同じ） |
| `check_exp061_setup.py` | ベースとの diff＝バッファ ann_file のみ・1 step 有限・ER は蒸留損失なしを検証 |

- 実行本体（KDGroundingDINO・replay の ConcatDataset）は既存を無変更で使用。
- 学習ジョブ数: 2 手法 × 4 条件 × 6 ドメイン = 48。評価は各条件 27（学習済み1..6＋ZCOCO×6）。

## 4. 記録・判定

`results/summary.md` に Ours・ER それぞれの base〜b5 の推移（各ドメイン・ZCOCO・Avg6）を
事実のみ記録する。判定基準はユーザーがノートで確定する。
比較の主眼は「バッファ拡大の利得が Ours と ER で同等か、手法により飽和点が違うか」。
