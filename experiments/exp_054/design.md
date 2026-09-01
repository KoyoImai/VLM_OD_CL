# exp_054 設計書: seed 頑健性の確認（Ours・ER・EWC・InfLoRA・ZiRa・DitHub × seed 1, 2）

作成 2026-08-31。**実行前のユーザー承認ゲート**（行動原理1・3）。
関連: [[../exp_035/design]]・[[../exp_040/design]]（Ours の枠），[[../exp_027/design]]（ER），
[[../exp_045/design]]（EWC・InfLoRA），[[../exp_039/design]]・[[../exp_041/design]]（ZiRa・DitHub）。

## 0. 位置付け

これまでの全実験は seed=0 固定である。note15 の主要 7 手法（2026-09-01 に Finetuning を追加）について seed=1, 2 の系列を
追加し（既存 seed=0 と合わせて n=3）、手法間の序列（適応・保持・ZCOCO）が seed に
依存しないことを確認する。**仮説と判定基準はユーザーがノートで確定する。**
参考: 同一条件の再実行差の実測は exp_033 にある（λ=1.0 反復で ZCOCO 0.421 vs 0.415 等）。

## 1. 条件（2026-08-31 確定）

- seed: **1 と 2**（randomness.seed。deterministic=False は現行どおり）
- ドメイン: **6 通し**（underwater → electromagnetic → videogames → aerial → microscopic → documents）
- 手法と枠（すべて既存 config を継承し、**seed 以外は一切変えない**）:

| 手法 | 枠（継承元 config） | 備考 |
|---|---|---|
| Ours | exp_035 kdE（t1–3）＋ exp_040 kdE（t4–6） | 蒸留E λ=10・旧バッチ [4,1,1]×6 |
| Finetuning | exp_024 replayfree（t1–3）＋ exp_040 replayfree（t4–6） | リプレイなし逐次 FT（2026-09-01 追加） |
| ER | exp_023/027 fullft_replay（t1–3）＋ exp_040 replay（t4–6） | リプレイのみ |
| EWC | exp_045 ewc_*（λ=1000 はドライバ指定） | Fisher 推定の --seed も変更 |
| InfLoRA | exp_045 inflora_*（r=16, α=16, lamb 0.95） | prepare / update / merge の --seed も変更 |
| ZiRa | exp_039・exp_041 の replay free 変種 | note15 と同じ変種 |
| DitHub | 同上 | 同上 |

- seed の通し方: 学習は `--cfg-options randomness.seed=$SEED`（config は変更しない）。
  EWC / InfLoRA / ZiRa / DitHub の補助スクリプト（Fisher 推定・部分空間推定・初期化）の
  `--seed` 引数も $SEED に揃える。
- **バッファは既存の exp_023 の json（各 500 枚・参照 1,000 枚）を共用**する。
  つまり「リプレイ画像の選択」は固定で、seed が変えるのはデータ順・拡張・dropout・
  DN ノイズ・LoRA 初期化等。バッファ選択の seed 依存まで見る場合は別実験とする。
- 逐次・評価は各手法の既存ドライバと同一（各 t で学習済み 1..t ＋ ZCOCO、last 採用）。
- ckpt は epoch_20 のみ・optimizer 状態なし。

## 2. 実装（design 承認後。新規モデルコード無し）

| 成果物 | 内容 |
|---|---|
| `run_{ours,er,ewc,inflora,zira,dithub}.sh` | 既存ドライバを exp_054 用に写し、SEED 環境変数で randomness.seed と補助スクリプト --seed を通す。出力は `experiments/exp_054/{手法}_s{seed}_*` |
| `sbatch_{手法}_s{1,2}.sh` | 12 本（exp_051/053 の器を写す） |
| `check_exp054_setup.py` | (1) seed が実効 config に届く（merge 後 randomness.seed==SEED）(2) seed 以外の全設定が既存実験と一致（全キー diff）(3) 各手法 1 step（seed=1）(4) ZiRa/DitHub/EWC/InfLoRA の補助スクリプトに --seed が渡ること |
| `README.md` | クラスタ手順（push・clone・sbatch はユーザー実施）・転送 |

## 3. 実行環境とコスト

MPRG クラスタ **14 ジョブ**（7 手法 × 2 seed。各 4 GPU・a6000_ada・`--time=144:00:00`）。
同時実行 4 本制限のため 3 波に分かれる。見積もり: Ours / ER 各 90〜120 h、
EWC / InfLoRA 各 60〜90 h、ZiRa / DitHub 各 60〜90 h → 全体で実時間 1.5〜2 週間規模。
ckpt 容量 ≒ 0.7 GB × 6 × 12 ≒ 50 GB（last のみ）。

## 4. リスク・前提確認

1. **EWC の t=3 videogames**: exp_045（seed=0）は当初 OOM で停止し、最終的に完走して
   いるが、**その際の最終構成（encoder num_cp / partition）が本環境の記録から確認できない**。
   クラスタ側 `exp_045 の ewc_videogames_work_dir/*/vis_data/config.py` の実効値を確認し、
   exp_054 はそれを踏襲する（ユーザーに確認をお願いする事項）。
2. ZiRa / DitHub のドライバが seed をどこから読むかは実装検証で確認する
   （config 経由でなければドライバ側に --cfg-options を追加）。
3. deterministic=False のため同一 seed でも完全ビット再現はしない（現行方針どおり）。

## 5. 承認をお願いする範囲

§1 の 6 手法 × seed {1, 2} × 6 ドメインの逐次学習・評価（クラスタ 12 ジョブ）。
本環境で行うのは ドライバ・sbatch・検証の作成と実行前検証まで。
