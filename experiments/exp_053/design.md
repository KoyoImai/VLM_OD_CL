# exp_053 設計書: 蒸留E × 学習可能モジュールのアブレーション・後半 3 ドメイン継続

作成 2026-08-25。**実行前のユーザー承認ゲート**（行動原理1・3）。
関連: [[../exp_052/design]]（前半 3 ドメイン。本実験はその継続），
[[../exp_040/design]]（後半 3 ドメインの蒸留E枠の正本），[[../exp_015/design]]（モジュール境界）。

## 0. 位置付け

exp_052（蒸留E＋リプレイ・旧バッチ [4,1,1]×6 で学習可能モジュールを 5 通りに制限，
underwater → electromagnetic → videogames）の続きとして，各条件の t=3 ckpt から
aerial → microscopic → documents（t=4..6）を逐次学習・評価する．
全モジュール学習の対照は exp_040 kdE（t=4..6），リプレイのみは exp_027/exp_040．

## 1. 条件

### 1.1 共通枠（exp_040 kdE と同一．差分は学習可能モジュールのみ）

- config は exp_040 の `kdE_{aerial,microscopic,documents}.py` を継承
  （バッチ [現在4, 汎用1, 過去プール1]×6．過去プールは学習済みドメイン×500 の入れ子
  ConcatDataset．蒸留E λ=10・バッファ由来 2 枚・教師 θ_{t-1}．20 epoch・seed 0・AMP 無し）
- 上書きは exp_052 と同一の 2 点だけ:
  1. paramwise custom_keys（凍結 = lr_mult 0.0．5 条件の表は exp_052 design §2.2 と同一）
  2. ckpt 保存方針（epoch_20 のみ・optimizer 状態なし）

### 1.2 初期重みと教師（継続の要）

- t=4 の load_from・teacher は **exp_052 の同一条件の** `<cond>_videogames_work_dir/epoch_20.pth`．
  条件をまたいだ接続はしない（swinNeck の続きは swinNeck）．
- t=5, 6 は本実験内の前ドメイン epoch_20（従来どおりドライバが渡す）．
- **前提: exp_052 のクラスタ実行が t=3 まで完了していること．**未完了の条件は投入しない．

### 1.3 評価

各 t（4..6）で学習済みドメイン 1..t と ZCOCO を plain 評価
（uw/em/vg は exp_023，aerial/microscopic/documents は exp_026 の評価 config．exp_051 と同一）．
t=6 で全 6 ドメイン＋ZCOCO が揃う．

## 2. 実装（design 承認後．新規コード無し）

| 成果物 | 内容 |
|---|---|
| `gen_configs.py` | 学習 config 15 本（`{5条件}_{aerial,microscopic,documents}.py`）．exp_052 の生成器の base を exp_040 kdE に替えたもの |
| `run_sequential.sh` | 逐次ドライバ（COND 指定，t=4..6）．初期 ckpt は `START` 環境変数（既定 `experiments/exp_052/<cond>_videogames_work_dir/epoch_20.pth`） |
| `sbatch_{cond}.sh` | 5 本．exp_052 クローンの `experiments/exp_052` を read-only bind して t=3 ckpt を参照 |
| `check_exp053_setup.py` | exp_040 kdE との全キー diff（差分 = paramwise と ckpt 方針のみ）・optimizer 実体での lr>0 集合照合・実データ 1 step（凍結不変・学習対象のみ更新） |
| `README.md` | クラスタ手順（push/clone/sbatch はユーザー実施）・結果転送 |

## 3. 実行環境とコスト

MPRG クラスタ 5 ジョブ（各 4 GPU・a6000_ada・`--time=96:00:00`）．
後半 3 ドメインは前半より画像数が多く（aerial 6,643・microscopic 15,989・documents 15,333），
exp_040 実績から各条件 約 60〜80 h と見積もる．ckpt は last のみ 0.7 GB × 3 × 5 ≒ 10 GB．

## 4. リスク・懸念

- exp_052 の t=3 ckpt が前提（§1.2）．条件ごとに完了確認してから投入する．
- microscopic/documents はプロンプトが長く（各 28 クラス・documents は多クラス），
  旧バッチ 6/GPU での VRAM は exp_040 で実績あり（同一構成のため新規リスクなし）．
- qsel 条件（学習 0.30M）は t=4 以降さらに適応が出にくい可能性があるが，それ自体が測定対象．

## 5. 承認をお願いする範囲

exp_052 の 5 条件を，各条件の t=3 ckpt から aerial → microscopic → documents へ
継続学習・評価すること（クラスタ 5 ジョブ．push・clone・sbatch はユーザー実施）．
