# exp_025: リプレイ無し逐次ファインチューニング（条件B＝特徴抽出+融合）

- 作成日: 2026-07-24
- 状態: **承認待ち（未承認・未実行）**。承認前に実験は実行しない（行動原理3）。
- 元となる実験ノート: [[../../experiment_notes/note04]]（実験1＝比較ベースラインの整備）。
- 対になる実験: [[../exp_024/design]]（同一設定で**条件A＝全モジュール**）。本実験は学習範囲のみが異なる。
- 実行環境: **MPRG GPU クラスタ**（[[../../README_cluster]]）。
- **着手条件**: [[../exp_023.5/design]] の Tier2（ゼロショット→1エポック学習→評価）が完了し、クラスタで学習・評価が端まで通ることを確認してから実装・実行に入る。exp_023.5 が通らない場合、クラスタ実行という前提自体が成立しないため、本実験の設計も見直しが必要になる。
- **環境の役割分担**: 本環境（GPU4枚）は exp_023 の完遂に使用し、**本実験はクラスタで実行**する。exp_024 とは別ジョブとして同時に走らせられるため、長時間の逐次学習2本を並行させる。

---

## 0. 位置付け・目的・仮説

- **位置付け**: 実験1の残りベースラインのうち、リプレイ無し逐次FT の**条件B**版。exp_024（条件A）と対になる。
- **目的**: 学習範囲を「画像-テキスト特徴抽出部＋特徴融合部」に限定した場合の、リプレイ無し逐次学習の適応・忘却を測る。これにより
  (i) exp_023 の条件B＋リプレイ（`condA_replay_*` として実装済み）との**リプレイ有無**の比較、
  (ii) exp_024（条件A）との**学習範囲**の比較、
  の2軸が同時に成立する。
- **仮説**: 下流（decoder / bbox_head / query）を凍結するため、条件A より適応 mAP は低くなる可能性がある一方、共有検出経路の更新が抑えられる分、ZCOCO・過去ドメインの保持は条件A より良い可能性がある。ただし exp_004（凍結FT は適応↓・忘却↑の両負け）の前例があるため、**保持が改善する保証はない**。判定は実測値で行う。

## 1. 手法の定義（1条件のみ）

| 項目 | 内容 |
|---|---|
| 学習範囲 | **条件B＝特徴抽出+融合のみ学習**。学習: `backbone`(lr_mult 0.1) / `language_model`(0.1) / `neck`(1.0) / `text_feat_map`(1.0) / `encoder`(1.0) / `level_embed`(1.0)。**凍結**（lr_mult=0.0）: `decoder` / `bbox_head` / `memory_trans_fc` / `memory_trans_norm` / `query_embedding` / `dn_query_generator` |
| リプレイ | **無し**（現在ドメインのみ） |
| 逐次 | あり（3ドメインを順に学習し、重みのみ引き継ぐ） |

- **呼称は note04 の定義に従う**（条件A＝全モジュール、条件B＝特徴抽出+融合）。exp_023 の `condA_replay_*.py` は名称に反して**この条件B**を実装しているため、本実験の config は `condB_replayfree_*` と命名し、取り違えを防ぐ。
- 学習対象モジュールの lr_mult は exp_023 の条件B 実装と**同一値**にし、リプレイ有無以外の差分を作らない。

## 2〜5. 共通条件（exp_024 と完全に同一）

- **ドメイン順序**: underwater → electromagnetic → videogames（3ドメイン・単一順序）。
- **テキスト入力**: OD-ODVG ＋ `RandomSamplingNegPos`（同一ドメイン負例）、num_sample_negative=85 / full_sampling_prob=0.5 / max_tokens=256。`num_classes=256` 統一。
- **バッチ**: 現在ドメインのみ **4/GPU**（`batch_size=4`）、4GPU 分散。
- **スケジュール**: epoch=20、milestones=[15]（gamma=0.1）、lr=1e-4、AdamW（wd=1e-4）。各ドメインで LR・optimizer をリセット、重みのみ前ドメイン last を `load_from`（t=1 は θ0）。適応 mAP は **last**。**seed=0**。

学習範囲（`paramwise_cfg`）以外は exp_024 と1つも変えない。これにより両実験の差分が学習範囲だけに閉じる。

## 6. 評価・判定基準

- exp_024 と同一: 各ドメイン学習後に (a) 現ドメインの適応 mAP、(b) ZCOCO、(c) 過去に学習した各ドメインごとの検出性能。
- **忘却はドメインごとに個別に計算**（平均は取らない）。
- 比較ベースラインの整備が目的のため、合否判定ではなく実測値の収集を行う。

## 7. 実装（承認後に着手、実行前に検証）

- **新規 config**（既存無変更）:
  - `experiments/exp_025/configs/condB_replayfree_{underwater,electromagnetic,videogames}.py` — **exp_024 の同名ドメイン config を直接継承**し、`optim_wrapper.paramwise_cfg`（上表の lr_mult）だけを上書きする。別途ベース config は作らない。exp_023 の条件B 実装（`condA_replay_*`）と同じパターンで、**学習範囲以外の同一性が構造的に保証される**（exp_024 側の変更が自動的に反映され、設定のドリフトが起きない）。
  - 評価は exp_023 の `eval_{domain}.py` / `eval_base_coco.py` を流用（無変更）。
- **backbone の `init_cfg` を無効化する**（`model.backbone.init_cfg=None`。2026-07-24 決定）。理由・検証は [[../exp_024/design]] §7 と同じ——`init_weights()` が読む ImageNet Swin-T は直後の `load_from` で backbone 187 パラメータ全てが上書きされるため最終重みに影響せず、全 908 テンソルの一致を実測で確認済み。exp_024 と設定を揃えるためにも同一にする。
- **逐次ドライバ**: `experiments/exp_025/run_sequential_condB_replayfree.sh`。
- **クラスタ実行**: `experiments/exp_025/{sbatch.sh, train_val.sh}`。exp_024 と同構造。
- **実装検証（実行前）**: 凍結は `requires_grad=False` ではなく **`lr_mult=0.0`** で実現するため、「勾配が流れないこと」では検証できない（勾配は計算されるが実効学習率が 0 になる）。したがって次で検証する:
  - (a) optimizer の param_group で、**凍結対象の実効 lr が 0**、学習対象が期待値（backbone/language 1e-5、他 1e-4）であること。
  - (b) **実データで loss→backward→optimizer.step() を1回実行**し、凍結対象の重みが1つも変化せず、学習対象が変化すること。検証は θ0 をロードした実際の学習条件で行う（ランダム初期化では一部モジュールの勾配が 0 になり、実条件と挙動が異なるため）。
  - (c) 負例が同一ドメインに閉じること（exp_024 と同一データ経路のため exp_024 の検証で代替可）。
  - (d) exp_024 と**学習範囲以外の設定が完全一致**していること（`train_dataloader` / `train_cfg` / `param_scheduler` / `randomness` / `default_hooks` / `load_from` / `backbone.init_cfg` の突き合わせ）。

## 8. この実験で言えないこと（限界）

- 単一順序のみのため、順序依存は言えない。
- 新 regime のため旧 regime の exp_004（凍結FT）とは直接比較できない。凍結範囲も exp_004 とは異なる。
- 提案手法（蒸留）の有効性は扱わない。

## 9. 成果物

- `experiments/exp_025/{domain}_work_dir/`、評価出力、`experiments/exp_025/results/`（事実記録）。
