# exp_027: リプレイあり逐次FT（全モジュール／特徴抽出+融合）のクラスタ実行

- 作成日: 2026-07-25
- 状態: **実装着手の指示を受領（2026-07-25）**。呼称の統一・Objects365 デバッグ実行の追加を指示され、本 design を更新した。**クラスタでの本実行はまだ行っていない**（投入はユーザーが行う）。
- 元となる実験ノート: [[../../experiment_notes/note04]]（実験1＝比較ベースラインの整備）。**本実験に対応するノートは未作成**のため、下記「0. 位置付け・目的・仮説」は暫定であり、ノート作成時に差し替える。
- 関連:
  - [[../exp_023/design]] — 本環境で同一の2手法（手法5・手法6）を実行中。
  - [[../exp_024/design]] — クラスタ・**リプレイなし**・全モジュール。
  - [[../exp_025/design]] — クラスタ・**リプレイなし**・特徴抽出+融合。
- 実行環境: **MPRG GPU クラスタ**（[[../../README_cluster]]）。
- **着手条件**: [[../exp_023.5/design]] の Tier2 が完了していること（**完了済み**）。加えて、本実験はリプレイを使うため **Objects365v1 のステージング済みデータがクラスタに転送済み**であること（§6 の事前確認）。

---

## 0. 位置付け・目的・仮説

- **位置付け**: 実験1（比較ベースラインの整備）のうち、**リプレイありの逐次FT 2条件**をクラスタで取る実験。exp_024/025 がクラスタで**リプレイなし**の同2条件を担うので、本実験が入ることで、クラスタ上に

  | | リプレイなし | リプレイあり |
  |---|---|---|
  | **全モジュール** | exp_024 | **exp_027-1** |
  | **特徴抽出+融合** | exp_025 | **exp_027-2** |

  という 2×2 が揃う。
- **目的**: リプレイの有無と学習範囲の2要因を、**同一環境・同一設定で**対比できるようにする。現状はリプレイありが本環境（exp_023）、リプレイなしがクラスタ（exp_024/025）で、環境が要因と交絡する。本実験はその交絡を除く。
- **副次的な効果**: exp_023 の手法5・手法6 と設定が同一なので、突き合わせれば**本環境とクラスタで同一設定の学習結果が一致するか**（環境間の再現性）も確認できる。ただしこれは副産物であり、本実験の主目的ではない。
- **仮説**: 本実験は基準線の整備が主目的であり、強い仮説検証ではない。リプレイありはリプレイなしに比べて ZCOCO と過去ドメインの保持が改善すると期待されるが、**判定は実測値で行う**。

## 1. 構成（2つのサブ実験）

| # | 内容 | 学習範囲 | リプレイ | 本数 |
|---|---|---|---|---|
| **027-1** | 逐次FT + リプレイ（**全モジュール**） | 全モジュール | あり | 1ジョブ（3ドメイン直列） |
| **027-2** | 逐次FT + リプレイ（**特徴抽出+融合**） | Swin + neck + BERT + text_feat_map + encoder + level_embed | あり | 1ジョブ（3ドメイン直列） |

- 027-2 の凍結対象は decoder / bbox_head / memory_trans_fc / memory_trans_norm / query_embedding / dn_query_generator。凍結は `requires_grad=False` ではなく `lr_mult=0.0`（実効 lr が 0）で実装されており、exp_023 の既存 config がそのまま該当する。
- **呼称は exp_024 以降に統一する**（2026-07-25 決定）。すなわち note04 の定義に従い、**条件A＝全モジュール／条件B＝特徴抽出+融合**とする。exp_027 の config 名・work_dir 名・評価出力先のすべてをこの呼称で揃え、`fullft_replay_*`（027-1）／`condB_replay_*`（027-2）とする。
  - 経緯として、exp_023 の config ファイル名は note02 の旧呼称に従っており、**`condA_replay_*.py` が実装しているのは特徴抽出+融合（note04 の条件B）**である。exp_027 では新呼称の config を1行の継承エイリアスとして置き、旧名が exp_027 の中に現れないようにする（§6）。exp_023 側のファイルは実行中のため改名しない。

## 2. 対象ドメイン・逐次順

- underwater → electromagnetic → videogames（note04 の使用ドメイン3つ、exp_023/024/025 と同一順）。

## 3. データ・リプレイの混合

exp_023 の設定をそのまま使う（config 流用のため定義上も同一）。

- **バッファ**: 参照 = Objects365v1 から 1,000 枚（`buffer/reference_o365v1_1000.odvg.json`）、過去プール = 各過去ドメイン 500 枚（`buffer/{domain}_500.odvg.json`）。いずれもリポジトリに git 管理されており、`git pull` で配布される（計 3.3 MB）。
- **混合比**: `CurrentEpochMultiSourceSampler`、`batch_size=6`（4 GPU）。
  - t=1（underwater）: `[現在, 参照]`、`source_ratio=[2,1]` → 現在 4 ・参照 2。
  - t≥2: `[現在, 参照, 過去プール]`、`source_ratio=[4,1,1]` → 現在 4 ・参照 1 ・過去 1。過去が複数ドメインになる t=3 では、入れ子 `ConcatDataset` で1つの「過去プール」に束ねて一様に引く。
- **現在ドメインのバッチは全 t で 4** に揃っており、リプレイなしの exp_024/025（`batch_size=4`）と現在ドメインの露出が一致する。
- テキスト入力は OD-ODVG ＋ `RandomSamplingNegPos`（同一ドメイン負例、num_sample_negative=85 / full_sampling_prob=0.5 / max_tokens=256）、`num_classes=256` 統一、**seed=0**。

## 4. 学習スケジュール

- 2手法とも epoch=20、milestones=[15]（gamma=0.1）、lr=1e-4、AdamW（wd=1e-4）、Swin/BERT は lr_mult=0.1。
- 初期値は t=1 が θ0、t≥2 は前ドメインの last（`load_from` で重みのみ継続。LR スケジュールと optimizer 状態は毎ドメインでリセット）。
- 適応 mAP は last（epoch_20）を採用。in-training val は行わない（`val_interval = max_epochs + 1`）。
- ckpt は 1 エポックごとに全保持（`max_keep_ckpts=-1`）。

## 5. 評価・判定基準

- 各ドメイン t の学習後に、**現在ドメイン ＋ それまでの全過去ドメイン ＋ ZCOCO（COCO2017-val）** を評価する。評価プロトコルは exp_002.5 準拠（`FixScaleResize scale=(800,1333)`、batch_size=1、CocoMetric bbox）。
- 忘却は個々のドメインごとに計算し、**全ドメイン平均は取らない**。
- 本実験は基準線の整備が目的のため、合否判定ではなく実測値の収集を行う。
- 参照値: θ0 ゼロショット（exp_001 実測）は COCO 0.504 / underwater 0.051 / electromagnetic 0.024 / videogames 0.016。

## 6. 実装（承認後に着手、実行前に検証）

- **config は「1行の継承エイリアス」だけを新規作成する**。設定値は複製せず、exp_023 の既存 config を `_base_` で丸ごと継承する。呼称を exp_024 以降に統一しつつ（§1）、設定のドリフトを構造的に不可能にするため。
  - 027-1: `experiments/exp_027/configs/fullft_replay_{domain}.py` → `_base_ = '../../exp_023/configs/fullft_replay_{domain}.py'`
  - 027-2: `experiments/exp_027/configs/condB_replay_{domain}.py` → `_base_ = '../../exp_023/configs/condA_replay_{domain}.py'`（継承元は旧呼称。中身は特徴抽出+融合）
  - 評価: `experiments/exp_023/configs/eval_{domain}.py` ／ `configs/mm_grounding_dino/eval_base_coco.py` をそのまま使う（exp_024/025/026 と同じ）。
  - 上書きは一切書かない。したがって本環境（exp_023）との差は**実行環境だけ**に閉じ、§0 の副次的な再現性確認が成立する。エイリアスが設定を変えていないことは実装検証で確認する（§6 の検証項目 5）。
- **θ0 の与え方**: exp_023.5 以来の方針どおり、ドライバが `--cfg-options load_from=<明示パス>` で `/workspace/kouyou/ckpt/...` を渡す（URL・キャッシュ状態に依存させない）。
- **backbone の `init_cfg` を無効化する**（`--cfg-options model.backbone.init_cfg=None`）。exp_024/025/026 と揃える。最終重みへの影響が無いことは exp_024 §7 で実測確認済み（`load_from` が backbone 187 パラメータを全て上書きし、全 908 テンソルが一致）。
- **依存モジュール**（いずれも git 管理済みで `git pull` で配布される）:
  - `exp023_np_compat.py`（リポジトリルート。`np.long` 互換シム）
  - `mmdet/datasets/samplers/current_epoch_multi_source_sampler.py`（`__init__.py` には未登録のため、config の `custom_imports` がフルモジュールパスで読み込む）
- **ドライバ**: `experiments/exp_027/run_sequential_fullft_replay.sh`（027-1）／`run_sequential_condB_replay.sh`（027-2）。exp_024/025 のドライバと同型（t=1→3 の逐次、各 t で現在＋全過去＋ZCOCO を評価）。
- **クラスタ実行**: `experiments/exp_027/{sbatch.sh, train_val.sh}`。`RUN_STAGE` で段階を選ぶ（`debug` / `fullft` / `condB` / `all`）。exp_024/025 と異なり **Objects365 の bind が必要**：
  ```
  --bind "$HOME_DATA/o365v1_stage":/workspace/kouyou/datasets/o365v1_stage
  ```
  RF100 は local_cache へステージング、Objects365 は読み込み枚数が少ない（1,000 枚）ためホームから直接 bind する（2026-07-24 のユーザー決定、exp_023.5 と同じ方針）。

### 6.1 Objects365 のデバッグ実行（`RUN_STAGE=debug`）

本実験の新しい要素は「クラスタ上で Objects365 のバッファ画像が読めるか」だけである。40 時間級のジョブを投げてから発覚すると損失が大きいため、**学習を伴わない短時間の確認ジョブ**を用意する（2026-07-25 指示）。exp_023.5 の Tier1（プラミング確認）と同じ位置づけ。

`experiments/exp_027/debug_o365.py` が以下を順に確認し、失敗した項目を明示して非ゼロ終了する。

1. **bind の確認**: `/workspace/kouyou/datasets/o365v1_stage/Objects365_v1/2019-08-02/train/` が存在し読めること。
2. **バッファ JSON の確認**: `experiments/exp_023/buffer/reference_o365v1_1000.odvg.json` が 1,000 件で、`o365v1_label_map.json` が読めること。
3. **画像ファイルの実在確認**: 参照バッファが指す 1,000 枚すべてについて実ファイルの存在を確認し、欠損があれば件数と先頭例を出す（**転送漏れはここで捕まる**）。過去プール（RF100 各 500 枚）も同様に確認する。
4. **デコード確認**: 実際に数枚を読み込んでデコードできること（転送途中で切れたファイルを検出する）。
5. **混合比の確認**: 各 t の config から dataloader を組み、数バッチ取り出して 1 バッチの内訳が **t=1 で 現在4・参照2**、**t≥2 で 現在4・参照1・過去1** になっていることを実データで確認する。
6. **1 ステップの学習**: θ0 をロードして `model.loss()` まで通し、loss が有限であること・凍結対象（027-2）の実効 lr が 0 であることを確認する。

所要は数分〜十数分。`RUN_STAGE=debug` 単体で投入でき、本実行の前に必ず1回通す。

- **事前確認（実行前）**: 上記 `RUN_STAGE=debug` が全項目 OK であること。加えて `buffer/*.odvg.json` が `git pull` 後にクラスタ側に存在すること（項目 2 で検出される）。
- **実装検証（本環境で実行前に実施）**:
  1. 027-1／027-2 の config が exp_023 の対応 config と**完全に同一の設定を生成する**こと（エイリアスが何も変えていないこと）。
  2. 027-2 の凍結が効いていること（実効 lr = 0、1 ステップ後に凍結対象の重みが不変）。θ0 をロードした実際の学習条件で確認する（exp_025 §7 で得た教訓）。
  3. リプレイの混合が t ごとに意図どおりであること（`debug_o365.py` の項目 5 と同じ確認を本環境で先に通す）。

## 7. コスト・運用

- **所要時間**: exp_023 の実測（electromagnetic が 1,588 iter/epoch × 20 epoch で 22時間34分、約 2.5 秒/iter）からの概算で、3ドメイン合計 約 57,900 iter ≒ **40 時間**＋評価 約 3 時間 ＝ **1手法あたり 43 時間前後**。2手法で**クラスタ 2 枠を 43 時間ずつ**占有する。
- **同時実行の枠**: クラスタは同時 4 ジョブ。exp_024・exp_025 と exp_027 の2本で 4 枠が埋まり、exp_026 は空きが出てから流すことになる。枠の割り当ては実装完了後に投入順としてまとめる（2026-07-25、まず実装を進める方針）。
- **デバッグ実行のコスト**: `RUN_STAGE=debug` は GPU 1 ジョブ枠を数分〜十数分だけ使う。本実行前に必ず1回通す。
- **ディスク**: ckpt は 2手法 × 3ドメイン × 20 個 × 約 2.0 GB ＝ **約 240 GB**。exp_024/025 の 240 GB と合わせて約 480 GB で、ホーム（約 3 TB）には収まる。

## 8. この実験で言えないこと（限界）

- リプレイのバッファ構成（参照 1,000／過去 500、混合比）は1通りしか試さないため、リプレイの設計空間に対する一般的な結論は出せない。
- 3ドメインのみの逐次であり、より長い系列での挙動は分からない。
- 提案手法（蒸留）の有効性は扱わない（実験2）。
- 本環境との一致確認は seed 固定下の1試行同士の比較であり、`deterministic=False` のため完全一致は保証されない。乖離があった場合、環境差か実行間のばらつきかを本実験だけでは切り分けられない。

## 9. 成果物

- `experiments/exp_027/fullft_replay_{domain}_work_dir/`、`condB_replay_{domain}_work_dir/`
- `experiments/exp_027/fullft_replay_eval_after_{domain}/{on_{evd},zcoco}/`、`condB_replay_eval_after_{domain}/{on_{evd},zcoco}/`
- `experiments/exp_027/results/`（事実記録のみ。解釈・考察は書かない＝行動原理8）
