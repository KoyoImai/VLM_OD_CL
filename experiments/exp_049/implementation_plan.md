# exp_049 実装計画: DGS（CVPR 2025/2026, Dynamic Group Subspace）の移植

作成 2026-08-21。公式実装 https://github.com/Never-wx/dgs（`/workspace/kouyou/dgs_official`
に参照用 clone 済み・リポジトリ外）に準拠し、mmdetection（本環境）に移植する。
論文側の根拠は [[../../papers/DGS_summary]]。学習・実験の実行は別途 design.md の承認後。

## 1. 公式実装の調査結果（2026-08-21 実測）

### 1.1 構造

公式リポジトリは **mmdetection のフォーク**で、DGS は `projects/DGS/` に自己完結
（計 9,694 行）。mmdet 本体への改変は **2 箇所だけ**（diff で全数確認）:
`dino.py` と `dino_head.py` に `dn_cfg=None` を許すガード（dn 無効化のため）。
それ以外の grounding_dino 系ファイルは上流と完全一致。
→ **本体無変更の方針（本プロジェクトの規約）と両立可能**。dn 無効化は本プロジェクト
既存の `NoDNQueryGenerator`＋`NoDNGroundingDINOHead` 系機構（ZiRa/DitHub/lora_cl/ewc_cl
で実績）で代替する。

### 1.2 要約の「欠落情報 7 点」の解明（コード実測）

| 論文で不明だった点 | コードでの実体 |
|---|---|
| 式5 マージ係数の向き | `LoraLinearRep.__rep__`: **新 ← 0.2·base ＋ 0.8·新** を計算し、それを base に昇格して新を破棄（`lambda_A=lambda_B=0.2`、最終 epoch 後の `MergeHook` で実行） |
| 推論時のガウス近似 | テスト画像の**共分散は作らない**。backbone 最終レベル特徴を GAP した 1 ベクトル x に対し、N(x, Σ_task) と N(μ_task, Σ_task) の対称 KL（実質マハラノビス距離）で全タスクと比較し最小を取る。`min_kl > ood_th`（500）なら **-1 = アダプタ無効（zero-shot にフォールバック）** — これが ZCOCO 保持の機構 |
| DTG の特徴・KL | タスクの学習データ全体を事前に 1 パスして backbone 特徴を GAP・保存（`extract_feats`）。平均と**全共分散**（固有値を max·1e-3 でフロアする SVD 正則化）から対称 KL。閾値 `expand_th=150` |
| topology KD 損失 | `inter_loss.py`（inter-intra 関係蒸留）＋ `qd_distn_loss.py`（クエリ蒸留）。stage2 の `feat_distn`: img L2 w=3.0・text L2 w=5.0、`subtype='opt1'` |
| 擬似ラベル | `threshold_pseudo`・hardlabel・sigma=0.4・IoU 0.7（stage2 の `label_distn`） |
| スケジュール | 12 epoch・MultiStepLR milestone [11]・γ0.1。stage1 lr **8e-4**・stage2 lr **5e-4**（論文記載の 1e-3/5e-4 と stage1 が異なる。config が正） |
| stage1/2 の切替 | 毎タスク前にドライバが `adaptive_task_mapping.py`（DTG）を実行し、既存グループ所属なら stage2（蒸留あり）・新グループなら stage1（無制約）の config を選ぶ |

### 1.3 その他の確定事項

- IGA: `AdaptiveExpandMoELora`（experts_num=1, top_k=1 の縮退 MoE = グループ別 LoRA 束）。
  **r=16, alpha=32（scaling=2）**、挿入は feature enhancer 6 層の**画像側・テキスト側 FFN**
  （`replace_layer_type=['enc_ffn_img','enc_ffn_text']`）。base 以外全凍結
  （`frozen_cfg` + `exclude_keywords=['lora_']`）。
- グループ内の新タスク学習: `LoraLinearRep` が base ペア（凍結）＋新ペア（学習）を持ち、
  `WeightsTransformHook(moe_group_init)` が学習開始時に base → 新 をコピー（Group Init）。
- 公式 IVLOD（ODinW-13）: batch 2/GPU・8 GPU（合計16）・`--amp`・dn 無効（`dn_cfg=None`）・
  `contrastive_cfg(max_text_len=256, log_scale=None, bias=None)`（**上流 GDINO-T 既定**。
  MM-GDINO の log_scale='auto', bias=True とは異なる → 本移植では θ0 に合わせ MM-GDINO 側を使う）。
- データ形式は ZiRa 形式の ODinW-13（`annotations_without_background.json`）で、
  本環境の `data/odinw`（exp_034 で使用済み）とそのまま互換。

## 2. 移植方針

**`projects/dgs_cl/` に公式 `projects/DGS/dgs` の必要部分を移植する**（ewc_cl / lora_cl と
同じ流儀: 本体無変更・config の custom_imports で登録・検証スクリプト同梱）。
可能な限り公式コードを逐語的に保ち、変更は次の 3 種に限定して全てコード内に記録する:

1. **import パスと登録**（`projects.DGS.dgs` → `projects.dgs_cl`）。
2. **dn 無効化の方式**: 公式はフォークの `dn_cfg=None` ガードに依存 → 本移植は検出器
   サブクラス内で `NoDNQueryGenerator` に差し替え（既存機構）。head 側の dn_meta 分岐は
   移植時に実データで検証（検証項目 §4）。
3. **θ0 と評価系**: MM-GDINO θ0・本プロジェクトの評価 config（per-dataset 評価・ZCOCO は
   eval_base_coco）。公式の union 評価（coco_inc_metric）は CDIOD 用であり IVLOD 再現には不要。

### 2.1 移植するファイル（公式 → 本リポジトリ）

| 公式 | 移植先 | 備考 |
|---|---|---|
| `layers/`（moe_adaptive_expand_lora, base_moe, routers, group_lora, ffn_extension, builder） | `projects/dgs_cl/layers/` | IGA の本体。attn_extension・moe_lora 等の未使用型は移植しない |
| `detectors/gdino_dgs_base.py`・`gdino_dgs.py` | `projects/dgs_cl/` | 凍結・層差し替え・ルーティング呼び出し・stage2 蒸留。dn 差し替えを追加 |
| `heads/gdino_head_inc.py`・`gdino_head_inc_dgs.py` | `projects/dgs_cl/` | 擬似ラベル・蒸留損失の入口 |
| `losses/inter_loss.py`・`qd_distn_loss.py` | `projects/dgs_cl/` | そのまま |
| `hooks/`（weights_transform, merge, domain_predictor, transform_builder, merge_utils） | `projects/dgs_cl/` | そのまま |
| `domain_predictor/`（adaptive, svd, builder） | `projects/dgs_cl/` | そのまま |
| `configs/adaptive_task_mapping.py`・`extract_feats` 系 | `projects/dgs_cl/tools/` | DTG の事前処理 CLI |
| dataset（`coco_increment.py` の stage2 用 distn_cfg 部分） | 必要最小限 | IVLOD はタスク=データセット単位なのでクラス分割機構は不要の見込み。stage2 の擬似ラベル経路が dataset 側に依存する場合のみ移植 |

### 2.2 逸脱（現時点で確定しているもの。すべて記録済みの慣行に従う）

1. **θ0**: 公式 groundingdino_swint_ogc（O365+GoldG+Cap4M）→ 本プロジェクトの MM-GDINO
   （O365+GoldG+GRIT+V3Det）。ZiRa/DitHub 再現（exp_034/037）と同じ扱い。
2. **dn 無効化の実装方式**（§2 の 2）。数学的挙動は同一（dn クエリ・dn 損失が無い）。
3. **contrastive_cfg**: θ0 に合わせ MM-GDINO 既定（log_scale='auto', bias=True）を使う。
4. **分散**: 公式 8 GPU 合計 16 → 本環境/クラスタ 4 GPU。バッチ合計は design で決定。

## 3. 確定した判断（2026-08-21 ユーザー決定）

1. **学習体制: 公式準拠のみ**（12 epoch・stage1 lr 8e-4 / stage2 lr 5e-4・wd 1e-4・
   MultiStepLR [11]・amp・dn 無効）。論文 Table 2（ZCOCO 46.4 / Avg 60.9）との照合が目的。
   本プロジェクト ODinW 枠（3000 iter）での再実行は行わない（既存表と並べる際は
   体制差を但し書きする）。
2. **タスク順: 公式スクリプトの順**（AerialMaritimeDrone → Aquarium → CottontailRabbits →
   EgoHands → NorthAmericaMushroom → Packages → PascalVOC → pistols → pothole →
   Raccoon → ShellfishOpenImages → thermalDogsAndPeople → VehiclesOpenImages）。
   exp_034/037/042 の表（seed 42 順）と直接の横並びはできない点を但し書きする。

## 4. 実装検証（実装完了後・実験前。check_dgs_setup.py）

1. 凍結の検証: 学習対象が lora_ のみ（公式 EPP 1.2% ≈ 本移植のパラメータ数と照合）
2. DTG: 特徴抽出 → 統計 → 2 タスクでの割当（閾値 150 の両側で分岐すること）
3. Group Init: 新 LoRA が base のコピーで始まること／マージ: __rep__ 後の値が
   0.2·base + 0.8·新 に一致すること（数値一致）
4. ルーティング: 学習に使ったタスクの画像で正しいグループが選ばれ、COCO 画像で
   ood（-1 = アダプタ無効）になること
5. dn 無効化: NoDN 差し替え後の 1 step（stage1/stage2 両方）で loss が有限、
   stage2 の擬似ラベル・蒸留損失が乗ること
6. state_dict: タスク間の引き継ぎ（load_from）でグループ構造が正しく復元されること
