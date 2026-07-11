# exp_019: 二重乖離実験（抽出・融合部のみ学習 vs 下流のみ学習）

作成日: 2026-07-07。**承認前・実行不可**。

## 1. 目的（答える問い）

**Q2a: 遠方ドメインへの適応には、モデルのどこを学習可能にする必要があるか。**

モデルを相補的な2つ（抽出・融合部／下流）に分割し、片側だけを学習する2条件を比較する。
- 十分性の検証: 抽出・融合部の学習だけで全学習と同等の適応が得られるか（条件A）
- 代替可能性の検証: 下流の学習で適応を実現できるか（条件B）

両者が対で揃うと、適応能力の帰属が集合の粒度で主張できる（片側だけでは
「大きな集合ならどこでも良い」「容量不足なだけ」という別解釈が残る）。

## 2. 条件

分割の境界 = Feature Enhancer 出力／decoder 入力（「表現を作る部分」と「表現から検出を読み出す部分」の境目）。

| 条件 | tag | 学習する（lr_mult） | 凍結する（lr_mult=0.0） |
|---|---|---|---|
| A: 抽出・融合部のみ | extfusion | `backbone.`(0.1) / `language_model.`(0.1) / `neck.` / `text_feat_map.` / `encoder.` / level_embed | `decoder.` / `bbox_head.` / query_embedding / memory_trans_fc / memory_trans_norm / dn_query_generator |
| B: 下流のみ | downstream | `decoder.` / `bbox_head.` / query_embedding / memory_trans_fc / memory_trans_norm / dn_query_generator | 条件Aの学習対象すべて |

- 凍結は exp_017/018 と同じ方式（optimizer custom_keys の lr_mult=0.0）
- **相補性の検査**: config 作成時に `named_parameters()` を全列挙し、全パラメータが
  ちょうど一方に属することを確認した対応表を `configs/param_assignment.md` に残す
  （帰属が自明でないもの: level_embed→A、dn_query_generator の label 埋め込み→B、を明示）

## 3. プロトコル（exp_018 と同一）

- ドメイン: underwater / videogames / electromagnetic（既存基準線と直接比較のため）
- config ベース: `grounding_dino_swin-t_finetune_8xb4_20e_<domain>.py`、20 epoch、
  MultiStepLR milestone=15、`randomness=dict(seed=0, deterministic=False)`
- 学習: 4GPU 分散（`tools/dist_train.sh`）
- 測定（各 run）:
  - 適応 = 自ドメイン valid mAP（best epoch）
  - ZCOCO = best ckpt を `configs/mm_grounding_dino/eval_base_coco.py` で評価（Q2b の材料として記録）
- 出力: `experiments/exp_019/<domain>_<tag>_work_dir`、ZCOCO は `<domain>_<tag>_zcoco`
- run 数: 2条件 × 3ドメイン = 6 run（各 ≈ 半日〜1日 + ZCOCO 評価 ≈ 30分）
- **実行順序（ユーザー指定 2026-07-07）**: ドメインごとに「下流（downstream）→ 抽出・融合部（extfusion）」
  の順で学習・評価を回す。ドメイン順は underwater → videogames → electromagnetic

## 4. 判定基準（実行前に宣言）

基準線（測定済み）:

| ドメイン | unfrozen（全学習） | 不足の基準線（bert_tfm / neckTfm の高い方） |
|---|---|---|
| underwater | 0.359 | 0.208 |
| videogames | 0.782 | 0.315 |
| electromagnetic | 0.502 | 0.222 |

- **高** = 適応が unfrozen − 0.03 以上（freeze_enc の適応損と同水準を許容）
- **低** = 適応が不足基準線 + 0.03 以下
- **中間** = 上のどちらでもない。十分性の主張には**不足**として扱い、細分化または境界の
  引き直しの根拠として記録する（結果を見てから線を引き直さない）

### 4象限の解釈（3ドメインの多数决ではなく、ドメインごとに記録して総合判断）

| A | B | 主張 | 次の一手 |
|---|---|---|---|
| 高 | 低 | 乖離成立: 適応の源泉は抽出・融合部に帰属（十分性＋代替不可能性） | 内部細分化・Q3 設計へ |
| 高 | 高 | 画像経路上なら場所を問わず容量があれば適応可（テキスト側のみは否定済み） | 選択基準を最小性・忘却量に移す |
| 低 | 低 | 片側では不足＝両側の共学習が必要 | 境界の引き直し（中間集合の探索） |
| 低 | 高 | 需要は下流に局在（予想と逆） | 前提の再点検・学習範囲の設計を反転 |

## 5. この実験が言えないこと（限界）

- 集合の内部（Swin か enhancer か等）の帰属 → 次の細分化ステップの仕事
- 「なぜそこか」の機序 → 特徴プローブ分析（本実験には**含めない**。別実験として要否を判断）
- 逐次設定・他3ドメインへの一般化
- seed=0 単一 run のため、判定が閾値の境界（±0.03 内）に落ちた場合は確定保留とし、
  複数 seed の追試を提案する

## 6. 成果物

- `results/dissociation.md`: 6 run の適応・ZCOCO の実測値、4象限判定、基準線との比較表
- 横断表（module_rollback_summary.md の凍結スイープ表）への2行追記

## 関連
- 問いの定義: 会話記録（2026-07-07 問いの階層 Q1〜Q3）
- 基準線の出典: [[../module_rollback_summary]]（exp_011/017/018）
- 凍結方式の前例: [[../exp_018/design]]
