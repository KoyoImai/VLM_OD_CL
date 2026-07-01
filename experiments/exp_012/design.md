# exp_012: モジュール入れ替えによる ZCOCO 変化のモジュール帰属（第1 swap: Swin→θ0）

## 目的
exp_010/011 で、backbone まで学習（unfrozen）すると ZCOCO（COCO ゼロショット保持）が
**underwater/aerial では低下しない（むしろ改善）／他4ドメインではより低下**した。
この **ZCOCO 変化が「どのモジュールの重み変化」に起因するか**を、**再学習なしの重みすげ替え（module swap）**で帰属する。
本 design は **第1 swap（Swin backbone を θ0 に戻す）** のみを対象とする（段階的に進める方針）。

- これは「ヘッド差し替えで対照モデルを新規学習する」案とは別物。**学習は一切行わず**、既存の学習済み重みを後付けで合成して ZCOCO を測るだけ。
- 能力軸: A（事前学習知識保持=ZCOCO）の変化の機序特定。

## 使用する重み（実体確認済み）
- **θ0（事前学習）**: `grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth`（ZCOCO=0.504）
- **θ1_d^U（unfrozen, lr_mult=0.1）** — 帰属対象:
  - underwater: `experiments/exp_010/underwater_unfrozen_work_dir/best_coco_bbox_mAP_epoch_18.pth`
  - aerial: `experiments/exp_011/aerial_unfrozen_work_dir/best_coco_bbox_mAP_epoch_19.pth`
  - microscopic: `.../microscopic_unfrozen_work_dir/best_coco_bbox_mAP_epoch_18.pth`
  - videogames: `.../videogames_unfrozen_work_dir/best_coco_bbox_mAP_epoch_20.pth`
  - documents: `.../documents_unfrozen_work_dir/best_coco_bbox_mAP_epoch_16.pth`
  - electromagnetic: `.../electromagnetic_unfrozen_work_dir/best_coco_bbox_mAP_epoch_19.pth`
- **θ1_d^F（frozen, 参考基準）**: underwater=exp_010 frozen / 他5=exp_005（既測 ZCOCO を流用、再評価不要）。

## 第1 swap の定義（ハイブリッド H_d）
**H_d = θ1_d^U の `backbone.*`（Swin, 187 params）だけを θ0 の値に置換。** それ以外
（`language_model.`=BERT, `neck.`, `encoder.`, `decoder.`, `bbox_head.` 等）は **unfrozen のまま**。
- 接頭辞は実体確認済み: Swin=`backbone.`（187）, BERT=`language_model.`（198）。本 swap で **BERT は触らない**。
- 置換は同一アーキ間の1:1（キー集合一致）。

### 生成手順（学習なし・重み合成のみ）
```python
# 擬似コード
import torch
t0 = torch.load(THETA0)['state_dict'] if 'state_dict' in ... else ...
tu = torch.load(THETA1_U)            # unfrozen
sd = tu['state_dict'].copy()
for k in list(sd):
    if k.startswith('backbone.'):
        sd[k] = t0[k]               # Swin だけ θ0 に戻す
tu['state_dict'] = sd
torch.save(tu, OUT_HYBRID_d)        # experiments/exp_012/hybrids/{d}_swin_theta0.pth
```
生成スクリプト: `experiments/exp_012/make_hybrid_swin_theta0.py`（サニティ assert 込み・学習なし）。

## 評価
- ZCOCO を `configs/mm_grounding_dino/eval_base_coco.py`（**(800,1333) keep_ratio、これまでと同一**）で測定、4GPU分散。
  ```bash
  bash tools/dist_test.sh configs/mm_grounding_dino/eval_base_coco.py \
    experiments/exp_012/hybrids/{d}_swin_theta0.pth 4 --work-dir experiments/exp_012/{d}_swinθ0_coco
  ```
- サニティ: H に対し θ0 の backbone 同一性（`backbone.*` が θ0 と完全一致、他が θ1_d^U と一致）をロード前に assert。

## 比較表（出力テンプレート）
| ドメイン | θ0 | frozen θ1_d^F | unfrozen θ1_d^U | **H_d (Swin→θ0)** | H−unfrozen の向き |
|---|---|---|---|---|---|
| underwater | 0.504 | 0.389 | 0.407 | ? | ? |
| aerial | 0.504 | 0.318 | 0.369 | ? | ? |
| microscopic | 0.504 | 0.327 | 0.288 | ? | ? |
| videogames | 0.504 | 0.324 | 0.315 | ? | ? |
| documents | 0.504 | 0.275 | 0.211 | ? | ? |
| electromagnetic | 0.504 | 0.331 | 0.269 | ? | ? |

## 解釈ルール（合意済み・符号で読む）
**主証拠は「ZCOCO が低下した4ドメイン（micro/vg/doc/em）」**で読む：

| H_d の ZCOCO（vs unfrozen） | 解釈 | クリーンか |
|---|---|---|
| **上に回復**（frozen/θ0 方向） | backbone の変化が ZCOCO 低下の主因 | ✅ クリーン |
| 低下 / 不変 | 帰属不能（**共適応の不整合**と区別不能） | ❌ 曖昧 → 次 swap へ |

- 根拠: 共適応の不整合は常に性能を**下げる**方向に働くため、**それを乗り越えて「回復」したなら backbone 起因の確証**になる。
- **近ドメイン（underwater/aerial, ZCOCO 上昇側）は単独では曖昧**（「有益な backbone 適応の除去」と「不整合」が同符号）。本 swap では補助的に見るのみ。

## 追補（同 swap で適応も帰属）: Swin 学習のドメイン適応への寄与
同じ hybrid H_d を**各ドメインの valid（適応 mAP）**でも評価し、ZCOCO と対称に
**Swin 学習がドメイン適応(C)にどれだけ寄与したか**を切り分ける。
- 評価: ドメインFT config（自ドメイン valid・(800,1333)）で 4GPU。
  ```bash
  bash tools/dist_test.sh configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_{d}.py \
    experiments/exp_012/hybrids/{d}_swin_theta0.pth 4 --work-dir experiments/exp_012/{d}_swin_theta0_adapt
  ```
  - H_d の bbox_head は unfrozen（domain適応済み）なので num_classes/max_text_len は当該configと整合。
- 比較: frozen 適応 / unfrozen 適応 / **H_d 適応**。
- 解釈（符号・共適応を踏まえる）: H_d 適応 vs unfrozen 適応で、
  - **小さな低下**＝Swin 適応の寄与は小（不整合の handicap があってもなお落ちないなら確証）。
  - **大きな低下**＝Swin 適応寄与 or 不整合の区別困難（＝Swin 寄与の**上限**）。

## 第2 swap: Text Backbone(BERT) を θ0 に戻す（Swin と対称）
**H^BERT_d = θ1_d^U の `language_model.*`（BERT）だけを θ0 に置換。** Swin(`backbone.`) は
**unfrozen のまま（ドメイン適応済み）**、neck/enc/dec/head も unfrozen。
- 接頭辞: BERT=`language_model.`。θ0 は 198 キー、unfrozen は 197（θ0 のみ `...embeddings.position_ids`＝
  定数バッファ・学習対象外）。**swap は両者共通の 197 キーを θ0 値に置換**（position_ids は定数のため省略で無害）。
- 生成: `experiments/exp_012/make_hybrid_bert_theta0.py`（サニティ assert 込み・学習なし）。
- 評価: ZCOCO（eval_base_coco.py）＋ 適応（自ドメインFT config）、いずれも (800,1333)・4GPU。
- 解釈: Swin と同じ符号ルール。**低下4ドメインで ZCOCO が回復すれば BERT 適応が忘却の一因（クリーン）**。
  適応側は同符号で曖昧（クリーンは frozen vs unfrozen）。
- 出力: `hybrids/{d}_bert_theta0.pth`, `{d}_bert_theta0_coco/`, `{d}_bert_theta0_adapt/`。

## 段階的方針
1. 本 swap（6ドメイン）の ZCOCO を測る。
2. **低下4ドメインで「回復」が出るか**を主判定。回復＝backbone 起因のクリーンな帰属。
3. 「低下/不変」で曖昧なら、**不整合を切り分ける追加 swap を別途設計**（本 design の範囲外）。

## コスト
重み合成（CPU, 数秒×6）＋ ZCOCO 評価6本（forwardのみ、新規学習ゼロ）。

## 承認ゲート
行動原理①②に従い、本 design.md の承認後に hybrid 生成・評価を実行する。

## 関連
- 現象の出所: [[../exp_011/results/frozen_vs_unfrozen_6domains]] / [[../exp_010/results/underwater_frozen_vs_unfrozen]]
- アーキ（モジュール構成・接頭辞）: [[../../setup_and_code_analysis/code_analysis/mmgrounding_dino_architecture]]
- 能力軸: [[../exp_009/minutes]]
