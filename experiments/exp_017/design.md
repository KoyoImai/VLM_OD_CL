# exp_017: 射影アダプタのみ学習（neck ＋ text_feat_map）— 低忘却適応の実証

## 目的
exp_013/014 のロールバック分析で、**neck / text_feat_map（射影アダプタ）は「大きく動くが COCO を害さない」inert**、
一方 **忘却源は encoder の画像 self-attn** と判明した。exp_017 はこの知見を**構成的に検証**する：
**画像/テキスト backbone・Feature Enhancer 以降を全て凍結し、射影アダプタ（neck＋text_feat_map, 計18 params）だけを学習**する。

- これは重み合成でなく**実学習**（exp_010/011 と同種の FT）。能力軸: A（ZCOCO保持）と C（適応）。
- **仮説**: 忘却源（encoder 画像 self-attn 等）を凍結するため **ZCOCO はほぼ θ0=0.504 を保持**。
  適応は限定的だが、18 params の射影調整だけで frozen/unfrozen にどこまで迫れるかを測る。
- 継続学習の**軽量・低忘却戦略**（射影のみ適応）の有効性を実データで判定する。

## 学習設定（凍結機構）
標準 finetune config（`grounding_dino_swin-t_finetune_8xb4_20e_{domain}.py`）は既に backbone/language_model を
`lr_mult=0.0` で凍結している。exp_017 はこれに以下を追加凍結し、**neck と text_feat_map のみ学習可能**にする。

- **凍結（lr_mult=0.0）**: backbone, language_model, **encoder, decoder, bbox_head,
  memory_trans_fc, memory_trans_norm, query_embedding, level_embed, dn_query_generator**（計10モジュール）。
- **学習（既定 lr_mult=1.0, lr=1e-4）**: neck（ChannelMapper, 16）＋ text_feat_map（Linear, 2）＝**18 params**。
- 検証済み: custom_keys の substring 一致で学習対象は neck.*/text_feat_map.* のみ（891 凍結・混入なし）。
- **seed=0 を明示**（`randomness=dict(seed=0)`。標準 config は未設定＝[[always-fix-seed-0]] に従い必須）。
- スケジュール: 20 epoch・lr=1e-4・MultiStepLR(milestone=15)（frozen/unfrozen と同一＝比較可能）。

### config（各ドメイン、experiments/exp_017/configs/）
`neck_tfm_only_{domain}.py` = 該当 finetune config を `_base_` に、上記 custom_keys ＋ seed=0 を上書き。

## 対象ドメイン
underwater / aerial / microscopic / videogames / documents / electromagnetic（6）。

## 評価
- **適応**: 学習中の domain valid で `best_coco_bbox_mAP` が保存される＝適応 mAP をそのまま採用。
- **ZCOCO**: 学習後、best ckpt を `eval_base_coco.py`（(800,1333)・4GPU）で評価。
- 出力: `experiments/exp_017/{domain}_neckTfm_work_dir/`（学習・best ckpt・適応 mAP）、
  `experiments/exp_017/{domain}_neckTfm_zcoco/`（ZCOCO）。

## 比較（実 mAP）
[[../module_rollback_summary]] の frozen / unfrozen / θ0 と同じ表に **neckTfm-only** 列を追加して対比：
- **ZCOCO**: neckTfm-only ≈ θ0(0.504) に近ければ「射影のみ適応は忘却をほぼ起こさない」を実証。
- **適応**: frozen(0.34〜0.72) / unfrozen(0.36〜0.78) に対し neckTfm-only がどこまで届くか。
  18 params で得られる適応の上限＝「射影アダプタが担える適応能力」を定量化。

## 解釈の枠組み
- neckTfm-only は **frozen の下位集合**（frozen は neck＋encoder＋decoder＋head を学習、exp_017 は neck＋tfm のみ）。
  → 適応は frozen 以下になるはず。その差 = 「検出経路（encoder以降）の学習が適応に与える寄与」の裏返し。
- ZCOCO は frozen(0.28〜0.39) より高く θ0 寄りになるはず（検出経路を凍結したため）。
  → **「適応をどれだけ諦めれば、忘却をどれだけ防げるか」のトレードオフ点**を1つ与える。

## コスト
実学習 6ドメイン（各 20ep・4GPU ≈ 4時間）＝ **約24時間**（逐次）＋ ZCOCO 評価6本（~40分）。
※ backward は凍結層も通るため wall-clock は unfrozen とほぼ同等。

## 段階的方針
1. まず1ドメイン（例: aerial）で学習が意図通り（neck/tfm のみ更新・loss 低下・COCO 保持）を確認。
2. 問題なければ残り5ドメインを逐次学習。
3. 全ドメインの適応＋ZCOCO を [[../module_rollback_summary]] に neckTfm-only 列として統合。

## 承認ゲート
行動原理①②に従い、本 design.md の承認後に学習・評価を実行する。
実学習で高コスト（~24h）のため、**まず1ドメインで検証 → 承認 → 残り**の段階実行を推奨。

## 関連
- 射影が inert の根拠: [[../exp_013/results/feature_rollback_imgfeat_vs_txtfeat]]（neck/tfm 回復/ドリフト最小）
- 忘却源＝encoder画像self-attn: [[../exp_014/results/feature_enhancer_rollback]]
- 比較基準の統合表: [[../module_rollback_summary]]
- 凍結は選択の問題: [[encoder-freezing-is-a-choice]]
