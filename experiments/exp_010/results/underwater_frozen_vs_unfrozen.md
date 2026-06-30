# exp_010 結果まとめ：underwater の frozen vs unfrozen（Image/Text Backbone も学習）

作成日: 2026-06-28。**すべて seed=0・同一config系統（標準 finetune config 継承）・発散なし**。
設計: [[../design.md]]。アーキ: [[../../setup_and_code_analysis/code_analysis/mmgrounding_dino_architecture]]。

## 設定
- ドメイン: underwater 単独FT、事前学習済みモデルから。
- 唯一の差: backbone(Swin)・language_model(BERT) の `lr_mult`。
  - **frozen** = 0.0（従来設定）/ **unfrozen** = 0.1（実効1e-5相当、Swin/BERT も学習）。
- 据え置き: base lr=1e-4 / 実効64(per-GPU8×4×累積2) / 20ep[15] / AdamW wd=1e-4 / clip_grad(0.1) / 評価(800,1333)。
- **seed=0 固定**（`randomness=dict(deterministic=False, seed=0)`）。

## 結果（適応=自ドメイン mAP / ZCOCO=COCO 保持 mAP）

| 設定 | 適応(underwater) | ZCOCO(COCO保持) | best epoch |
|---|---|---|---|
| **frozen × seed=0** (lr_mult=0) | 0.337 | 0.389 | 20 |
| **unfrozen × seed=0** (lr_mult=0.1) | **0.359** | **0.407** | 18 |

参考（基準値）: zero-shot 0.051 / 0.504。

## 主要な所見

1. **backbone を 0.1×学習させた方が、適応も COCO保持も両方良い＝Pareto 改善**（0.337/0.389 → 0.359/0.407）。
   「学習パラメータを増やすほど忘却が増える」という素朴な予想と**逆**。
2. **BERT を学習させても安定**（grad_norm に inf/nan なし、val mAP 単調収束）。lr_mult=0.1 は妥当。

## 解釈の仮説（n=1・要検証）
exp_004 の知見「凍結すると適応↓・忘却↑、**忘却源＝共有検出経路**」と整合的に読める：
- **backbone 凍結** → 適応の負荷が全て共有検出経路に集中し歪む → COCO を忘れる。
- **backbone 0.1×学習** → 適応を backbone が分担し共有検出経路の負荷が減る → 忘却が小さい。
→「**適応をどこに吸収させるか**」が忘却を左右する、という手法（LoRA挿入箇所）への示唆。

## 注意・訂正

- **以前のランダムseed時の解釈は誤り**：旧 unfrozen(ランダムseed=1187114715) は 0.359/**0.404**。これを exp_004/005 の
  frozen 0.414 と比べ「適応の代償に忘却が増える（トレードオフ）」と報告したが、あれは**別config系統・別seedとの比較**で不当。
  同一config・同一seed=0 で取り直すと frozen=0.389 となり、**Pareto 改善に逆転**した。seed=0 で取り直した意義がここに出た。
- **cross-exp の系統差に注意**：exp_010 frozen(0.337/0.389) は exp_004 frozen(ランダムseed, 0.316/0.414) と乖離。
  config 系統の違い（exp_004 は exp_003 config 継承、exp_010 は標準 config 継承）と推定。
  → **今後 underwater の frozen 正本は exp_010 の seed=0・標準config 値（0.337/0.389）**とし、exp_003/004 との直接比較は系統差に注意。

## 成果物
- 学習: `underwater_unfrozen_work_dir/`（lr_mult=0.1）, `underwater_frozen_work_dir/`（lr_mult=0.0）。
- 評価: `underwater_unfrozen_coco/`, `underwater_frozen_coco/`（ZCOCO）。
- 旧 `underwater_work_dir/`（ランダムseed unfrozen, 0.359/0.404）は**破棄扱い・参考保持**。

## 次のステップ
- **exp_011**: 残り5ドメイン（aerial / microscopic / videogames / documents / electromagnetic）を **unfrozen×seed=0** で学習し、
  Pareto 改善の一般性とドメイン距離依存を確認。frozen×seed=0 は exp_005（他5ドメイン, seed=0）を流用。

## 関連
- [[../design.md]] / 機序: [[../../experiments/exp_004/design.md]] / 能力軸の議論: [[../../exp_009/minutes]]
