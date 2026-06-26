# ZSCL: Preventing Zero-Shot Transfer Degradation in Continual Learning of CLIP（要約）

- 出典: Zheng et al. *Preventing Zero-Shot Transfer Degradation in Continual Learning of Vision-Language Models*. ICCV 2023.
- arXiv: 2303.06628
- 領域: 基盤モデル適応／ゼロショット保持（既調査4領域の外側・ZCOCO保持の最重要先行）

## 中核アイデア
CLIP を逐次 fine-tune すると**ゼロショット転移能力が破滅的に劣化**する（＝ZCOCO低下の CLIP 版を実証）。
対策 **ZSCL**:
1. **特徴空間蒸留**: 現在モデルと**初期(事前学習)モデル**の間で蒸留。蒸留用の **reference dataset** は
   「意味的多様性があればよく、ラベル不要・事前学習に含まれる必要なし・画像テキスト対応も不要」。
2. **重み平均**（パラメータ空間の正則化）を併用。

## 本研究への含意
- **【手法・最重要】ZCOCO 保持は直交化以外の機序でも達成可能**。
  MM-Grounding DINO の**凍結事前学習特徴に対する特徴蒸留**を、未ラベルの意味的多様な参照画像で行うことで ZCOCO を保護できる可能性。
  → 提案手法(InfLoRA型)に**追加できる補完成分**であり、強力な**ベースライン/アブレーション**でもある。
- **【分析設計】ゼロショット/転移指標(ZCOCO mAP)を各ドメイン精度とは別軸でタスク系列を通して追跡**すべき、という設計指針。
- caveat: **CLIP分類での検証**。grounded detection への外挿は未実証（要・我々の検証）。ZSCL の完全版は重み平均も使う。

## 関連
- 重み平均系: [[WiSE-FT_summary]] [[ModelSoups_summary]] / 推論時ルーティング: [[MoE-Adapters_summary]]
- 拡張サーベイ: [[../../experiments/exp_006.5/related_work_survey_extended]]
</content>
