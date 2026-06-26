# Textual Inversion for Efficient Adaptation of OVD Without Forgetting（要約）

- 出典: Ruis, Burghouts, Kuijf. *Textual Inversion for Efficient Adaptation of Open-Vocabulary Object Detectors Without Forgetting*. 2025.
- arXiv: 2508.05323
- 領域: OVD適応・忘却抑制（領域8精査・**直接の競合/ベースライン**）

## 中核アイデア
拡散モデルの **Textual Inversion** をオープン語彙検出に応用。
VLM全体を fine-tune せず、**少数例(≥3 instances)から新規/改良トークン埋め込みのみを学習**し、本体重みは凍結。
→ ゼロショット能力とベンチ性能を保ったまま、適応コストをトークン埋め込み次元のみに圧縮。

## 本研究への含意（手法・ベースライン）
- **【重要・別パラダイムの競合/ベースライン】**LoRA(重み差分)ではなく**テキスト埋め込み側のみを適応**して
  忘却を避ける路線。我々の「凍結＋低ランク差分」と直接対比でき、**必須の比較ベースライン**。
- **【手法アイデア】**few-shot 適応・プロンプト/トークン適応を C3 に組み込む余地。本体凍結でZCOCO保持と整合的。
- caveat: 新規概念の語彙適応が主で、ドメイン増加の逐次CL・過去ドメイン保持は本研究側で評価が必要。

## 関連
- LoRA系適応との対比: ../LoRA-CL/InfLoRA_summary / プロンプト系: ../LoRA-CL/CODA-Prompt_summary
</content>
