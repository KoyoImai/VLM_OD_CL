# Three Scenarios for Continual Learning（要約）

- 出典: van de Ven, Tuytelaars, Tolias. *Three types of incremental learning*. Nature Machine Intelligence 2022.
- arXiv/DOI: 10.1038/s42256-022-00568-3
- 領域: CL基礎・全体俯瞰（既調査4領域の外側）

## 中核アイデア
継続学習を3シナリオに分類する標準的枠組み。
- **Task-IL**: タスクIDがテスト時に既知（タスク別ヘッド可）。最易。
- **Domain-IL**: ラベル空間（タスク構造）は同一だが**入力分布が変化**、テスト時に**タスクIDは与えられない**。
- **Class-IL**: 新クラスが逐次増え、全クラスを区別する必要。最難。

重要な主張: **「Domain-IL では忘却を設計だけで防ぐことは不可能（preventing forgetting by design is not possible）」**。

## 本研究への含意（位置づけ・分析設計）
- 本研究は**形式的に Domain-IL に最も近い**（ドメイン＝入力分布変化、推論時はクラス名テキストで評価＝明示的タスクID無し前提）。
  → 「忘却が原理的に難しい設定」であることを taxonomy で正当化でき、ZCOCO/過去ドメイン保持の困難さを主張する土台になる。
- **caveat（検証者指摘）**: RF100 各ドメインは**非重複のクラス集合**を持つため、厳密には Domain-IL（同一ラベル空間）ではなく
  **domain/class-incremental のハイブリッド**。論文では「ドメイン増加CL」と呼ぶが、評価プロトコル設計時にこの差異を明記すべき。

## 関連
- 上位サーベイ: [[CL-Survey-TPAMI_summary]] / 拡張サーベイ [[../../experiments/exp_006.5/related_work_survey_extended]]
</content>
</invoke>
