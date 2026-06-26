# Linear Mode Connectivity in Multitask and Continual Learning（要約）

- 出典: Mirzadeh, Farajtabar, Gorur, Pascanu, Ghasemzadeh. *Linear Mode Connectivity in Multitask and Continual Learning*. ICLR 2021.
- arXiv: 2010.04495
- 領域: 損失地形・機序分析（領域8精査）

## 中核アイデア
**逐次学習解**と**マルチタスク(joint)解**は、**同一初期値なら低損失の線形経路で接続できる**（linear mode connectivity）。
この幾何的性質を使い、逐次解をマルチタスク解の特性へ整合させる正則化型CL手法(MC-SGD)を提案。
→ 忘却は「解が幾何的に非互換」だからではなく、**最適化軌道が分岐する**ことに起因。

## 本研究への含意（分析A4＋手法/ベースライン保持）
- **【分析設計 A4】**各ドメイン適応解↔COCO凍結解、ドメイン間解 の**線形補間経路の損失バリア**を測定。
  バリアが低ければ **WiSE-FT/model soups の重み補間・平均が有効**な前提が満たされる（保持機構の成否を事前判定）。
- **【手法】**joint(上限)解へ向けて逐次解を引き寄せる正則化＝忘却抑制の追加成分。
- caveat: 「同一初期値」が前提（凍結事前学習を共通初期とする本設定と整合的）。分類での検証。

## 関連
- 重み補間/平均: [[../VLM-adaptation/WiSE-FT_summary]] [[../VLM-adaptation/ModelSoups_summary]] / 平坦性: [[C-Flat_summary]]
</content>
