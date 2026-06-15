# exp_001 メモ：COCO2017 + RF100 各ドメインの zero-shot 性能

## 結論
事前学習 MM-Grounding DINO の zero-shot 性能は、**COCO で mAP=0.504** と高い一方、
**RF100 の専門ドメインでは 0.002〜0.051** と壊滅的に低い。
これが本研究の出発点（lower bound）であり、課題1「事前学習に乏しいドメインでの性能低下」を定量的に裏付ける。

| 対象 | mAP |
|---|---:|
| COCO2017 | 0.504 |
| underwater | 0.051 |
| aerial | 0.034 |
| electromagnetic | 0.024 |
| videogames | 0.016 |
| documents | 0.006 |
| microscopic | 0.002 |

→ COCO と RF100 ドメインの間に **約 10〜250 倍**の性能差。RF100-VL 論文の知見
（ODinW-13 49.2 → RF100-VL 15.7 への低下）とも整合し、RF100 が VLM にとって極めて難しいことを再確認。

## ドメイン別の観察
- **underwater(0.051) / aerial(0.034)**: 比較的高い。`fish/shark/car/boat` 等、COCO/事前学習語彙に
  近い汎用クラスを含むため。aerial は mAP_l=0.019 と大物体が弱く、小物体(mAP_s=0.049)の方が高い特異な傾向。
- **electromagnetic(0.024)**: X線/熱画像など。中程度。
- **videogames(0.016)**: キャラクター名など固有名詞クラスが多く、テキスト対照がほぼ機能しない。
  mAP_l=0.065 と大物体のみ拾える。
- **documents(0.006)**: 表/図/レイアウト要素。自然画像と乖離が大きい。
- **microscopic(0.002)**: 細胞・細菌など。**ほぼゼロ**。事前学習の概念・見えと最も乖離するドメイン。

## 妥当性
- 全 7 評価が分散推論で完走（exit 0）。COCO・underwater の健全性チェック双方通過。
- videogames は max_text_len=512 設定下で完走（265 tokens を処理できている）。

## 次の問い・改善案
1. **fine-tune による伸び（exp_002 候補）**: 最も低い microscopic と最も高い underwater を fine-tune し、
   「lower bound からどれだけ伸ばせるか（適応余地）」を両極端で測る。これが継続学習の upper bound。
2. **忘却の基準確立**: 本表の COCO=0.504 が「忘却で失う上限」。逐次 fine-tune 後に COCO を再評価して
   ZCOCO 低下量を測るプロトコル（ZiRa の hAP）を exp で固定する。
3. **プロンプト改善の効果切り分け**: documents/microscopic の専門クラス名を説明的表現に置換した
   zero-shot 再評価で、低性能が「未知概念」か「テキスト表現」かを分離する。
4. **ドメイン提示順序の設計**: 継続学習では順序が忘却に影響。zero-shot 性能が高い順/低い順/ランダムなど、
   順序が最終 Avg・ZCOCO に与える影響を検証する設計を準備。

## 関連
- 出発点表は [[../../papers/ZiRaGroundingDINO_summary]] の ZCOCO/Avg/hAP 評価系に対応。
- RF100 の難しさは [[../../papers/Roboflow100_summary]] の知見と整合。
