# 参照データ候補の調査記録（リプレイ用 reference data）

- 作成日: 2026-07-21
- 目的: 継続学習のリプレイで θ0 の汎用・ゼロショット知識（ZCOCO）を守るための「参照データ」候補を整理し、決定する。
- 制約: (1) **OD（検出）形式**であること（過去実験との一貫性のため。VG=グラウンディング形式は除外）、(2) **COCO2017-val と重複しない**こと（重複すると ZCOCO のゼロショット性が壊れる）。
- 対象チェックポイント: `grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det`（ZCOCO = θ0 で 0.504）。事前学習構成 = Objects365v1 + GoldG + GRIT(9M) + V3Det。

## 決定

- **第一候補（使用予定）: Objects365v1**
- **第二候補: V3Det**
- どちらも事前学習に含まれる OD データ。

---

## 1. 事前学習4データセットの特色

| データセット | 形式 | 訓練画像数 | クラス数 | ボックス/注釈 | ドメイン | 参照可否 |
|---|---|---:|---:|---:|---|---|
| Objects365 v1 | **OD** | ~638K | 365 | ~10M | 日常物体（Flickr 由来） | 可 |
| V3Det | **OD / OVD** | 183,354 | 13,204 | ~1.75M | 広い細粒度・実世界（Flickr/Bamboo 由来） | 可 |
| GoldG (GQA + Flickr30k) | VG（グラウンディング） | GQA ~113K / Flickr30k ~31K | 自由文 | — | 日常シーン | 不可（OD でない。COCO 画像は除去済み） |
| GRIT (grit9m) | VG（擬似ラベル） | ~9M | 自由文 | ~137M | 一般 Web 画像 | 不可（OD でない） |

- OD 形式で参照に使えるのは **Objects365v1 と V3Det** の2つ。GoldG・GRIT はグラウンディング形式なので OD 参照には使えない。

## 2. OD 参照候補の一覧（事前学習内・外・除外）

| 候補 | 事前学習内 | 画像数 | クラス数 | ボックス数 | COCO 重複 | 事前学習分布への近さ |
|---|---|---:|---:|---:|---|---|
| Objects365 v1 | ○ | ~638K | 365 | ~10M | 画像単位の非重複は未証明（注） | 非常に近い（COCO80 ⊂ O365 365） |
| V3Det | ○ | 183,354 | 13,204 | ~1.75M | COCO 由来でない | 近い（超多語彙・細粒度） |
| Objects365 v2 | ✕（v1 を包含） | ~1,742,289 | 365 | ~30M | 同上（未証明）＋ v1 画像を再リプレイ | 非常に近い |
| Open Images V6/V7 | ✕ | 1,743,042 | 600 | 14,610,229 | COCO 由来でない（網羅監査なし） | 近い（広く・ノイジー） |
| ADE20K | ✕ | 20,210 | 150 | ~390K（概算） | なし | 中（シーン解析） |
| BDD100K | ✕ | 70,000 | 10 | ~1.84M | なし | 遠い（運転） |
| nuImages | ✕ | 67,279 | 23 | ~800K | なし | 遠い（運転） |
| Mapillary Vistas | ✕ | ~18,000 | 66 | 密マスク | なし | 遠い（街路） |
| **LVIS v1.0** | ✕ | 100,170 | 1,203 | ~2M | **COCO 画像そのもの（minival ≡ val2017）** | **除外** |
| **Visual Genome** | ✕ | ~108,077 | 自由 | ~3.8M | **~49〜51K が COCO 画像** | **除外** |
| **COCO** | ✕ | — | 80 | — | 評価セット | **除外** |

## 3. 決定の根拠（共同研究者の評価）

- **ZCOCO 妥当性の観点で、実際の事前学習 OD データ（Objects365v1 / V3Det）が最も安全。** θ0 は既にこれらを学習済みで、ZCOCO=0.504 はその前提の上で「ゼロショット」と扱われている。これらをリプレイしても新たな COCO-val 混入を持ち込まない。事前学習外データ（Open Images 等）を使うと、θ0 が見ていない新規データを CL 中に学習することになり、COCO-val 混入の新規リスクが生じる。
- **Objects365v1 を第一候補とする理由**: 汎用日常物体、COCO 80 クラスを包含、ZCOCO（COCO 80クラス）が測る知識に最も近い。「事前学習コーパスをリプレイする」という LLM 継続事前学習のレシピに最も素直に対応する。
- **V3Det を第二候補とする理由**: 同じく事前学習 OD だが、13,204 クラスの超多語彙・細粒度で、汎用日常物体というより語彙の広さの保護に寄る。ZCOCO（80 common クラス）との整合は Objects365v1 より弱い。

## 4. 注意点（確定前に潰すこと）

- **COCO-val ハッシュ照合**: Objects365（v1/v2）・Open Images はいずれも Flickr 由来で、COCO2017-val との画像単位の非重複が一次資料で証明できない（カテゴリ集合の関係 O365 ⊃ COCO80 のみ文書化）。厳密には、確定前に参照データと COCO2017-val の画像ハッシュ照合を1回かけるのが安全。
- **除外の確証**: LVIS は CONFIRMED で COCO 画像そのもの（minival ≡ val2017）。Visual Genome は ~49〜51K が COCO 画像を含む（`coco_id` を持つ）。いずれも参照に使うと COCO-val がリークする。

## 5. 出典

- MM-Grounding DINO 論文 Table 2: https://arxiv.org/html/2401.02361v2
- リポジトリ内: `configs/mm_grounding_dino/dataset_prepare.md`（事前学習・追加データセットの構成と入手）、`configs/mm_grounding_dino/README.md`（ZCOCO=0.504 の基準）
- Objects365: https://www.objects365.org/ ／ V3Det: https://v3det.openxlab.org.cn/ ／ Open Images: https://storage.googleapis.com/openimages/web/index.html ／ LVIS: https://www.lvisdataset.org/ ／ Visual Genome: https://arxiv.org/abs/1602.07332

## 確認できていない点

- Objects365・Open Images の COCO2017-val との画像単位重複（一次資料で未証明。要オフライン照合）。
- Visual Genome ↔ COCO の正確な重複枚数（~51K の桁は信頼できるが正確値は 48,749〜51,498 と資料で揺れる）。
- ADE20K の総ボックス数（~390K は1画像あたり平均からの概算）。
