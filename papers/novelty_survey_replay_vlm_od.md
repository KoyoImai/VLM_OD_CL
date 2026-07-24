# 新規性の文献調査：VLM物体検出の継続学習におけるリプレイ／データ混合

## この記録について
- 目的：中心主張「事前学習済み VLM 物体検出器のドメイン増加型継続学習で，リプレイ／リハーサル／データ混合（過去タスクの exemplar バッファ，事前学習分布の参照データ混合）を**主機構**とした研究は存在しない」の成否を，文献で検証する．
- 実施日：2026-07-20．
- 手法：deep-research ワークフロー（5系統の並列検索 → 22出典取得 → 79主張抽出 → うち25主張を3票の反証検証で確定 → 統合）．検証は全25主張が 3-0（全会一致）で確定，反証・未確定はゼロ．
- 検証の信頼度区分：本記録では出典を2段階で区別する．
  - **[検証済]** = 25主張の反証検証を 3-0 で通過（一次出典の本文・要約に照合）．
  - **[抽出]** = 一次出典から主張抽出はされたが，最終25主張の反証検証セットには含まれない（検索・抽出段階の情報．信頼度はやや低い）．
- caveat（不在の証明ではない）：これは「調査範囲に見当たらない」という不在の主張であり，未索引・同時期・未公開の反例が存在し得る．主要出典の一部（COVD 2605.27116，ABRA 2603.12409）は2026年の未査読 arXiv であり，分野の進行が速い．D-Know は MDPI が HTTP 403 を返し全文未読（検索スニペット経由で2系統から裏取り）．

---

## 結論

**主張は調査範囲において成立する．ただし以下の限定が必要．**

調査した2023〜2026年の文献で，事前学習済み VLM／オープン語彙物体検出器の継続学習手法は**すべて replay-free** だった（該当6手法：ZiRa, DitHub, COVD/NoIn-Det, D-Know, ABRA, CASA）．過去タスクの exemplar バッファや事前学習分布データの混合を主機構とした手法は存在しない．

限定すべき3点：
1. **設定**：VLM検出CLの大半は class/concept-incremental であり，真に domain-incremental なのは D-Know（OD-COD）1件のみ．それも replay-free．よって「domain-incremental 自体が空白」とは言えず，「domain-incremental の VLM検出でリプレイを主機構にした初」が正確．
2. **ベンチマーク**：調査した全手法で **RF100 を使ったものは皆無**（ODinW-13, ODinW-O, Novel-114, COCO, OD-CODB のみ）．
3. **参照データ混合**：事前学習分布データを fine-tuning に混ぜてゼロショット維持を図る機構には CLIP**分類**での先例がある（Anchor-based Robust Finetuning）．ただし**検出には一例もない**．「機構が新しい」ではなく「検出器＋domain-incremental へ持ち込んだのが初」．

### 最も擁護できる新規性の言い方（調査の推奨）
> 事前学習済みオープン語彙物体検出器の継続学習において，リプレイ／リハーサル／事前学習データ混合を**主機構**とした初の手法．既存の VLM検出CL手法は一様に replay-free（再パラメータ化・アダプタ・正則化・パラメータ分離）であり，データ混合・参照リプレイは CLIP分類の fine-tuning と非VLMの class-incremental 検出にしか先例がない．

軸は「リプレイ機構が新しい」ではなく「**この機構をオープン語彙検出器の継続学習に主機構として持ち込んだのが初**」に置く．

### 先行研究が空白にした理由（新規性の論拠として使える）
ZiRa はリプレイを明示的に不適と論じている（「大規模VL事前学習分布を，限られたバッファで代表することは困難」）[検証済]．本設定は事前学習から遠いドメインが逐次増える設定であり，「事前学習パラメータだけでは対応できない」ため，この前提への反論として提案手法を位置付けられる．

---

## 調査した先行研究の記録

### A. VLM／オープン語彙物体検出器の継続学習（すべて replay-free）

| 手法 | 機構 | リプレイ | 設定 | ベンチマーク | 出典 | 検証 |
|---|---|---|---|---|---|---|
| **ZiRa** | 再パラメータ化二重分岐（Reparameterizable Dual Branch）＋ゼロ干渉損失（Zero-interference Loss），frozen Grounding DINO | なし（明示的に否定） | task/domain-incremental（IVLOD を提唱） | COCO + ODinW-13 逐次 | arXiv 2403.01680, NeurIPS 2024 | [検証済] |
| **DitHub** | バージョン管理された LoRA モジュール辞書を重み空間で統合 | なし | class-incremental 寄り（クラス再出現を評価） | ODinW-13 + ODinW-O | arXiv 2503.09271, NeurIPS 2025 | [検証済] |
| **COVD / NoIn-Det** | 表現安定性蒸留＋知識対応パラメータ分離（LLMDet 基盤，視覚エンコーダ凍結，共通概念のテキストのみで表現空間維持） | なし（追加パラメータ・過去画像なし） | concept-incremental（新規概念注入） | Novel-114（新規，114概念×7タスク） | arXiv 2605.27116, 2026 | [検証済] |
| **D-Know（OD-COD）** | ドメイン一般事前分布をスケーラブルなドメイン知識ベースへ分離（知識分離） | なし（過去データ非アクセス） | **domain-incremental**（OD-COD を提唱） | OD-CODB（新規，6ドメイン） | MDPI Appl. Sci. 15(23):12723, 2025 | [検証済]（全文は403で未読，スニペット2系統照合） |
| **ABRA** | 重み空間の幾何輸送（source/target ドメイン専門家を整列） | なし（source画像・exemplar・混合なし） | domain転移 | — | arXiv 2603.12409, 2026 | [検証済] |
| **CASA** | 凍結共有属性＋割当行列，OWL-ViT 基盤，パラメータ増 約0.7% | なし（画像バッファなし） | class-incremental | COCO（2相/多相） | arXiv 2410.05804, 2024 | [検証済] |

補足：
- 最も設定が近いのは **ZiRa**（Grounding DINO 基盤，ODinW-13 逐次，COCO でゼロショット維持を測る）．本設定とほぼ同じだが replay-free なので，新規性を弱めるのではなく補強する．
- 真に domain-incremental なのは **D-Know** のみ．最も近い先行研究のため全文入手が別途必要（補助的な蒸留/リプレイ成分の有無は全文未確認．ただし「過去データ非アクセス」の記述から exemplar バッファは排除される）．
- **MoE-Adapters**（arXiv 2403.11549, CVPR 2024）は rehearsal-free と明記されており，COVD の比較対象群（ZiRa, SGVF, MoE-Adapters, LADA, C-CLIP, KeepLoRA）はすべて replay-free [検証済]．この部分野の比較景色は一様に replay-free．

### B. 反例ではないもの（VLM を名乗るが該当しない）

| 手法 | なぜ反例でないか | 出典 | 検証 |
|---|---|---|---|
| **VLM-PL** | VLM（Ferret）は疑似GTの検証器として yes/no を返すだけ．継続学習される検出器は closed-set の Deformable DETR（ResNet-50）．オープン語彙VLM検出器の継続学習ではない | arXiv 2403.05346, CVPRW 2024 | [検証済] |

### C. 参照データ混合／ロバスト fine-tuning（CLIP分類のみ，検出には先例なし）

| 手法 | 機構 | リプレイ/混合 | 対象 | 出典 | 検証 |
|---|---|---|---|---|---|
| **Anchor-based Robust Finetuning (ARF)** | 事前学習類似データから image-text ペアをアンカーとして混合し，ゼロショット/OOD維持 | **あり（参照データ混合）** | CLIP**分類**（ドメインシフト＋未学習カテゴリのゼロショット認識）．検出成分なし | arXiv 2404.06244, CVPR 2024 | [検証済] |
| **WiSE-FT** | ゼロショットと fine-tuned の重みを事後線形補間 | なし（重み空間） | CLIP分類 | arXiv 2109.01903, CVPR 2022 | [検証済] |
| **LDIFS** | 特徴空間での L2 距離正則化（凍結事前学習モデルに近接維持） | なし（正則化） | CLIP等（分類） | arXiv 2308.13320, TMLR 2024 | [検証済] |

補足：ARF が主張のデータ混合案に最も近い先行研究．ただし CLIP分類に限定され，検出には持ち込まれていない → 本研究の空白であり，かつ「検出で参照混合がゼロショット維持に効くか」は誰も未検証．

### D. 隣接領域（リプレイの先例はあるが，オープン語彙検出器ではない）

| 手法 | 機構 | 設定 | 出典 | 検証 |
|---|---|---|---|---|
| **CL-DETR** | exemplar replay＋知識蒸留＋較正 | class-incremental（非VLM，Deformable DETR/UP-DETR，COCO） | arXiv 2304.03110, CVPR 2023 | [検証済] |
| **Generative Negative Text Replay** | 生成テキストリプレイ | 継続対比VLP（ゼロショット分類・画像テキスト検索，検出でない） | arXiv 2210.17322, ECCV 2022 | [検証済] |
| **ERD** | 応答（分類/回帰ヘッド）の知識蒸留 | class-incremental 検出（非VLM，リプレイ不使用） | arXiv 2204.02136, CVPR 2022 | [抽出] |
| **SDDGR** | 拡散モデルによる生成リプレイ（旧クラスの合成画像） | class-incremental 検出（非VLM，COCO） | arXiv 2402.17323, CVPR 2024 | [抽出] |

### E. LLM／基盤モデルの継続事前学習におけるリプレイ・混合比率（分野は異なるが，本手法の着想源）

| 出典 | 要点 | 検証 |
|---|---|---|
| Ibrahim et al. "Simple and Scalable Strategies to Continually Pre-train LLMs"（arXiv 2403.08763, TMLR 2024） | LR の re-warming/re-decaying＋過去事前学習データのリプレイで，全データ再学習と同等．リプレイ比率 1%/5% 等を検証 | [抽出] |
| arXiv 2407.17467（EMNLP 2024） | 一般（事前学習）コーパスとドメインコーパスの混合が忘却対策として確立 | [抽出] |
| arXiv 2406.01375 | 一般/ドメインの混合比率選択が中心的な未解決問題 | [抽出] |
| TiC-CLIP（arXiv 2310.16226, ICLR 2024） | CLIP の継続事前学習でリプレイを主機構（時系列増分，分類/検索） | [抽出] |

補足：リプレイ・データ混合が忘却対策として確立しているのは LLM／CLIP 継続（事前）学習であり，**オープン語彙検出器には未適用**．本研究は LLM継続事前学習のデータミキシングの知見を，この空白（VLM検出のドメイン増加継続学習）へ持ち込む位置付け．

---

## 研究計画に効く未解決の問い（調査が挙げたもの）

1. **参照データ混合は「検出で」ゼロショット維持に効くか**：先例（ARF）は CLIP分類で有効を示すが，オープン語彙検出器では誰も検証していない．新規性の空白であると同時に，最初に測るべき経験的問い（三者混合＝reference/current/past の最初の決定実験と一致）．
2. **RF100 domain-incremental の VLM検出CL 評価は前例なし**：replay の有無を問わず RF100 を使った VLM検出CLは見当たらず，本設定・評価自体が新しい．
3. **D-Know の OD-COD 設定と本研究の RF100 domain-incremental の異同**：最も近い先行研究．全文アクセスがブロックされ，リプレイ/リハーサルの ablation の有無を含め関係が未確定．全文入手が必要．
4. **2026年の同時期プレプリント**：COVD/ABRA 以外に，オープン語彙検出器CLへリプレイ/データ混合を適用する同時期・未来の研究が，より広い/後の検索で surface し得る．

## 主要出典URL一覧
- ZiRa: https://arxiv.org/pdf/2403.01680
- DitHub: https://arxiv.org/pdf/2503.09271
- COVD/NoIn-Det: https://arxiv.org/html/2605.27116
- D-Know: https://www.mdpi.com/2076-3417/15/23/12723
- ABRA: https://arxiv.org/pdf/2603.12409
- CASA: https://arxiv.org/pdf/2410.05804
- VLM-PL: https://arxiv.org/pdf/2403.05346
- ARF: https://arxiv.org/pdf/2404.06244
- WiSE-FT: https://arxiv.org/abs/2109.01903
- LDIFS: https://arxiv.org/abs/2308.13320
- CL-DETR: https://arxiv.org/abs/2304.03110
- Generative Negative Text Replay: https://arxiv.org/pdf/2210.17322
- ERD: https://arxiv.org/abs/2204.02136
- SDDGR: https://arxiv.org/abs/2402.17323
- Ibrahim et al. (LLM CPT): https://arxiv.org/abs/2403.08763
- LLM CPT mixing (EMNLP 2024): https://arxiv.org/abs/2407.17467
- LLM CPT mixture ratio: https://arxiv.org/abs/2406.01375
- TiC-CLIP: https://arxiv.org/abs/2310.16226
