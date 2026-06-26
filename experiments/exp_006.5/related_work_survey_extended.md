# 関連研究サーベイ【拡張版】— 既調査4領域の外側

作成日: 2026-06-21。deep-research（24ソース取得・105主張抽出・25主張を3票敵対的検証→23確認/2棄却）に基づく。
目的: [[related_work_survey]]（直球4領域）の調査範囲を**広域化**し、(i) **行うべき分析の設計**、(ii) **手法・ベースラインの発掘** に資する。
個別要約は `papers/CL-foundations/`, `papers/VLM-adaptation/`, `papers/OVD-CL/`, `papers/forgetting-analysis/` 配下。

> 既調査4領域（深掘り済み・本ファイルでは扱わない）: ①LoRA/PEFT-CL, ②CL×物体検出, ③VLM/OVD-CL(ZiRa/DitHub), ④機序分析方法論の一部(GPM/Adam-NSCL/Intrinsic Dim)。

---

## 総括：3本柱に収束
広域調査の結果、4領域の外側で本研究に効く知見は3本柱に整理できる。

1. **基礎・理論**: 本設定は形式的に **Domain-IL**（同一タスク構造・入力分布変化・テスト時タスクID無し）で、
   「設計だけで忘却は防げない」難設定。**NTK/部分空間理論**（NTK重なり行列・OGD頑健性証明・GPM）が
   (a) 提案手法(直交化LoRA)の**理論的支柱**、(b) **分析ツール**（ドメイン間＋COCO-task0 のアラインメント測定で忘却を予測）の両方を与える。
2. **ZCOCO 保持は独立した故障モードであり、直交化以外の機序でも対処可能**（CLIP で実証＝ZSCL）。
   **特徴蒸留(ZSCL)・重み平均/補間(WiSE-FT, model soups)・task arithmetic・推論時ルーティング(MoE-Adapters/DDAS)** が
   ZiRa/DitHub の外側の**必須ベースライン群＆C3の補完成分**。
3. **評価・地形図**: **Roboflow100-VL** が自然なタスク系列かつ適応問題の定量根拠（GDINO 49.2→~16 mAP）。
   OVD頑健性研究(OWL-ViT/YOLO-World/GDINO)・**OWL-ST(自己学習)** が検出側の地形図と可塑性ブースタを与える。

**最大の caveat**: マージ/蒸留/ルーティング系はほぼ **CLIP分類で検証**され、grounded detection への外挿は未実証。
→ ベースライン化には**検出固有の適応**（領域別ルーティング, box/textヘッドのマージ, text埋め込み補間）が必要＝**我々の貢献余地**。

---

## 領域5: 継続学習の基礎・忘却理論（新規）
| 文献 | 中核 | 本研究への含意 |
|---|---|---|
| **Three Scenarios**(van de Ven, Nat.MI22) | Task/Domain/Class-IL の分類。Domain-IL は「設計で忘却を防げない」 | **位置づけ**: 本設定=Domain-IL寄り。忘却の困難さを taxonomy で正当化。[[../../papers/CL-foundations/ThreeScenarios_summary]] |
| **CL Survey**(Wang, TPAMI24) | 手法ファミリ俯瞰。DIL定義 | ベースライン選定の俯瞰。**ただし「5分類」「リプレイ最強」は検証で棄却** [[../../papers/CL-foundations/CL-Survey-TPAMI_summary]] |
| **EWC**(Kirkpatrick, PNAS17) | Fisher重み付き二次ペナルティ | **正則化ベースライン**＋Fisherを重み重要度の**分析ツール**に転用 [[../../papers/CL-foundations/EWC_summary]] |
| **NTK overlap**(Doan, AISTATS21) | タスクアラインメント↑→忘却↑。NTK重なり行列 | **分析の核**: 重なり行列で忘却/ZCOCO低下を予測。直交化の理論的支柱 [[../../papers/forgetting-analysis/NTK-overlap_summary]] |
| **OGD**(Bennani, 2020) | 過去勾配に直交射影、NTKで頑健性証明 | **InfLoRA型直交化の理論的支柱**＋簡易ベースライン [[../../papers/forgetting-analysis/OGD_summary]] |

**棄却された主張（断定しない）**: 「リプレイが全シナリオで唯一の最強」(0-3)、「CLは正準に5分類」(0-3)。
→ リプレイ/アーキテクチャ系は「有力候補」止まり。分類は便宜的に扱う。

## 領域6: 基盤モデル適応・ZCOCO保持（新規・最重要）
| 文献 | 中核 | 本研究への含意（手法/ベースライン） |
|---|---|---|
| **ZSCL**(ICCV23) | 凍結初期モデルへの**特徴蒸留**＋重み平均。参照集合は未ラベル・多様でよい | ZCOCO保持の**最重要先行**。蒸留成分を C3 に追加＆ベースライン化 [[../../papers/VLM-adaptation/ZSCL_summary]] |
| **WiSE-FT**(CVPR22) | ゼロショット重みとFT重みの**線形補間**、推論コスト0 | 安価で強い**ZCOCO保持ベースライン**＋C3候補成分(α制御) [[../../papers/VLM-adaptation/WiSE-FT_summary]] |
| **Model Soups**(ICML22) | 同一初期FTの**重み平均**でOOD/ゼロショット向上 | ドメイン別適応の**マージ系ベースライン**(パラメータ一定) [[../../papers/VLM-adaptation/ModelSoups_summary]] |
| **Task Arithmetic**(ICLR23) | task vector の加算=マージ、減算=アンラーニング | 加算=多ドメイン蓄積ベースライン、減算=**制御忘却の分析ツール** [[../../papers/VLM-adaptation/TaskArithmetic_summary]] |
| **MoE-Adapters/DDAS**(CVPR24) | OOD入力は凍結CLIPへ、in-distはアダプタへ**ルーティング** | ZCOCO保持の**別パラダイム**＋ルーティング系ベースライン [[../../papers/VLM-adaptation/MoE-Adapters_summary]] |

## 領域7: OVD地形図・評価（新規）
| 文献 | 中核 | 本研究への含意 |
|---|---|---|
| **RF100-VL**(2025) | 100データ/564クラス/7ドメイン。GDINO 49.2→~16 mAP | **タスク系列の上流**＋適応問題の定量根拠。medical等が最難 [[../../papers/OVD-CL/RF100-VL_summary]] |
| **OVD Robustness**(2024) | OWL-ViT/YOLO-World/GDINO の分布シフト頑健性横並び | 代替バックボーン/比較対象＋ZCOCO頑健性プロトコル参考 [[../../papers/OVD-CL/OVD-Robustness_summary]] |
| **OWL-ST**(NeurIPS23) | Web規模自己学習でLVIS rare 31.2→44.6 AP | **擬似ラベル・リプレイ**の手掛かり(exemplar無しZCOCO保持) [[../../papers/OVD-CL/OWL-ST_summary]] |

## 領域8: 横断分野（手法/分析に転用しうる・**精査済み 2026-06-21**）
deep-research が取得したが10件の統合 finding に残らなかった9ソースを WebFetch で精査・個別要約済み。**3件が極めて関連高**（★）。

| 文献 | 中核 | 本研究への含意 |
|---|---|---|
| **Probing Representation Forgetting**(CVPR22, 2203.13381) | linear probe で表現の忘却を測り、性能低下と区別 | **分析A2/A3**: 忘却が「表現崩壊」か「ヘッド上書き」かを切り分け [[../../papers/forgetting-analysis/ProbingRepresentationForgetting_summary]] |
| **The Tunnel Effect**(NeurIPS23, 2305.19753) | 深層が extractor＋tunnel(圧縮層, OOD/転移を損なう)に分割 | 層別の LoRA/凍結設計の根拠。ZCOCO劣化を tunnel で説明 [[../../papers/forgetting-analysis/TunnelEffect_summary]] |
| **Linear Mode Connectivity**(ICLR21, 2010.04495) | 逐次解とjoint解は同一初期なら低損失線形経路で接続 | **分析A4**: 補間バリア測定でWiSE-FT/soupsの成否を事前判定 [[../../papers/forgetting-analysis/LinearModeConnectivity_summary]] |
| **C-Flat**(2024, 2404.00986) | CLで平坦極小を狙うplug-and-play最適化 | 手法に上乗せ＋sharpness↔忘却の分析(A4) [[../../papers/forgetting-analysis/C-Flat_summary]] |
| **PCGrad**(NeurIPS20, 2001.06782) | 勾配コサイン<0の衝突成分を法平面へ射影除去 | **分析**: ドメイン間勾配衝突率↔忘却。ベースライン＋統合候補 [[../../papers/forgetting-analysis/PCGrad_summary]] |
| ★**Low-rank Orthogonal Subspaces**(NeurIPS20, 2010.11635) | 各タスクを互いに直交する低ランク部分空間で学習(Stiefel) | **提案手法の直接的先行**。新規性を「VLM検出＋COCO保護＋ZCOCO保持」に絞る根拠＋ベースライン [[../../papers/forgetting-analysis/LowRankOrthogonalSubspaces_summary]] |
| ★**Textual Inversion for OVD w/o Forgetting**(2025, 2508.05323) | 本体凍結、トークン埋め込みのみ少数例で適応 | **別パラダイムの必須ベースライン**(重み差分でなくテキスト側適応) [[../../papers/OVD-CL/TextualInversion-OVD_summary]] |
| ★**Dynamic-DINO**(ICCV25, 2507.17436) | Grounding DINO 1.5 Edge を MoE 化(FFN分解＋ルーティング) | **同系バックボーンでのMoE実装の前例**。モジュール式案の技術参照 [[../../papers/OVD-CL/Dynamic-DINO_summary]] |
| **OVD Survey**(2023/24, 2307.09220) | OVDの6軸サーベイ | 背景地図。蒸留/擬似ラベル軸がZCOCO保持と直結 [[../../papers/OVD-CL/OVD-Survey_summary]] |

**精査の主な発見（当初推測からの訂正）**:
- 2305.19753 は CKA ではなく **Tunnel Effect**（層構造の知見）。2010.11635 は勾配干渉ではなく **低ランク直交部分空間CL**＝**我々の手法の最重要先行**。
- 2508.05323（Textual Inversion）・2507.17436（Dynamic-DINO）は本設定にほぼ直撃の **OVD適応/MoE** 研究で、ベースライン候補。

---

## A. 行うべき分析の設計（C2機序分析・実証）
広域調査から、本研究で**実施すべき分析**を「使う手法 → 測る量 → 主張できること」で具体化する。
いずれも**既存チェックポイント(exp_005/006)＋軽い順伝播/勾配計算**で実施可能（再学習ほぼ不要）。

| 分析 | 使う手法（出典） | 測る量 | 主張できること | 既存計画との対応 |
|---|---|---|---|---|
| **A1 タスクアラインメント⇄忘却** | NTK重なり行列(Doan) / GPM活性化部分空間の主角度(Saha) | 各ドメイン×過去ドメイン＋**COCO(task-0)** のアラインメント／部分空間重なり | 「忘却・ZCOCO低下 ∝ アラインメント」を定量化。直交化の動機を理論で裏付け | 議事録 分析C/D/E の理論化 |
| **A2 表現ドリフト** | linear probing(Probing Representation Forgetting) / Tunnel Effectの層構造 / CKA | 適応前後でのドメイン別・COCO特徴の表現類似度低下、層別の崩壊深度 | 忘却が「表現崩壊」か「ヘッドの上書き」か、どの深さで起きるかを切り分け | 議事録 分析A の精緻化 |
| **A3 忘却のモジュール局在化** | レイヤ別重み巻き戻し再評価＋Fisher情報(EWC) | どのモジュールを戻すと忘却が回復するか／Fisher重要度分布 | LoRA挿入箇所の決定。検出器の対照/クエリヘッドでも「分類は忘れ局在は頑健」か検証 | 議事録 分析A |
| **A4 損失地形・モード接続性** | linear mode connectivity(2010.04495)、sharpness/平坦性(C-Flat) | ドメイン間・COCO↔適応 の補間経路の損失バリア、解の鋭さ | 「重み平均/補間で保持できる」前提の成否、鋭さ↔忘却の相関 | 新規（領域5/6/8由来） |
| **A5 適応の低ランク性** | ΔW の SVD（Intrinsic Dim, 既調査） | 特異値スペクトル・実効ランク・上位r エネルギー比 | rank r をデータdrivenに決定 | 議事録 分析B |
| **A6 直交射影 mini 検証** | OGD/GPM 流の直交射影を後付け適用 | 射影後の忘却・ZCOCO の変化 | 「直交化で実際に減る」因果の決定打 | 議事録 分析F |

**推奨ストーリーライン**: A5(低ランクで十分) → A1(忘却=アラインメント/干渉) → A2/A3(どこで・どう壊れるか) → A4(平均/補間が効く地形か) → A6(直交化で実際に減る)。
A1 と A3 を中核に据えると、手法(直交化LoRA+COCO保護)と補完成分(蒸留/補間)の両方を動機づけられる。

## B. 手法・ベースラインの発掘
### B1. 比較ベースライン（公平比較の必須セット）
- **下限/上限**: 素朴逐次FT（exp_006）／joint（上限）。
- **既調査の直接競合**: ZiRa, DitHub, O-LoRA, （InfLoRAは自手法の基盤）。
- **CL各ファミリ代表（新規追加）**:
  - 正則化: **EWC/online-EWC**（+LwF）。
  - 勾配射影/直交部分空間: **GPM / OGD / 低ランク直交部分空間(Chaudhry NeurIPS20)**（自手法と同族＝**直接の先行**, 差別化必須）。
  - 勾配干渉緩和: **PCGrad/CAGrad**（領域8、ドメイン間勾配衝突の緩和）。
- **OVD適応の別パラダイム（新規・重要）**: **Textual Inversion(トークン埋め込みのみ適応, 2508.05323) / Dynamic-DINO(GDINOのMoE化, 2507.17436)**。
  → LoRA(重み差分)以外の適応軸の競合。特に Textual Inversion は本体凍結でZCOCOと整合的＝必須比較。
- **ZCOCO保持の機序系（新規・重要）**: **WiSE-FT補間 / model soups / task-vector加算 / ZSCL特徴蒸留 / MoE-Adapters(DDAS)ルーティング**。
  → ZiRa/DitHub 以外の「ゼロショット保持」軸の競合を網羅でき、自手法の優位条件を明確化できる。

### B2. C3手法に取り込める成分（直交化LoRA＋COCO-task0 への加算オプション）
- **特徴蒸留(ZSCL流)**: 凍結事前学習特徴に対する蒸留で ZCOCO を明示保護（直交化と補完的か検証）。
- **重み補間/平均(WiSE-FT/soups)**: 適応重みと凍結重みの補間で適応⇄保持を α 制御。推論コスト増なし＝DitHubへの単純さ優位。
- **task arithmetic**: ドメインを task vector として加算統合（パラメータ一定で蓄積）。符号衝突は TIES/DARE/TSV で緩和。
- **推論時ルーティング(DDAS流)**: OOD(=COCO的)入力を凍結経路へ。検出は**領域別ルーティング**化が必要＝拡張余地。
- **擬似ラベル・リプレイ(OWL-ST流)**: 未ラベル/Web画像でCOCO/ゼロショット概念を擬似復習（exemplar不要のZCOCO保持）。

### B3. 残る問い（open questions）
1. これらマージ/蒸留/ルーティングは grounded detector で成立するか（CLIP分類からの外挿の検証が貢献になる）。
2. NTK/GPM 流のアラインメント測定は大規模Transformer検出器＋非重複ラベルで実用的に計算でき、忘却を予測するか。
3. 直交化LoRA+COCO-task0 と上記補完成分は**加算的か/冗長か/競合か**。最良の stability–plasticity–ZCOCO パレート前線はどれか。
4. 非重複クラス集合(domain/class-IL ハイブリッド)での ZCOCO 保持・忘却の正しい評価プロトコルは何か。

---

## 注意点（全体 caveat）
- **外挿ギャップ（最重要）**: ZSCL/WiSE-FT/soups/task arithmetic/MoE-Adapters はほぼ **CLIP分類**での検証。検出への移植は非自明。
- **理論の前提**: NTK領域の忘却/OGD保証は無限幅・過剰パラメータの理想化で実機劣化あり。忘却はタスク類似度に**非単調**な報告(2401.12617)もある→アラインメントは「予測子」。
- **設定**: RF100は非重複クラス集合の domain/class-IL ハイブリッド。純 Domain-IL ではない。
- **マージの劣化**: task数増で重み干渉・符号衝突により精度低下（TIES/DARE/TSVが緩和）。
- 多くの強い finding は単一だが査読付き一次ソースに依拠。領域8の未読ソースは要精査。

## ソース（新規・一次中心）
ThreeScenarios(Nat.MI22, s42256-022-00568-3) / CL-Survey(TPAMI, 2302.00487) / NTK-overlap(2010.04003) / OGD(2006.11942) / EWC(1612.00796) /
ZSCL(2303.06628) / WiSE-FT(2109.01903) / ModelSoups(2203.05482) / TaskArithmetic(OpenReview 6t0Kwf8-jrj) / MoE-Adapters(2403.11549) /
RF100-VL(2505.20612) / OVD-Robustness(2405.14874) / OWL-ST(2306.09683) /
（領域8・精査済 2026-06-21）ProbingRepresentationForgetting(2203.13381) / TunnelEffect(2305.19753) / LinearModeConnectivity(2010.04495) /
C-Flat(2404.00986) / PCGrad(2001.06782) / LowRankOrthogonalSubspaces(2010.11635) / TextualInversion-OVD(2508.05323) /
OVD-Survey(2307.09220) / Dynamic-DINO(2507.17436)

## 関連
- 直球4領域: [[related_work_survey]]
- 研究計画: [[../research_plan]] / 議事録: [[research_plan_minutes]]
</content>
