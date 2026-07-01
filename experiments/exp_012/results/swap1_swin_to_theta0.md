# exp_012 第1 swap 結果：unfrozen の Swin を θ0 に戻す（ZCOCO 帰属）

作成日: 2026-06-30。設計: [[../design.md]]。生成: `make_hybrid_swin_theta0.py`（学習なし・重み合成のみ）。
評価: `eval_base_coco.py`（(800,1333) keep_ratio、従来と同一）、4GPU。

## ハイブリッド H_d の定義
**H_d = θ1_d^U（unfrozen）の `backbone.*`（Swin 187 params）だけを θ0 に置換**。
BERT(`language_model`)・neck・encoder・decoder・bbox_head は **unfrozen のまま**。
- swap 正確性は検証済み: backbone 187 のうち 175 が θ0 と異なる（=戻している）、残り12 は
  `relative_position_index`（定数バッファ・学習対象外）。Swin 外への漏れ・テキスト系の混入なし。

## 結果（ZCOCO = COCO 保持 mAP）

| ドメイン | θ0 | frozen θ1_d^F | unfrozen θ1_d^U | **H_d (Swin→θ0)** | H − unfrozen | 総忘却(θ0−unfrozen) | backbone寄与率 |
|---|---|---|---|---|---|---|---|
| underwater | 0.504 | 0.389 | 0.407 | **0.432** | +0.025 | 0.097 | ~26% |
| aerial | 0.504 | 0.318 | 0.369 | **0.404** | +0.035 | 0.135 | ~26% |
| microscopic | 0.504 | 0.327 | 0.288 | **0.323** | +0.035 | 0.216 | ~16% |
| videogames | 0.504 | 0.324 | 0.315 | **0.345** | +0.030 | 0.189 | ~16% |
| documents | 0.504 | 0.275 | 0.211 | **0.258** | +0.047 | 0.293 | ~16% |
| electromagnetic | 0.504 | 0.331 | 0.269 | **0.321** | +0.052 | 0.235 | ~22% |

## 判定（合意した符号ルールに基づく）

1. **低下4ドメイン（micro/vg/doc/em）すべてで H_d が回復**（+0.030〜+0.052）。
   共適応の不整合は性能を下げる方向にしか働かないため、回復は**曖昧さなく**次を意味する：
   → **backbone(Swin) の適応が、これら4ドメインの ZCOCO 低下の原因の一部**である（第1 swap の目的＝帰属を達成）。

2. **回復は partial**。H_d は 0.26〜0.43 で θ0=0.504 に遠い。backbone を戻しても残る低下は、
   unfrozen のまま残した**共有検出経路**由来。backbone の寄与率は **約16〜26%**（不整合により過小評価＝**下限**）。
   → **忘却の主因は依然 共有検出経路（exp_004 と整合）、backbone はそこに 2〜3割を上乗せ**。

3. **回復は改善側 underwater/aerial でも発生**（H > unfrozen）。不整合は下げ方向なのに上がった以上、
   → **適応後の Swin は、どのドメインでも θ0 の Swin より COCO に悪い**（backbone 適応は常に COCO 特徴を傷つける）。

## 解釈上の含意（仮説・要追加 swap）
- 上記3は「ドメイン距離」仮説に修正を迫る：**「近ドメインの backbone が COCO に優しい」のではなく、
  backbone はどこでも COCO を害している**。unfrozen が frozen を上回った近ドメインの改善は、
  backbone ではなく**検出経路側の違い**による可能性。
- ただしこの「検出経路側」主張は H_d vs frozen の比較に不整合が混じるため**未確定**。
  次 swap（検出経路 or BERT を θ0 に戻す等）で切り分ける。

## 追補: Swin 学習のドメイン適応(C)への寄与（同 hybrid を自ドメイン valid で評価）

### 適応 mAP（自ドメイン valid）

> **訂正(2026-06-30)**: 下表「unfrozen−frozen」列は当初「Swin 寄与」と記したが**誤り**。frozen は
> **Swin と BERT の両方**を凍結(lr_mult=0.0)しているため、これは **Swin+BERT 合算の寄与**。Swin 単独へは
> 既存モデルからは分離不可（要「Swin だけ unfrozen」run）。詳細: [[swap2_bert_to_theta0]]。

| ドメイン | frozen 適応 | unfrozen 適応 | H_d 適応 (Swin→θ0) | unfrozen−frozen<br>(**Swin+BERT 合算**寄与) | frozen−H_d<br>(純・不整合=共適応強度) |
|---|---|---|---|---|---|
| underwater | 0.337 | 0.359 | 0.287 | +0.022 | 0.050 |
| aerial | 0.468 | 0.486 | 0.438 | +0.018 | 0.030 |
| microscopic | 0.499 | 0.538 | 0.420 | +0.039 | 0.079 |
| videogames | 0.719 | 0.782 | 0.667 | +0.063 | 0.052 |
| documents | 0.478 | 0.542 | 0.312 | +0.064 | 0.166 |
| electromagnetic | 0.454 | 0.502 | 0.411 | +0.048 | 0.043 |

### 方法論的注意（適応では H_d の符号がクリーンにならない）
- ZCOCO では「回復(上昇)」が不整合と逆符号でクリーンだったが、**適応では「Swin適応寄与の消失」も「共適応の不整合」も
  どちらも適応を下げる同符号**。実際 **H_d 適応は全ドメインで frozen すら下回る**（frozen−H_d>0）。
- → **H_d−unfrozen の落差は Swin 適応寄与の "上限"（不整合で水増し）**。Swin 適応重要度の**クリーンな指標は
  「frozen vs unfrozen」**であり、H_d 単独では答えにならない。

### 所見（クリーンな frozen vs unfrozen で読む）
1. **適応の大半は検出経路だけ（frozen）で達成済み**（frozen で 0.337〜0.719）。**Swin 学習の上乗せは +0.018〜+0.064 と控えめ**。
   → 適応の主役も検出経路。Swin 学習は top-up。
2. **上乗せは遠ドメインほど大**（documents +0.064 / videogames +0.063 ≫ aerial +0.018 / underwater +0.022）。
3. **共適応強度（frozen−H_d）は documents で突出（0.166）**＝最も遠いドメインで Swin と検出経路が最も強く依存し合う。

### 適応(C) と 忘却(A) の対称的まとめ

| 能力軸 | 主役モジュール | Swin 学習の寄与 |
|---|---|---|
| 適応 (C) | 検出経路（frozen で大半） | **+0.018〜+0.064**（遠ドメインで大） |
| 忘却 (A=ZCOCO) | 共有検出経路（exp_004） | **−0.025〜−0.052** を上乗せ（下限） |

→ **適応も忘却も主役は検出経路。Swin 学習は両方に副次的に効く＝「遠ドメインで適応を少し稼ぐ代わりに COCO を少し失う」トレードオフ部品。**

## 成果物
- ハイブリッド: `experiments/exp_012/hybrids/{domain}_swin_theta0.pth`（6本）
- 評価: `experiments/exp_012/{domain}_swin_theta0_coco/`（ZCOCO）, `experiments/exp_012/{domain}_swin_theta0_adapt/`（適応）

## 関連
- 現象の出所: [[../../exp_011/results/frozen_vs_unfrozen_6domains]]
- 忘却源=共有検出経路: [[../../exp_004/design]]
- アーキ・接頭辞: [[../../../setup_and_code_analysis/code_analysis/mmgrounding_dino_architecture]]
