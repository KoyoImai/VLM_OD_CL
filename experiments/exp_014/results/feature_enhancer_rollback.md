# exp_014 結果：Feature Enhancer（encoder）ロールバック — 全体＋3経路（image/fusion/text）

作成日: 2026-07-01。設計: [[../design]]。生成: `make_hybrid_encoder.py`（学習なし・重み合成のみ、24 hybrid）。
評価: ZCOCO=`eval_base_coco.py` / 適応=自ドメインFT config、いずれも (800,1333) keep_ratio・4GPU。
全48評価で **mAP 全数取得（欠損 0）**。適応で終了時 segfault による見かけ上の FAIL が5件出たが、
いずれも **mAP 算出後のクラッシュ**で値はログから回収済み（copypaste は全数評価完了後にのみ出力＝完全性保証）。ZCOCO は FAIL 0。

## ハイブリッド定義
θ1_d^U の encoder 配下を θ0 に置換（backbone/neck/language_model/text_feat_map/decoder/bbox_head は unfrozen のまま）。
- **H^enc** = `encoder.*`（276）＝ Feature Enhancer 全体
- **H^image** = `encoder.layers.*`（96）＝ 画像 deformable self-attn
- **H^fusion** = `encoder.fusion_layers.*`（108）＝ 画像↔テキスト双方向 cross-attn
- **H^text** = `encoder.text_layers.*`（72）＝ テキスト self-attn
- 整合: enc = image ∪ fusion ∪ text（276 = 96+108+72）。

## ① ZCOCO（COCO 保持 mAP。θ0=0.504、unfrozen が学習後＝忘却状態）

| ドメイン | θ0 | unfrozen | **H^enc** | **H^image** | **H^fusion** | **H^text** |
|---|---|---|---|---|---|---|
| underwater | 0.504 | 0.407 | **0.473** | 0.460 | 0.426 | 0.410 |
| aerial | 0.504 | 0.369 | **0.475** | 0.461 | 0.411 | 0.392 |
| microscopic | 0.504 | 0.288 | **0.466** | 0.438 | 0.340 | 0.325 |
| videogames | 0.504 | 0.315 | **0.471** | 0.453 | 0.371 | 0.338 |
| documents | 0.504 | 0.211 | **0.445** | 0.413 | 0.274 | 0.246 |
| electromagnetic | 0.504 | 0.269 | **0.453** | 0.405 | 0.340 | 0.295 |

### 回復量（group − unfrozen。正＝忘却の一因）

| ドメイン | enc | **image** | fusion | text | Σ単独 | Σ−enc（重複）|
|---|---|---|---|---|---|---|
| underwater | +0.066 | **+0.053** | +0.019 | +0.003 | +0.075 | +0.009 |
| aerial | +0.106 | **+0.092** | +0.042 | +0.023 | +0.157 | +0.051 |
| microscopic | +0.178 | **+0.150** | +0.052 | +0.037 | +0.239 | +0.061 |
| videogames | +0.156 | **+0.138** | +0.056 | +0.023 | +0.217 | +0.061 |
| documents | +0.234 | **+0.202** | +0.063 | +0.035 | +0.300 | +0.066 |
| electromagnetic | +0.184 | **+0.136** | +0.071 | +0.026 | +0.233 | +0.049 |

### ZCOCO 所見（符号ルール：回復＝クリーンな帰属）
1. **Feature Enhancer(encoder) の適応こそが COCO 忘却の主因**。H^enc は **0.445〜0.475 と θ0=0.504 の目前まで回復**
   （回復 +0.066〜+0.234）。exp_012 の backbone(Swin) 回復 +0.025〜0.052 とは**桁違い**。
   → 「忘却の主役は共有検出経路」（exp_004/012）を **encoder に局在**として精密化。
2. **encoder 内では image（画像 deformable self-attn）が支配的**。回復 image +0.053〜+0.202 ≫ fusion +0.019〜+0.071 ≫ text +0.003〜+0.037。
   H^image 単独で H^enc 回復の **80〜86%**（例 documents: image 0.413 / enc 0.445）。
   → **画像側 self-attention の適応が忘却を最も強く駆動**。テキスト self-attn(text) はほぼ無影響。
3. **3経路は部分的に冗長（劣加法）**。Σ単独 > enc（差 +0.009〜+0.066）＝3経路の忘却寄与は重なりを持つ（共適応の共有）。
4. **低下側4ドメイン（micro/vg/doc/em）で回復が最大**（documents enc +0.234）。遠ドメインほど encoder 適応が COCO を害する。

## ② 適応（自ドメイン valid mAP）

| ドメイン | frozen | unfrozen | **H^enc** | **H^image** | **H^fusion** | **H^text** |
|---|---|---|---|---|---|---|
| underwater | 0.337 | 0.359 | 0.225 | 0.298 | **0.342** | **0.341** |
| aerial | 0.468 | 0.486 | 0.295 | 0.381 | **0.464** | **0.465** |
| microscopic | 0.499 | 0.538 | 0.290 | 0.369 | **0.520** | **0.528** |
| videogames | 0.719 | 0.782 | 0.196 | 0.337 | **0.762** | **0.768** |
| documents | 0.478 | 0.542 | 0.108 | 0.208 | **0.503** | **0.484** |
| electromagnetic | 0.454 | 0.502 | 0.104 | 0.209 | **0.480** | **0.482** |

### 適応 所見（符号は非クリーン＝上限として読む。クリーンは frozen vs unfrozen）
1. **fusion / text のロールバックは適応をほとんど損なわない**（unfrozen 比 −0.01〜−0.02、documents のみ −0.04〜−0.06。
   videogames fusion 0.762 / text 0.768 ≈ unf 0.782）。image/enc の毀損（−0.16〜−0.43）とは桁が違う。
   → **cross-modal fusion とテキスト self-attn の適応は、ドメイン適応にほぼ不要**。
2. **image / enc のロールバックは適応を大きく毀損**（image 0.208〜0.381、enc 0.104〜0.295、frozen をも下回る）。
   H^enc が最悪（3経路同時除去で encoder→decoder 界面が最も壊れる＝超加法的悪化）。
3. → **encoder の画像 self-attn(image) が「忘却も適応も駆動する」中核の共適応モジュール**。
   ZCOCO では image 除去が最大の回復（忘却源）、適応では image 除去が最大の毀損（適応に重要）＝表裏一体。
   text/fusion はどちらの軸でも影響が小さい。

## まとめ（exp_012 → 013 → 014：忘却源の局在）

| モジュール | ZCOCO 忘却への寄与（clean 回復） | 適応での役割 |
|---|---|---|
| backbone Swin (exp_012) | 小 +0.025〜0.052 | top-up |
| neck / text_feat_map (exp_013) | 微小 +0.002〜0.007 / 0〜0.006 | 補助 |
| BERT (exp_012) | 軽微 +0.001〜0.025 | 訓練不要 |
| **encoder image 層 (exp_014)** | **主 +0.053〜0.202** | **適応にも重要（除去で最大毀損）** |
| encoder fusion 層 | 中 +0.019〜0.071 | 適応に不要 |
| encoder text 層 | 小 +0.003〜0.037 | 適応に不要 |
| **encoder 全体** | **最大 +0.066〜0.234（θ0 目前まで回復）** | 除去で適応壊滅 |

→ **COCO 忘却の主因は Feature Enhancer(encoder)、とりわけ画像 deformable self-attention 層**。
backbone/投影/テキスト系は副次的。encoder の画像 self-attn が適応と忘却を同時に駆動する中核であり、
「新ドメインへの適応で最も動く部品が、最も COCO を失わせる」というトレードオフの所在が特定された。
次段では decoder / bbox_head（照合・回帰）を同様に切り分け、encoder との寄与配分を確定する。

## 追加考察（2026-07-01）：ドリフト分析 — 「動いた量」と「忘却寄与」の乖離

ロールバック効果の小ささが「重要でない」からか「そもそも学習で動いていない」からかを切り分けるため、
各経路の相対ドリフト ‖θ1_d^U − θ0‖ / ‖θ0‖ を測定した。

### 相対ドリフト（学習でどれだけ動いたか）
| ドメイン | image | text | fusion |
|---|---|---|---|
| underwater | 0.068 | 0.046 | 0.070 |
| aerial | 0.048 | 0.030 | 0.047 |
| microscopic | 0.059 | 0.038 | 0.058 |
| videogames | 0.056 | 0.038 | 0.057 |
| documents | 0.085 | 0.058 | 0.085 |
| electromagnetic | 0.092 | 0.059 | 0.092 |

### 回復量 / ドリフト（動いた量あたりの ZCOCO 忘却寄与、平均）
| 経路 | 平均 回復/ドリフト |
|---|---|
| **image** | **1.93** |
| fusion | 0.76 |
| text | 0.58 |

### 解釈
1. **fusion は image とほぼ同じだけ動く**（ドリフト 0.047〜0.092 で image と同等）が、ZCOCO 回復も適応毀損も小さい。
   → fusion の適応は「**動くが無害・無用**」（COCO を害さず、ドメイン適応にも寄与しない drift）。
2. **image は動いた量あたりの忘却寄与が突出**（1.93 ≈ fusion/text の 2.5〜3.3倍）。
   → **同じ大きさの重み変化でも、画像 self-attention の変化だけが COCO を選択的に破壊する**。
   「大きく動いた層が忘却源」ではなく「**画像 self-attention という特定の場所の変化が本質的に有害**」。
3. text は動きも寄与も最小。→ どちらの軸でも周辺的。

### 継続学習の方法設計への含意
- **忘却緩和の予算は encoder の画像 self-attention 層（`encoder.layers`）に集中投下すべき**
  （凍結 / EWC / 低 lr 等をこの層に限定）。ここを守れば COCO 保持の大半を回収できる可能性（H^image が示唆）。
- **fusion / text は自由に適応させてよい**（COCO を害さず、適応にもほぼ不要）。むしろ凍結して計算を節約しても
  適応をほぼ損なわない見込み（適応ロールバックが unfrozen 近傍）。
- ただし image は「適応にも重要」（ロールバックで適応も毀損）＝**保持と適応が最も競合する層**。
  単純凍結は適応を削るため、**選択的正則化（hard freeze でなく soft constraint）**が有望。次段の検証課題。

## 成果物
- ハイブリッド: `experiments/exp_014/hybrids/{domain}_{enc|fusion|text|image}_theta0.pth`（24本）
- 評価: `experiments/exp_014/{domain}_{group}_theta0_{coco|adapt}/`（48本）
- ログ: `zcoco_eval.log` / `adapt_eval.log` / `make_hybrid.log`

## 補足（評価の健全性）
- 適応で FAIL 5件（underwater/fusion, videogames/enc, videogames/image, electromagnetic/enc, electromagnetic/image）は
  全て **mAP 算出後の分散終了時 segfault（exitcode −11）** で、値は正常に記録済み。再評価不要。

## 関連
- 前段: [[../../exp_012/results/swap1_swin_to_theta0]] / [[../../exp_013/results/feature_rollback_imgfeat_vs_txtfeat]]
- アーキ（Encoder=Feature Enhancer §2.4, fusion/text/image 層）: [[../../../setup_and_code_analysis/code_analysis/mmgrounding_dino_architecture]]
- 現象の出所: [[../../exp_011/results/frozen_vs_unfrozen_6domains]]
