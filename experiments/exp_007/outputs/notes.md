# exp_007 総括メモ：忘却の機序分析（手法設計への含意）

## 実施した分析と結果
| 分析 | 結果 |
|---|---|
| B 低ランク性（ΔW の SVD） | 適応は**中程度の低ランク**（実効ランク比 0.38〜0.63、上位16特異値で 56〜89%）。text_feat_map/memory_trans_fc は集中、decoder は分散。→ LoRA は**中rank（≥16目安）**が要る。 |
| C/D 部分空間干渉と忘却の相関 | **相関なし**。活性化部分空間の重なり（trace(ΔW C_X ΔWᵀ) や向き割合 I_dir≈0.34 一定）では COCO 忘却の差を説明できなかった。weight空間の重なり量と mAP 低下は非線形。 |
| **E 直交除去の因果検証**（documents/underwater） | **忘却=COCO部分空間成分（除けば COCO 回復: documents 0.275→0.48, underwater 0.414→0.49）。だが適応も同じ部分空間に依存（除くと自ドメイン崩壊: documents 0.478→0.089, underwater 0.316→0.168）。両極で普遍。** |

## 確定した機序（本研究の知見）
1. **忘却の正体**: 新ドメイン学習で**共有検出重みが COCO の重要部分空間方向に動く**こと（因果的に確認＝除けば回復）。
   ＝忘却源は backbone でなく**共有検出経路**（exp_004 とも整合）。
2. **適応と忘却は同一部分空間を奪い合う**: ドメイン適応に必要な方向が COCO 部分空間内にある。
   COCO から遠いドメインほど依存が強く崩壊が激しい（documents 保持18% < underwater 44%）。
3. ゆえに **単一の共有重みでは COCO 保持とドメイン適応を両立できない**（トレードオフが直結）。

## 手法(C3)への含意 ― 設計方針の更新
- **素朴な InfLoRA 型（共有重み更新を COCO 部分空間と直交させる）は本設定では不適**。
  COCO を守る＝適応を失う。検出は COCO とドメインが共有特徴空間を強く共有するため。
- **正しい方向 = base 凍結 ＋ ドメイン分離 LoRA**:
  - **COCO/ゼロショット推論は base 経路のみ → 忘却ゼロ（W_base 不変）**。
  - **ドメイン推論は base + そのドメインの LoRA**（別パラメータなので奪い合いが起きない）。
  - ＝ モジュール式/条件付き LoRA（DitHub 的）。推論時のドメイン識別（プロンプト/ゲーティング）が論点。
  - 過去ドメイン間の干渉抑制には InfLoRA/直交化が依然有効な可能性（COCO 保持とは別問題）。
- 代替/併用: ZiRa の ZiL（soft トレードオフ）。

## 妥当性・限界（正直な明示）
- C/D は相関が出ず（活性化部分空間重なりは忘却の良い予測子でない）。一方 E（因果介入）で機序は実証。
- E は層ごと直交射影＋複数層同時適用の評価。COCO が 0.504 に僅か届かないのは非対象 param(norm/bias)が ft のため。
- 検証は documents/underwater の2ドメイン（両極）。中間ドメインでの追試は将来課題。

## 次フェーズ（C3 手法）への申し送り
- 手法は **「base 凍結＋ドメイン分離 LoRA（モジュール式/条件付き）」** を主軸に設計。
- LoRA rank は中程度（分析B）。挿入箇所は共有検出経路（encoder/decoder/head; 分析C の層別 trace で重点層を選定可）。
- 比較: 素朴逐次FT(下限)/joint(上限)/ZiRa/DitHub/O-LoRA。指標 Avg/Forgetting/ZCOCO/hAP。

## 関連
- 結果: `results/analysisE_orthogonal_projection.md`, `results/dw_lowrank_*.json`, `results/interference_vs_forgetting_coco.json`
- 設計/妥当性: [[../design]], [[../validity_review]]
- 競合: [[../../papers/OVD-CL/DitHub_summary]], [[../../papers/ZiRaGroundingDINO_summary]], [[../../papers/LoRA-CL/InfLoRA_summary]]
