# exp_049 設計書: DGS の実装と ODinW-13（IVLOD）再現

作成 2026-08-21。**実行前のユーザー承認ゲート**（行動原理1・3）。
関連: [[implementation_plan]]（移植方針と公式コード調査の正本）、
[[../../papers/DGS_summary]]（論文要約）、
[[../exp_034/design]]・[[../exp_037/design]]（ZiRa / DitHub の ODinW-13 再現 = 同型の先行）、
[[../exp_042/design]]（EWC / InfLoRA の ODinW-13 実装検証）。

## 0. 位置付け

比較手法 DGS（Dynamic Group Subspace）を公式実装から `projects/dgs_cl/` に移植し、
公式準拠の学習体制で ODinW-13（IVLOD ベンチマーク、ZiRa プロトコル）を逐次学習・評価して
実装の正しさを確認する。RF100 6 ドメインへの展開は本実験の結果を見て別実験とする。

## 1. 目的

DGS を ODinW-13 の 13 タスクで逐次学習し、各タスクの mAP と ZCOCO を測る。
照合先は論文 Table 2（DGS: ZCOCO 46.4 / ODinW Avg 60.9。ただしベース検出器が
GroundingDINO-T であり本実験の MM-GDINO θ0 とは事前学習データが異なるため、
一致は「傾向と水準」で判断する。ZiRa 再現 exp_034 と同じ扱い）。
**仮説と判定基準はユーザーが確定する（note17 以降のノート）。**

## 2. 条件（2026-08-21 確定分）

### 2.1 学習体制（公式準拠。implementation_plan §3）

| 項目 | 値 | 出典 |
|---|---|---|
| スケジュール | 各タスク 12 epoch、MultiStepLR milestones=[11]、γ0.1 | 公式 IVLOD config |
| optimizer | AdamW、stage1 lr **8e-4** / stage2 lr **5e-4**、wd 1e-4 | 同上 |
| AMP | **不使用**（公式は --amp だが本環境で fp16 勾配 nan を実測。数値健全性を優先。2026-08-21 ユーザー決定） | §5.5 |
| バッチ | 公式は 2/GPU × 8 GPU = 16。本実験は **4/GPU × 4 GPU = 16** で合計を一致させる | 逸脱として記録（GPU 数の制約） |
| dn | **stage1 は有効・stage2 のみ無効**（公式 config の実測。移植は上流生成器を常時構築し stage2 では凍結・不使用） | 2026-08-21 訂正 |
| seed | 公式ドライバの 42 | 公式 |
| タスク順 | 公式順（AerialMaritimeDrone → … → VehiclesOpenImages の 13） | 2026-08-21 決定 |

### 2.2 DGS 固有（すべて公式 IVLOD config の値）

- IGA: r=16, alpha=32, enhancer 6 層の画像側・テキスト側 FFN、experts_num=1
- IGC: EMA マージ λ_A=λ_B=0.2（新 ← 0.2·base + 0.8·新 → base 昇格）、Group Init あり
- DTG: expand_th=150、SVD 正則化 min_eig_ratio=1e-3、特徴は backbone GAP
- ルーティング: 対称 KL、ood_th=500（超過で zero-shot フォールバック）
- stage2 蒸留: threshold_pseudo（sigma 0.4 / IoU 0.7）・inter-intra 特徴蒸留
  （img L2 w3.0 / text L2 w5.0）・query 蒸留（seperate_queryinit）
- 毎タスクの流れ: 特徴抽出 → DTG（グループ割当）→ stage1 か stage2 で学習 →
  最終 epoch 後にマージ → 次タスクへ load_from

### 2.3 評価

- 各タスク学習後に**学習済み全タスクを per-dataset 評価**（ZiRa プロトコル。
  推論はタスク ID を与えず画像単位ルーティング）＋ **ZCOCO**（eval_base_coco と同一の
  COCO2017 val・(800,1333)・batch 1。ルーティングは ood フォールバック込み）。
- 論文の報告は最終時点のみだが、途中時点も測る（既存再現実験と同じ）。

### 2.4 θ0・環境

- θ0: MM-GDINO Swin-T（本プロジェクト共通。公式の GDINO-T との違いは但し書き）
- 実行: 本環境 A100 40GB × 4（ODinW 再現系列は本環境実行の慣行）

## 3. 判定材料

**最終的な判定はユーザーが行う。**

1. 実装検証 6 項目（implementation_plan §4）がすべて OK であること。
2. 最終時点の ODinW-13 Avg / ZCOCO と論文 Table 2 の対比。
   参考水準: 論文 DGS は zero-shot 比 Avg +14.2 / ZCOCO −1.0。本 θ0 の zero-shot は
   Avg 0.4971 / ZCOCO 0.504（exp_034 実測）なので、実装が正しければ
   Avg が 0.60 前後・ZCOCO が 0.49〜0.50 に来ることが期待される。
3. DTG のグループ数（論文は τ=150 で 3 グループ。ドメイン近縁の ODinW-13 で
   グループがいくつできるかは θ0 依存であり、1〜4 の範囲なら異常なし）。

## 4. 実装（design 承認後に着手）

| 成果物 | 内容 |
|---|---|
| `projects/dgs_cl/` | implementation_plan §2.1 の移植一式＋README |
| `projects/dgs_cl/check_dgs_setup.py` | 実装検証 6 項目（implementation_plan §4） |
| `experiments/exp_049/configs/` | stage1 / stage2 / 特徴抽出 / ZCOCO 評価 config（公式 IVLOD 相当を本環境のデータパス・θ0 に合わせて生成） |
| `experiments/exp_049/run_dgs_odinw13.sh` | 逐次ドライバ（公式 IVLOD_distn.sh の写し: DTG → stage 選択 → 学習 → 評価） |

## 5. コスト見積り

各タスク 12 epoch。ODinW-13 のデータ量はタスク間で大きく異なり
（pistols 約 2.4k 枚 〜 PascalVOC 約 13k 枚）、合計約 35k 枚 × 12 epoch ÷ batch 16
≒ 26k iteration。LoRA のみ学習（backbone 凍結）で 1 iter は full FT より軽い。
AMP 込みで**学習合計 15〜25 h ＋ 評価（13 時点 × 学習済みタスク数 ＋ ZCOCO）約 8〜12 h**
を見込む（本環境 4 GPU）。ディスクは ckpt 13 個 ×（LoRA 込み state_dict ≈ 2 GB）≈ 26 GB。

## 5.5 実装で判明した事実（2026-08-21。実装検証 8/8 OK）

- **KD の解明（Appendix 取得により確定。2026-08-21）**: 論文の KD はクラスプロトタイプの
  対距離行列を base アダプタと一致させるトポロジー蒸留（L_kd = 3·img + 5·text。
  Appendix B.4–B.5、全実験で有効と明記）で、実装の 'inter-class' 分岐がその実体。
  **公開 config の `feat_distn.type='inter-intra'` は実装分岐と一致せず KD が不活性に
  なるリリース時の名称ズレとみられる**。logits 蒸留（loss_ld）が式(13) に無いのは正しい。
  → **'inter-class' に訂正して論文準拠とする（案A。2026-08-21 ユーザー決定）**。
  訂正後の stage2 1 step で inter_text_loss / inter_query_loss が乗ることを検証済み
  （check 項目 6。σ=0.4 でバッチ内に旧クラス物体が無ければ KD 項が出ないのは正しい挙動）。
- **DTG の統計はタスクあたり 260 枚以上で安定**（Appendix Fig.5。それ未満は Min KL が
  跳ね上がり過剰グループ化。40 枚サブセットで 13 グループになった本環境の実測は
  この既知挙動と整合）。ODinW-13 の Mushroom 41 枚・Packages 95 枚は scarce 領域。
- **ood_th=500 が既定・ZCOCO 評価は 200**（公開 ZCOCO.py）。Appendix Table 5 の感度と整合。
- **AMP で grad_norm=nan**（全ログ行。重みは有限で loss は低下するが fp16 勾配に
  nan が混入）。AMP 無しは健全（grad_norm 3.7〜4.3、iter +20%）。
  → **AMP 不使用で確定**（2026-08-21 ユーザー決定。公式準拠からの逸脱として記録）。
- tools/dist_train.sh の引数非引用転送のため、空白を含むデータパスは cfg-options で
  渡せない → タスク別 config 生成方式に変更（gen_configs.py）。
- 移植は公式 dgs パッケージの丸ごとコピー＋改変 3 点（implementation_plan §2 の
  選別コピー方針から変更。転記ミス低減と公式との diff 可能性のため）。

## 6. リスク・懸念

- **公式コードの動作前提**: stage2 の擬似ラベル経路が公式 dataset クラスに依存する場合、
  移植範囲が §2.1 より広がる（判明したら implementation_plan を更新して報告する）。
- **AMP**: 本プロジェクトの既存系列は AMP 不使用。公式準拠のため本実験は AMP を使うが、
  既存表と並べる際の但し書きになる。
- **θ0 差**: 論文値との照合は水準比較に留まる（exp_034/037 と同じ限界）。
- ルーティングの ood_th=500 は公式 GDINO-T の特徴スケールで調整された値であり、
  MM-GDINO の特徴では ood 判定率が変わり得る（検証項目 4 で COCO 画像の挙動を確認）。

## 7. 承認をお願いする範囲

§4 の実装（projects/dgs_cl への移植＋検証スクリプト＋config・ドライバ作成）と、
実装検証（§3 の 1）まで。**ODinW-13 の本走（13 タスク逐次）は実装検証の結果を
報告した後に改めて確認を取る。**
