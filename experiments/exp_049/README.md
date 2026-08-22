# exp_049 実行手順（DGS / ODinW-13 IVLOD・本環境）

design: [[design]]、移植の正本: [[implementation_plan]]。
**実装検証 8/8 OK（2026-08-21）。本走は design.md §7 のとおりユーザー確認後に実行する。**

## 0. 状態

| | 状態 |
|---|---|
| 移植（projects/dgs_cl。公式 projects/DGS/dgs の丸ごと移植＋改変 3 点） | 完了 |
| 学習 config 25 本・評価 config 14 本・ドライバ・検証スクリプト | 生成済み |
| 実装検証（check_exp049_setup.py --gpu） | **8/8 OK** |
| DTG 用特徴の全量抽出（experiments/exp_049/feats） | 実行済み/実行中（§2） |
| 本走 | 未実施（承認待ち） |

## 1. 移植の要点（公式からの改変。詳細は implementation_plan §2）

1. import パス（projects.DGS.dgs → projects.dgs_cl）。
2. dn: 本体無変更のため、`gdino_inc.py` で上流 DINO に常に有効な dn_cfg を渡して
   生成器を構築し（θ0 のキー互換）、`dn_cfg=None`（stage2）のときは凍結・不使用。
   head 側は `split_outputs` の dn_meta=None ガードを移植 head に上書き。
   **stage1 は公式どおり dn 有効**（dn label_embedding も公式どおり学習対象）。
3. タスク別 config の生成（gen_configs.py）: `tools/dist_train.sh` が引数を非引用
   転送するため、空白を含むデータパス（Aquarium 等）は --cfg-options で渡せない。
   タスク固有値は config に焼き込み、ドライバは load_from だけを渡す。

### 忠実性ノート（公式コード＋公式 config の実測挙動）

- **KD（Appendix で確定）**: 論文の KD はトポロジー蒸留（クラスプロトタイプ対距離行列の
  一致、γ₁=3/γ₂=5。`papers/DGS_summary.md` §7.1）＝実装の 'inter-class' 分岐。
  **公開 config の 'inter-intra' は実装分岐に無く KD が不活性になる名称ズレ。
  'inter-class' に訂正して論文準拠とした（案A。2026-08-21 ユーザー決定、design §5.5）。**
  補足資料は `papers/DGS_supplemental.pdf`、要約は `papers/DGS_summary.md` §7。loss_ld（logits 蒸留）が無いのは論文どおり。stage2 のその他の実体は
  擬似ハードラベル（σ=0.4/IoU0.7）での Group Alignment ＋ base expert クエリ初期化。
- マージは 新 ← 0.2·base ＋ 0.8·新 を base に昇格（check 項目 3 で数値一致を確認）。

## 2. 事前準備: DTG 用特徴の全量抽出（θ0 で 1 回だけ）

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/exp_049/extract_feats_odinw13.py
# -> experiments/exp_049/feats/<公式タスク名>/train/img_<id>.pt（計 約35k 件・数百 MB）
# 中断しても再実行で続きから（タスク単位でスキップ）
```

## 3. 実装検証（再現手順）

```bash
python experiments/exp_049/check_exp049_setup.py                 # 静的 2 項目
CUDA_VISIBLE_DEVICES=0 python experiments/exp_049/check_exp049_setup.py --gpu   # 全 8 項目
```

## 4. 本走（ユーザー確認後）

```bash
bash experiments/exp_049/run_dgs_odinw13.sh     # GPUS=4 が既定・本環境
```

各タスクで DTG → stage1/stage2 自動選択 → 12 epoch 学習 → 学習済みタスク
per-dataset 評価＋ZCOCO。再開はドライバ再実行（学習は epoch_12.pth、評価は
出力ディレクトリの有無でスキップ。**json の無い評価ディレクトリは消してから**）。

### 判断事項の状態

1. **AMP: 不使用で確定**（2026-08-21 ユーザー決定。公式は --amp だが本環境で
   fp16 勾配 nan を実測、数値健全性を優先。ドライバは AMP 無し）。
2. **DTG 閾値のスケール**: expand_th=150 は公式 GDINO-T の特徴スケールでの値。
   MM-GDINO の全量統計での KL がどの水準かは §2 完了後に実測して報告
   （サブセット統計では全タスクが別グループになった）。

## 5. 監視

```bash
tail -f experiments/exp_049/work_dirs/<task>/*/​*.log
cat experiments/exp_049/work_dirs/task_id_mapping.yaml   # グループ割当
# 評価ログにはルーティング精度が出る（Task prediction acc）
```
