# projects/lora_cl — LoRA / InfLoRA 付き MM-Grounding DINO

素の LoRA（exp_008 / exp_038 で確立）と、その上に実装した InfLoRA（CVPR 2024、
exp_042 design.md §2.3）。mmdet 本体は変更しない。

## 構成

| ファイル | 内容 |
|---|---|
| `lora_layers.py` | `LoRALinear`（ΔW=(α/r)·B·A、A kaiming / B 0 初期化） |
| `grounding_dino_lora.py` | `GroundingDINOLoRA`。挿入箇所の config 選択（include / exclude_components / パターン）、MHA の q/k/v/o 分解、base 凍結、dn 無効化 |
| `merge_lora.py` | タスク後マージ（W ← W + scaling·B·A、分解 MHA の再融合） |
| `optim.py` | `TrainableParamsConstructor`（requires_grad=True のみ optimizer 登録） |
| `inflora.py` | `InfLoRALinear` / `GroundingDINOInfLoRA`（下記） |
| `dual_gpm.py` | DualGPM メモリと A の設計（公式実装の移植） |
| `inflora_collect.py` | 入力共分散収集の共通処理（dropout 無効・決定的） |
| `inflora_prepare.py` | タスク学習前の A 設計 CLI |
| `inflora_update_memory.py` | タスク学習後の DualGPM メモリ更新 CLI |
| `check_lora_setup.py` ほか | 素の LoRA の検証（exp_038 で 10/10 OK） |
| `check_inflora_setup.py` | InfLoRA の検証（8 項目。2026-08-15 に 8/8 OK） |

## InfLoRA（公式実装 liangyanshuo/InfLoRA 準拠）

タスク t の 1 サイクル:

```bash
# 1. 学習前: θ_{t-1} で入力共分散を収集し、DualGPM メモリで射影して A を設計・固定
python projects/lora_cl/inflora_prepare.py <cfg> <θ_{t-1}.pth> --out design_t.pth \
    --memory mem_{t-1}.pth --max-samples 1000 --seed 0          # t=1 は --memory なし

# 2. 学習: config の model.inflora.design_path=design_t.pth を渡す（B のみ学習）

# 3. 学習後: 学習済み ckpt（分岐込み）で共分散を再収集し、メモリを更新
python projects/lora_cl/inflora_update_memory.py <cfg> <work_dir/iter_3000.pth> \
    --out mem_t.pth --memory mem_{t-1}.pth --task-index <t-1> --total 13 \
    --lamb 0.95 --lame 1.0 --max-samples 1000 --seed 0

# 4. マージ: θ_t = θ_{t-1} + B·A（alpha=r → scaling=1 が公式の ΔW=B·A に一致）
python projects/lora_cl/merge_lora.py <work_dir/iter_3000.pth> <θ_t.pth>
```

config は素の LoRA と同じ挿入選択に `alpha=r` と `inflora=dict(design_path=...)` を
加える（`inflora.py` の docstring に例）。

## 論文・公式実装との対応（2026-08-15 監査）

- 論文の記法では **B（r×d_I、設計・凍結）＝本実装の lora_A、A（d_O×r、ゼロ初期化・
  学習）＝本実装の lora_B**。forward `e = Wh + Σ A_iB_ih`（係数なし）は alpha=r
  （scaling=1）で一致し、タスク後の `W ← W + A_tB_t` は merge_lora.py が行う。
- 射影は公式コードどおり**片側**（`(I−MMᵀ)·C`、C は入力共分散）。論文の
  Ĥ=(I−MMᵀ)H から作る ĤĤᵀ=(I−MMᵀ)C(I−MMᵀ) とは重み付けが僅かに違うが、
  左特異ベクトルが M⊥ に入る性質は同じ（本実装は公式コードの形を採用）。
- DualGPM の双対表現（M または M⊥ を保持）は論文 §3.2.2 に明記されており、
  下記差分 1 はこの定義に合わせたもの。
- 共分散収集はサンプル選択・データ拡張の全 RNG を seed 固定（再現性）。

## 公式実装との意図的な差分（いずれもコード内に根拠を記載）

1. **DualGPM の最初のタスクで r ≥ d/2 のとき**（`dual_gpm.py`）: 公式は 'retain' と
   ラベルしつつ主部分空間を保存するが、retain の意味（直交補の保持）と直後の整合
   assert に矛盾する未実行の潜在バグ。双対表現の定義に従い直交補を保存する。
2. **設計行列のランク不足**（`dual_gpm.py` `design_A`）: 1 クラスタスクではプロンプトが
   全画像で同一のため、テキスト側（BERT）層の入力共分散の実効ランクが r=16 を下回る
   （実測: pistols で rank≈3）。SVD の零特異値側の列は任意方向でメモリと直交しないため、
   実効ランク（S > 1e-3·S[0]）を超える A の行は 0 にする（その成分の ΔW=0）。
   ViT 分類の公式実装では入力の多様性からこの領域に入らない。
3. **dropout の無効化**（`inflora_collect.py`）: 共分散収集は model.eval()＋検出器本体の
   training フラグのみ有効、で行う（loss がビット単位で決定的になることを実測確認）。
4. **optimizer**: 公式の SGD+cosine ではなく exp_042 共通枠の AdamW（design.md §6 に記録）。

## 検証（実測。2026-08-15、8/8 OK）

```bash
python projects/lora_cl/check_inflora_setup.py          # 純テンソルのみ
python projects/lora_cl/check_inflora_setup.py --gpu    # 全 8 項目（GPU 1 枚）
```

1. 232 層挿入・学習対象 lora_B のみ・scaling=1
2. design_A: 正規直交（1.2e-7）・remove と直交（2.3e-8）・retain 空間内（3.0e-8）
3. DualGPM: 基底正規直交・retain ≤ d/2・3 タスク一貫
4. 共分散収集で loss がビット一致（副作用なし）
5. 実データで prepare → メモリ更新 → 設計 A とメモリの直交性 max 8.6e-5
6. init_weights（DINO の xavier 一括初期化）後も設計 A 保存・A 凍結・B=0
6b. 実データ 1 step: loss 有限・A 不変・B のみ更新（226/232 層で非零）
7. 層単位で (W + B·A)x = 分岐 forward（4.8e-7）

## 追加検証（挙動レベル。2026-08-15 実測）

- **干渉抑制の実効性**（タスク間・実データ）: pistols（16 枚）でメモリを作り
  （task1 入力エネルギーの中央値 61.9% を被覆）、PascalVOC の共分散で A を設計。
  射影ありの設計は、射影なしの設計に比べて task1 入力への応答エネルギー
  tr(A·cov₁·Aᵀ) を**中央値 23.3%（最大 63.3%）まで削減**した（remove 232 層）。
  ※対照は「射影なしの top-r 設計」。乱数 A との比較は、乱数がほぼ零分散方向に
  エネルギーを撒くため対照として不適（実測で確認済み）。
- **実ランナー統合＋マージ**: `tools/train.py` 30 iter（設計 A 適用は init_weights 経由
  でも維持）→ `merge_lora.py`（232 層・MHA 6 個再融合・lora キー残存 0）→
  plain GroundingDINO でロード。マージ済み plain はその場マージとビット一致し、
  スコア > 0.3 の検出は分岐モデルと 4 画像すべてで数一致・bbox 最大 0.011 px・
  スコア最大 1.7e-3 の差。なお全 900 予測を順位のまま比較すると、同点近傍の
  低スコア検出の順位入れ替わり（マージによる演算順序の丸め差）で bbox 差が
  大きく見えるが、検出内容の差ではない。
