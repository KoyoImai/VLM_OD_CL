# Swin Transformer 要約（アーキテクチャ・コード対応の観点）

出典: Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
（arXiv:2103.14030v2, 2021-08-17, Microsoft Research Asia）。`papers/SwinTransformer.pdf` に保存。
本プロジェクトのベースモデル MM-Grounding DINO の**画像バックボーン**（Swin-T）の原典。
以降のコード解析（`mmdet/models/backbones/swin.py`）はこの論文を土台にする。

## 概要（一言）
画像を非重複パッチに分割し、**局所窓内 self-attention** を計算する階層型 ViT。
窓を1ブロックごとに **shift** することで窓間の情報をつなぎ、計算量を画像サイズに対して
**線形**（通常の ViT は二乗）に抑える。CNN 同様の多解像度特徴を出すので検出・分割の汎用 backbone になる。

## 中核設計（§3）

### 3.1 全体構成（Fig 3）
- **Patch Partition + Linear Embedding**：4×4 パッチ → `4·4·3=48` 次元 → 線形で `C` 次元へ。
- **階層 4 stages**：解像度 `H/4 → H/8 → H/16 → H/32`、チャンネル `C → 2C → 4C → 8C`。
- **Patch Merging**：2×2 近傍を concat（4C）→ 線形で 2C、解像度を 1/2 に（各 stage 先頭で次解像度を作る）。

### 3.2 Shifted Window Self-Attention（本論文の肝）
- **W-MSA**：画像を `M×M`（既定 `M=7`）の非重複窓に分割し、**窓内だけで** self-attention。
- **SW-MSA**：次ブロックで窓を `⌊M/2⌋` だけずらす → 窓をまたいだ接続を作る（Fig 2）。
- 連続2ブロックで W-MSA→SW-MSA を交互に適用。式(3)が1ペアの計算：
  `ẑ=W-MSA(LN(z))+z; z=MLP(LN(ẑ))+ẑ; ẑ=SW-MSA(LN(z))+z; z=MLP(LN(ẑ))+ẑ`。
- **効率的バッチ計算（Fig 4）**：シフトで窓数が増える問題を、**cyclic shift（torch.roll）→ 1回の masked MSA
  → reverse cyclic shift** で回避。シフトで非隣接領域が同じ窓に混ざるため、**attn_mask** で
  サブ領域間の注意を遮断する（コードで最も非自明な部分）。
- **相対位置バイアス** 式(4)：`Attention = SoftMax(QKᵀ/√d + B)V`、`B∈ℝ^{M²×M²}`。
  各軸の相対位置は `[-M+1, M-1]` → 学習テーブルは `(2M-1)×(2M-1)`。絶対位置埋め込みより精度が高い（Table 4）。

### 計算量（式1,2）
- MSA: `Ω = 4hwC² + 2(hw)²C`（`hw` に二乗）。
- W-MSA: `Ω = 4hwC² + 2M²hwC`（`M` 固定なので `hw` に線形）。

### 3.3 変種（Table 1 基準）
- **Swin-T**: `C=96, layers={2,2,6,2}`（本プロジェクトの構成）
- 共通: `head_dim d=32`, `window M=7`, MLP 拡張率 `α=4`。
- 他: Swin-S `{2,2,18,2}`, Swin-B `C=128`, Swin-L `C=192`。

## 論文 ↔ コード対応（`mmdet/models/backbones/swin.py`）

| 論文（§/図/式） | 実装 |
|---|---|
| Patch Partition + Linear Embedding | `PatchEmbed`：`Conv2d(3→96, k=4, s=4)` が**分割と射影を1演算に融合**（:580）。論文は2段、コードは1 conv |
| 階層 4 stages | `self.stages`（:604-633）。本プロジェクトは `out_indices=(1,2,3)` で stride 8/16/32 のみ neck へ |
| Patch Merging | `PatchMerging`（各 stage 末尾, :608） |
| Swin block 式(3) | `SwinBlock.forward`（:359-372）pre-norm 残差2段＋`FFN`(GELU, ratio4) |
| W-MSA/SW-MSA 交互 | `shift=False if i%2==0 else True`（:442）, `shift_size=window//2=3`（:340） |
| Fig 4 cyclic shift + masked MSA | `ShiftWindowMSA.forward`（:181-）`torch.roll`＋attn_mask |
| 式(4) 相対位置バイアス | `WindowMSA.relative_position_bias_table`（`(2M-1)²=169, nH`）（:60-70, 90-） |
| 変種 Swin-T | config と完全一致（pretrain config:32-36） |

## コード解析時の注意点（論文を読んで判明）
- **PatchEmbed の融合**：論文の「分割→線形」2段は、コードでは `stride=kernel=4` の単一 Conv に等価実装。図と1対1ではない。
- **window_size 変更時**：相対位置テーブルは bicubic 補間が必要（コード `init_weights` :721-741）。継続学習で解像度/窓を変える設計をするなら影響箇所。
- **DropPath（stochastic depth, rate=0.2）**：学習時のみ有効。本プロジェクトの「凍結」は `lr_mult=0.0`（train モード維持）なので、
  凍結 backbone でも学習時 forward は確率的に揺らぐ点に注意。[[../setup_and_code_analysis/code_analysis/mmgrounding_dino_architecture]] §5 参照。

## 関連
- バックボーンを使う検出器全体: [[MM-GroundingDINO_summary]]
- アーキテクチャ精読メモ（コード即応）: `setup_and_code_analysis/code_analysis/mmgrounding_dino_architecture.md`
