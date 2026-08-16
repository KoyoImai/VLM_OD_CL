# projects/ewc_cl — EWC 付き MM-Grounding DINO

exp_042 design.md §2.2 の実装。原論文（Kirkpatrick et al., PNAS 2017）と
スター数上位の公開実装（Avalanche / GMvandeVen / Mammoth / GEM、2026-08-15 調査）の
慣行に従う。設計の経緯と根拠は `experiments/exp_042/design.md`、適用例の調査は
`papers/CL-detection/EWC-in-detection_survey.md`。

## 構成

| ファイル | 内容 |
|---|---|
| `ewc_grounding_dino.py` | `EWCGroundingDINO`。学習対象の選択と凍結、loss への `loss_ewc` 加算、dn 無効化（lora_cl と同じ NoDNQueryGenerator 差し替え） |
| `ewc_state.py` | 2 バッファ状態（A=ΣF_k、B=ΣF_k·θ*_k、const）の入出力・更新・ペナルティ計算 |
| `estimate_fisher.py` | タスク終了時の対角・経験 Fisher 推定と状態更新の CLI |
| `check_ewc_setup.py` | 実装検証（5 項目。2026-08-15 に 5/5 OK） |

## 仕組み

- **学習対象 = EWC 対象**。`ewc.target_components` にトップレベルモジュール名のリスト
  （例 `['backbone', 'encoder']`）か `'all'` を指定する。対象外は `requires_grad=False`。
- **ペナルティは 2 バッファで原論文と等価**。タスク別二次項の和
  Σ_k (λ/2)·F_k·(θ−θ*_k)² は A・B・const に厳密に畳める（論文 §2 が明記する等価変形）。
  勾配は λ·(A·θ−B)。メモリはタスク数によらず一定（全モジュール対象で約 1.4 GB）。
- **Fisher は対角・経験 Fisher**。学習と同じ検出損失の勾配²を batch=1 で平均。
  サンプルは上限枚数（既定 1000）まで、超える場合は seed 固定の無作為抽出。
  dropout は無効化して推定する（`model.eval()` で関数型 dropout も含め全て切り、
  loss 経路の構造分岐に使う検出器本体の training フラグだけを立て直す。この状態で
  loss はビット単位で決定的。Dropout/DropPath モジュールの列挙では DecomposedMHA や
  融合層の `F.dropout(training=self.training)` を取りこぼすことを実測して修正済み）。
- 状態は checkpoint に含めず、ファイルで持ち回る（ckpt の肥大と下流の
  unexpected keys を避ける）。

## 論文・公開実装との対応（2026-08-15 監査）

- **ペナルティの係数**: 原論文 式(3) は `Σ (λ/2)·F_i·(θ_i−θ*_i)²`。本実装と
  GMvandeVen（`ewc_loss` の `(1./2)*sum`）はこの ½ 付きの定義。
  **Avalanche と Mammoth は ½ を付けない**（`λ·Σ F(θ−θ*)²`）ため、同じ効果を
  得る λ は本実装の 2 倍になる。λ を跨いで比較するときはこの流儀の差に注意。
- **Fisher の正規化**: 本実装はサンプル数で平均（batch=1）。GMvandeVen も
  サンプル数割り（`est_fisher_info /= index`）。Avalanche はミニバッチ数割り
  （batch=1 なら同一）。
- **タスク間の蓄積**: 本実装の 2 バッファは、GMvandeVen の `offline` モード
  （タスク別の F と θ* を全て保持）と数学的に同一のペナルティを O(1) メモリで
  計算するもの（論文 §2 の等価変形）。Avalanche の既定 `separate` とも同じ。
  減衰 γ 付きの online EWC とは別物。
- **再現性**: Fisher 推定はサンプル選択・データ拡張の全 RNG を seed 固定
  （CUDA 原子加算による ~1e-7 相対の揺らぎは残る）。
- **マルチ GPU の注意**: dn 無効時は損失に関与しないパラメータ（dn 用
  label_embedding 等）があり、DDP では `find_unused_parameters=True` が必要。
  exp_042 は 1 GPU なので不要。

## 使い方

config（タスク t の学習):

```python
custom_imports = dict(imports=[..., 'projects.ewc_cl'], allow_failed_imports=False)
model = dict(
    type='EWCGroundingDINO',
    use_dn=False,                     # ODinW-13 共通枠
    bbox_head=dict(type='NoDNGroundingDINOHead'),
    ewc=dict(
        target_components='all',      # 学習対象 = EWC 対象
        lam=1000.0,                   # λ（パイロットで決める）
        state_path=None))             # t>=2 はドライバが --cfg-options で上書き
```

タスク終了時（学習 → Fisher 推定 → 状態更新）:

```bash
python projects/ewc_cl/estimate_fisher.py <task_config.py> <work_dir/iter_3000.pth> \
    --prev-state <state_t-1.pth> --out <state_t.pth> \
    --max-samples 1000 --seed 0 --task-name <task>
```

次タスクの学習はドライバが `--cfg-options load_from=<θ_t> model.ewc.state_path=<state_t.pth>`
を渡す。t=1 は `state_path=None`（ペナルティ無し＝素のフル FT と一致）。

## 検証（実測。2026-08-15）

```bash
python projects/ewc_cl/check_ewc_setup.py          # 純テンソルのみ
python projects/ewc_cl/check_ewc_setup.py --gpu    # 全 5 項目（GPU 1 枚）
```

1. 対象選択・凍結・対象外の不動（encoder+text_feat_map 対象 278 params / all 凍結 0 / 対象外変動 0）
2. ペナルティ勾配 = λ·(A·θ−B)（人工状態で最大相対誤差 1.8e-7）
3. 2 バッファ ≡ タスク別保持（値・勾配一致）
4. Fisher 推定（非負・有限、θ=θ* でペナルティ ≈ 0：const 項スケール 5.4e+04 に対し |−1.1e-2|）
5. 実データ 1 step で loss_ewc が乗り backward が通る

**数値の注記**: 実 Fisher 状態では B=F·θ* のため、θ≈θ* での勾配 λF(θ−θ*) は
「大きな 2 数の差」になり、fp32 の丸めが差分に対して相対 1e-3〜1e-2 程度に見える
ことがある（絶対誤差は 〜1e-7·|F·θ*| で訓練勾配のノイズより桁違いに小さく無害）。
数学的一致の検証は良条件の人工状態で行う（項目 2）。

## 追加検証（挙動レベル。2026-08-15 実測）

- **拘束の実効性**: θ0 アンカーの状態で実データ 60 step（AdamW lr 1e-4・grad clip 0.1）。
  λ=0 はドリフトが単調増加（3.14e-3 → 6.21e-3）、λ=1e6 は飽和して引き戻される
  （3.18e-3 → 3.62e-3。loss_ewc も 68.7 → 7.6 に減少）。ペナルティが実際の最適化経路を
  通して効くことの確認。なお AdamW は勾配スケールに漸近不変（m/√v 正規化）のため、
  少ステップ（10 step）では λ の差はほぼ現れない — これは実装の欠陥ではなく
  optimizer の性質（検証ログ `test1/test1b`）。
- **実ランナー統合**: `tools/train.py` で 30 iter（λ=1e3・状態あり）。`loss_ewc` が
  ログに乗り、アンカーから離れるにつれ 7.50 → 9.97 と増加。
