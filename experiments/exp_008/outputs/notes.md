# exp_008 実験メモ（LoRA の挙動・所見・次への示唆）

作成日: 2026-06-26。数値集約は [`../results/lora_summary.md`](../results/lora_summary.md)、生ログは
[`../results/sweep_underwater.txt`](../results/sweep_underwater.txt)。本書は定性的な所見と次への示唆を残す。

---

## 1. 実装の確立（この exp の第一目的）

MM-Grounding DINO（mmdet）への LoRA 注入を `projects/lora_cl/` に独立モジュールとして実装し、
`custom_imports` でオプトイン読み込みする形で確立した（mmdet 本体は不変）。

- `lora_layers.py`：`LoRALinear`（`Wh + (α/r)·B(A h)`）。元 weight/bias を同キー名で保持し事前学習 ckpt をそのままロード可。A 乱数・B ゼロ初期化（初期は元と同一出力）。`merge()` で base へ統合。
- `grounding_dino_lora.py`：`GroundingDINOLoRA`。対象 `nn.Linear` を置換し LoRA 以外を `requires_grad=False`。
- `optim.py`：`TrainableParamsConstructor`。optimizer に LoRA のみ登録。
- `merge_lora.py`：学習後に LoRA を base へマージし plain ckpt 化（評価は既存 plain config で実施）。

**検証済みの事実**：実学習で base 重み（LoRALinear base / MHA in_proj / Swin / BERT）が完全不変（max|Δ|=0）、
LoRA(lora_A/B) のみ変化、optimizer 登録は LoRA のみ（2.40M）。
→「学習されるのは LoRA だけ・事前学習済みモデルは一切更新されない」を実証。

### 実装上の注意（次の自分への申し送り）
- MHA（`in_proj` は生 Param、`out_proj` は functional 経由）は LoRA 対象外・自動スキップ。attention に LoRA を入れたい場合はこの制約を要回避。
- `text_feat_map` / `memory_trans_fc` は top-level 名のため `include=['encoder','decoder','bbox_head']` に**含まれない**。付ける場合は include に明示追加。
- 評価はマージ後 plain config。COCO 評価の num_classes 不一致警告は既知・影響なし。

---

## 2. 挙動の所見（sweep から）

1. **適応の律速は lr**。rank を上げても lr=1e-4 では frozen FT(0.316) に届かない。lr=1e-3 で一気に回復。
   → LoRA は「学習パラメータが少ない＝低 lr」という直感は誤りで、むしろ通常学習より高めの lr が要る。
2. **素の LoRA はマージしても忘却を減らさない**。適応を取り戻すと COCO が崩れる、という full/frozen FT と同形のトレードオフをなぞるだけ。
3. **lr=5e-3 は高すぎ**。適応は伸びず COCO だけ壊滅（0.136/0.097）、r64 は発散。学習安定性の上限が見えた。

---

## 3. 反直感だった点・つまずき

- 当初 lr=1e-4（通常学習と同一）で始めたら適応 0.218 と低く、「LoRA で忘却が減る」どころか「適応だけ下がる」結果に見えた。原因切り分けのため lr×rank sweep を追加し、**lr 不足**が主因と判明（容量 rank は副次的）。
- → 単一点の結果で手法の良し悪しを判断せず、lr 軸を必ず確認すべき、という教訓。

---

## 4. 次への示唆（手法 exp_009 以降）

- **比較下地**：この sweep が引いた「素の LoRA トレードオフ曲線」が手法の評価基準。これを Pareto 改善できれば貢献。
- **設計の焦点**：rank/lr の微調整では曲線は動かない（本 exp が実証）。動かすには **LoRA 上の干渉制御**——
  - 過去ドメインには（ほぼ）直交化、COCO は soft 正則化で緩く保護（hard 直交は適応を壊す＝exp_007-E）。
  - rank/スケールを抑えマージ量自体を小さくして奪い合いを緩和。
- **既定ハイパラ**：lr=1e-3、rank=r16〜r32 を出発点にする。

## 5. 残タスク（exp_008 のスコープ外）

- 第1層：electromagnetic / documents の単純 LoRA（lr=1e-3, r16）取得 → 3 ドメインで参照値を揃える。
- 第2層：3 ドメイン逐次 素朴FT（下限）＋ joint（上限）。design.md 承認ゲートを通してから実行。

## 関連
- [[../results/lora_summary.md]] / [[../design.md]] / [[../../exp_008.5/research_plan_minutes]] / [[../../exp_007/design.md]]
