# exp_022: 特徴抽出・融合部の細分化探索（実験3・問い2 の 2 サイクル目）

作成 2026-07-15。実行前のユーザー承認ゲート（行動原理1）。
実験ノートの正本は [[../../experiment_notes/note03]]。本書はそれを実行可能な形に
落としたもので、位置付け・目的・分岐規則は note03 に従う。

## 1. 目的（答える問い）

exp_019 の条件A（特徴抽出・融合部、160.64 M）の内部を機能単位に細分し、
どの機能群（またはその組み合わせ）を学習すれば条件A・条件3 と同等の
新ドメイン適応が得られるかを探索する。本実験は探索（仮説を立てる一段階目）で
あり、適応可否の確定は行わない。確定は次の仮説検証と機能分析が担う。

機能単位は 3 群: 画像特徴抽出（Swin + neck）、テキスト特徴抽出
（BERT + text_feat_map）、特徴融合（feature enhancer + level_embed）。
level_embed は encoder 入力への加算のため融合側に帰属（exp_019 と同じ裁定）。

## 2. 条件（学習可能パラメータ数つき）

3 群の非空部分集合は 7 通り。うち全体集合 = 条件A（exp_019 実施済み）、
テキスト特徴抽出単独 = exp_018 実施済みのため、新規学習は 5 条件。

| 条件 | 学習するモジュール | 学習可能パラメータ | 出所 |
|---|---|---|---|
| A1 | Swin + neck | 29.64 M (17.1%) | 新規 |
| A2 | BERT + text_feat_map | 109.09 M (63.1%) | exp_018 流用 |
| A3 | feature enhancer + level_embed | 21.91 M (12.7%) | 新規 |
| A4 | A1 + A2（抽出全体） | 138.73 M (80.2%) | 新規 |
| A5 | A1 + A3（画像抽出+融合） | 51.55 M (29.8%) | 新規 |
| A6 | A2 + A3（テキスト抽出+融合） | 131.00 M (75.8%) | 新規 |
| (A) | A1+A2+A3 = 条件A 全体 | 160.64 M (92.9%) | exp_019 流用 |

下流（decoder / bbox_head / memory_trans_fc / memory_trans_norm /
query_embedding / dn_query_generator、12.29 M）は全条件で凍結
（exp_019 で適応に不要と判定済み）。これにより全条件が条件A の内部分割となり、
差は分割だけに帰属する。

## 3. プロトコル（exp_017/018/019 と同一）

- 基底 config: `grounding_dino_swin-t_finetune_8xb4_20e_{domain}.py`
- seed=0（randomness 明示）、20 epoch、milestones [15]、base lr 1e-4、
  4 GPU、実効バッチ 16
- 学習率規約: 学習するモジュールの lr_mult は条件3 と同一
  （Swin 0.1 / BERT 0.1 / neck・text_feat_map・encoder・level_embed 1.0）。
  凍結は lr_mult=0.0（exp_017 以降の慣行）
- ドメイン: underwater / videogames / electromagnetic（実験1・2 と同じ 3 つ。
  aerial / microscopic / documents は次の仮説検証のために未使用のまま残す）
- 評価: 適応 = 各ドメイン valid の bbox_mAP（20 epoch 中の best epoch、
  exp_019 と同じ扱い）。観察用に ZCOCO（`eval_base_coco.py` 相当、
  best epoch の ckpt）。判定（分岐）は適応 mAP のみで行う

## 4. 分岐規則（note03 の宣言。実行前に固定）

条件3（unfrozen）の適応 mAP の 90% 以上に達した機能群を、次の仮説検証
（「モジュール XX を学習すれば適応が条件3 と同等になる」の XX 候補）へ持ち越す。

適用する具体値（条件3 実測 × 0.9）:
underwater 0.359 → **0.323** / videogames 0.782 → **0.704** /
electromagnetic 0.502 → **0.452**（3 ドメインすべてで達成した条件を持ち越す）

- 達成が 1 条件: その機能群を持ち越す
- 達成が複数条件: 全てを持ち越す（次実験で切り分け）
- 達成なし: 条件A 全体のみを持ち越す（= この粒度の部分集合では適応を担えず、
  特徴抽出・融合の全体が必要、という仮説になる）

本実験では「適応可能」の確定判定は行わない（探索であり、90% は持ち越しの
足切りであって同等性の主張ではない）。

## 5. 実行前検証

学習開始前に、各条件の config について optimizer の param group を dump し、
lr_mult > 0 のパラメータ集合が上表の定義と一致すること（誤凍結・凍結漏れなし、
和が条件A の 160.64 M + 凍結 12.29 M = 全体と整合すること）を確認して
`param_assignment.md` に記録する（exp_019 と同じ手続き）。

## 6. この実験が言えないこと（限界）

- 探索であり、持ち越した機能群の適応可否は確定しない（仮説検証は未使用
  3 ドメイン + 事前登録基準で別途行う）
- なぜその機能群が適応を担うかは示さない（機能分析は本実験の後、
  結果を見てから設計する）
- 単発ドメイン学習であり、逐次学習での挙動は測らない

## 7. コスト

新規 5 条件 × 3 ドメイン = 15 run。実測の 1 run 所要
（underwater 約 6.5 h / videogames 約 4.8 h / electromagnetic 約 14 h）から
1 条件セット約 25 h、合計約 127 h ≈ 4 GPU 直列で 5 日強。
ZCOCO 評価は 1 run 数分で誤差の範囲。

## 8. 成果物

- `experiments/exp_022/configs/{cond}_{domain}.py`（15 本）
- `experiments/exp_022/{domain}_{cond}_work_dir` / `{domain}_{cond}_zcoco`
- `experiments/exp_022/param_assignment.md`（実行前検証の記録）
- `experiments/exp_022/results/`（適応 / ZCOCO の事実記録。解釈は書かない）

## 関連
- 実験ノート: [[../../experiment_notes/note03]]
- 前サイクル: [[../exp_019/design]]（条件A/B の二重乖離）
- 流用する既存結果: exp_018（A2 = BERT + text_feat_map）、exp_019（条件A）
