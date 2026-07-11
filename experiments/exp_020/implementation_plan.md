# exp_020 実装計画書: ZiRa の MM-Grounding DINO 上での再現

作成 2026-07-10。実装前のユーザー承認ゲート。手法の詳細は
[[../../papers/ZiRa_implementation_notes]] を正本とし、本書は「mmdetection の上で
どう作るか」だけを定める。

## 1. 方針（ユーザー指定）

公式実装（https://github.com/JarintotionDin/ZiRaGroundingDINO）に準拠し、
mmdetection に合わせて調整する。学習のハイパーパラメータは論文・公式実装ではなく
本プロジェクトの exp_xxx 系列（lr 1e-4、20 epoch、seed=0、4 GPU）に合わせる。
実装コードは mmdetection の同種プログラム群と同じディレクトリ（`mmdet/models/`
配下）に**新規ファイルとして**追加する。既存ファイルは `__init__.py` を含めて
一切変更しない。レジストリへの登録は、config の `custom_imports` で新規ファイルを
直接 import することで実現する（`__init__.py` への追記が不要になる）。mmdet は
editable install のため、新規ファイルは再インストールなしで即座に有効になる。
mmdetection の構造上忠実に実装できない点が出たら、実装を止めて相談する。

## 2. 何を作るか（新規ファイル台帳）

既存ファイルの変更はゼロ。追加する新規ファイルは以下の通りで、本節が
「どこに何を置いたか・役割は何か」の記録を兼ねる（実装後も本書を正とする）。

mmdet 側（同種プログラム群と同じディレクトリに新規追加）:

| パス | 役割 |
|---|---|
| `mmdet/models/layers/zira_layers.py` | ZiRaLinear / ZiRaConv2d（RDB 本体。凍結重み+HLRB+LLRB+scaling と ZiL 計算） |
| `mmdet/models/necks/zira_channel_mapper.py` | ZiRaChannelMapper（ChannelMapper のサブクラス。GN 前に RDB 加算） |
| `mmdet/models/detectors/zira_grounding_dino.py` | ZiRaGroundingDINO（GroundingDINO のサブクラス。text_feat_map 差し替え・ZiL 回収・凍結） |

experiments 側（実験固有物）:

| パス | 役割 |
|---|---|
| `experiments/exp_020/implementation_plan.md` | 本書（実装計画・新規ファイル台帳） |
| `experiments/exp_020/design.md` | 学習実験の設計書（実装検証後に別途作成・承認） |
| `experiments/exp_020/configs/zira_{domain}.py` | 既存ドメイン config を継承し type を差し替える学習 config |
| `experiments/exp_020/check_zira_setup.py` | 実装検証スクリプト（学習前の健全性確認） |
| `experiments/exp_020/merge_hlrb.py` | Rep+ 用スタンドアロンのマージスクリプト（将来の逐次実験用） |
| `experiments/exp_020/run_train.sh` | exp_019 と同形式の実行スクリプト |

各 config の先頭に
`custom_imports = dict(imports=['mmdet.models.layers.zira_layers',
'mmdet.models.necks.zira_channel_mapper',
'mmdet.models.detectors.zira_grounding_dino'], allow_failed_imports=False)`
を記載して登録する。

## 3. 設計の要点

### 3.1 RDB 層（mmdet/models/layers/zira_layers.py）

公式の RepZeroLinear / RepZeroConv2d と同じ数式を、「凍結された元の層 + 学習される
2 枝」を 1 モジュールに同居させる形で実装する。

- `ZiRaLinear(nn.Linear)`: `self.weight / self.bias` は**事前学習の text_feat_map の
  重みそのもの**（凍結）。加えて `hlrb`（同形 Linear、定数 1e-8 初期化）、
  `llrb`（同形 Linear、0 初期化）、`scaling`（スカラー、初期値 0.1）を持つ。
  forward は `F.linear(x, W, b) + scaling·hlrb(x) + llrb(x)`。
- `ZiRaConv2d` も同様（1×1 ×3 本と 3×3 stride-2 ×1 本）。

この設計にする理由は checkpoint の互換性で、元の重みのキー名
（`text_feat_map.weight` 等）が変わらないため、既存の事前学習 checkpoint を
そのまま load できる（RDB の新キーは missing keys として無視される）。また
検出器の 3 箇所の呼び出し（`grounding_dino.py:482,559,583`）はモジュール差し替え
だけで済み、メソッドの上書きが不要になる。

ZiL は各層が forward（学習時のみ）で
`zil = SmoothL1(scaling·hlrb(x), 0) + SmoothL1(scaling·hlrb(x) + llrb(x), 0)`
（reduction='mean'）を計算して自身に保持し、検出器が回収する。

**評価時の forward は学習時と同じ全和**（元層 + s·HLRB + LLRB）とする。公式コードは
評価時 LLRB のみだが、それはタスク末尾に HLRB を LLRB へ融合してから評価する前提
であり、融合は線形なので「融合後に LLRB のみ」=「融合前に全和」。epoch ごとの val で
best checkpoint を選ぶ本プロジェクトの手順では全和で評価するのが等価かつ正しい。

### 3.2 neck（mmdet/models/necks/zira_channel_mapper.py）

`ZiRaChannelMapper(ChannelMapper)` を新規登録する。convs / extra_convs の名前と
構造は親のまま保持し（checkpoint 互換）、RDB conv 4 本を追加。forward だけ
上書きし、公式と同じく **GroupNorm の前**で加算する:
`out_l = GN_l(conv_l(x_l) + rdb_l(x_l))`。ConvModule を `.conv` と `.norm` に分解して
呼ぶだけで実現でき、上流コードの変更は不要。

### 3.3 検出器（mmdet/models/detectors/zira_grounding_dino.py）

`ZiRaGroundingDINO(GroundingDINO)` を新規登録する。変更は 3 点のみ。

1. `__init__` 末尾で `self.text_feat_map` を `ZiRaLinear` に差し替える。
2. `loss()` を「親の loss() を呼んだ後、全 RDB 層から ZiL を回収して
   `loss_zil = λ × Σ zil` を losses 辞書に追加」する形で上書きする
   （ログに loss_zil が個別表示され、監視できる）。
3. 凍結: `__init__` 末尾で、パラメータ名に `hlrb` / `llrb` / `scaling` を含むもの
   **以外**を全て `requires_grad=False` にする（公式の freeze_all + unfreeze adapter
   と同じ方式）。BERT の dropout 等が学習中も有効なままである点も公式と同じ。

### 3.4 学習率の差別化（η）

optimizer 側は config の `paramwise_cfg.custom_keys` に `llrb: dict(lr_mult=0.2)` を
足すだけでよい（HLRB=lr×1、LLRB=lr×0.2）。凍結は requires_grad で済んでいるので
custom_keys に凍結エントリは不要。

### 3.5 Rep+（タスク間の融合）: 公式実装の正確な挙動と本計画の対応

公式実装のタスク間処理を train_multidatasets.py まで追跡して確定させた記録。

タスク t の学習中は HLRB（lr×1）・LLRB（lr×η）・scaling の 3 種が同時に学習される
（LLRB は名前が freeze_* だが凍結されておらず、低学習率で動く）。タスク t の終了時、
Trainer.after_train が model.after_train() → 各 RDB の __rep__() を呼び、
`W_llrb += s·W_hlrb`、`b_llrb += s·b_hlrb` を実行して HLRB を 1e-8、s を 0.1 に
再初期化する。保存順序には罠があり、model_final.pth は最終 iter の after_step で
一度（融合前）保存されるが、彼らは PeriodicCheckpointer をサブクラス化して
after_train でもう一度 step を呼んでおり、融合後の状態で**上書き**される
（train_multidatasets.py:319-322）。タスク t+1 は新しいモデルを構築して
この融合済み checkpoint を strict=False で読み込むため、開始状態は常に
「LLRB=過去タスクの蓄積、HLRB≈0、s=0.1」になる。optimizer はタスクごとに
新規作成され、Adam のモーメントは持ち越されない。最終評価は base + LLRB のみの
forward だが、直前に融合済みなので情報の欠落はない。

本計画（exp_020、単発ドメイン）では学習ループ中の融合は発生せず、評価時 forward を
全和（base + s·HLRB + LLRB）にしているため、任意の epoch の評価が「その時点で
融合した場合」と数学的に一致する（3.1 参照）。将来の逐次実験用に、公式の __rep__ と
同一の処理（weight と bias の融合、HLRB→1e-8、s→0.1 の再初期化）を checkpoint に
適用して保存するスクリプト `experiments/exp_020/merge_hlrb.py` を用意する。
本プロジェクトの best_coco_bbox_mAP checkpoint は融合前の状態で保存されるので、
逐次学習で次ドメインへ渡す前に必ずこのスクリプトを通す。optimizer をタスクごとに
リセットする点は、ドメインごとに独立の学習 run として実行する我々の方式で
自動的に一致する。

## 4. ハイパーパラメータの決定表

| 項目 | 値 | 出所 |
|---|---|---|
| lr / optimizer | AdamW 1e-4, wd 1e-4, clip_grad max_norm 0.1 | exp_xxx（ベース config 継承） |
| epoch / decay | 20 epoch, milestones [15] | exp_xxx |
| batch / GPU | 4 GPU 分散（dist_train） | exp_xxx |
| seed | randomness=dict(seed=0) 明示 | プロジェクト規約 |
| λ（ZiL 重み） | 0.1 | 公式実装（exp_xxx に対応物なし） |
| η（LLRB lr 比） | 0.2 | 公式実装 softfreeze 構成 |
| s 初期値 | 言語・視覚とも 0.1 | 公式実装 |
| ZiL ノルム | SmoothL1（mean） | 公式実装の既定 |
| HLRB 初期値 | 1e-8 | 公式実装 |
| dn クエリ | **有効のまま**（公式は無効） | 下記 5.1 参照 |

注意点として、公式の lr は 1e-3、本計画は exp_xxx 準拠の 1e-4 であり、λ=0.1 の
妥当性は損失バランスの意味で自明でない。学習曲線で loss_zil と適応の両方を監視し、
適応が立ち上がらない場合は λ の再検討を相談する。

## 5. 公式実装からの意図的な逸脱（3 点）

1. **dn を無効化しない。** 公式 ZiRa は dn_number=0 だが、本プロジェクトの全基準線
   （exp_010〜019）は dn 有効で学習しており、無効化すると ZiRa だけ学習条件が変わり
   比較が壊れる。dn の損失は凍結された本体には影響せず RDB の勾配に加わるのみ。
2. **評価時 forward を全和にする**（3.1 の通り、数学的に等価）。
3. **スケジュールをタスクあたり 2000 iter でなく 20 epoch にする**（ハイパラを
   exp_xxx に合わせるというユーザー方針そのもの）。

## 6. 実装後・学習前の検証（check_zira_setup.py）

学習を開始する前に以下を確認し、結果を報告してから design.md の承認に進む。

1. **パラメータ監査**: 学習可能テンソルが RDB（hlrb/llrb/scaling）のみであること、
   本体 896 テンソルが全て requires_grad=False であることを列挙して確認。
2. **θ0 等価性**: RDB 初期状態（hlrb≈0, llrb=0）で、同一画像に対する検出出力が
   素の MM-GDINO と一致（誤差 ~1e-6）することを確認。checkpoint の load で
   missing/unexpected keys が想定通りかも確認。
3. **勾配疎通**: 1 step の学習で RDB 全パラメータに非ゼロ勾配が到達し、
   本体パラメータの勾配が計算されない（または更新されない）ことを確認。
4. **loss_zil の出力**: ログに loss_zil が現れ、初期値がほぼ 0 であることを確認。
5. **推論等価性（融合 vs 全和）**: RDB に乱数を入れたモデルで、「全和 forward の
   検出出力」と「merge_hlrb.py 適用後の base + LLRB の検出出力」が一致（誤差 ~1e-5）
   することを確認。評価時 forward を全和にする置き換え（3.1）の数値的な裏取り。
6. **逐次開始状態**: merge_hlrb.py 適用後の checkpoint を読み込んだモデルが、
   LLRB=融合値・HLRB=1e-8・s=0.1 になっていることを state_dict レベルで確認。
   公式のタスク t+1 開始状態（train_multidatasets.py の load 経路）との一致の裏取り。

## 7. 実行範囲（ユーザー指定 2026-07-10: 全ドメイン）

実装検証後、全 6 ドメイン（underwater / aerial / videogames / microscopic /
documents / electromagnetic。real_world は CLAUDE.md の規約により除外）で
単発ドメイン学習 + ZCOCO 評価（exp_019 と同一プロトコル）を行う。
これにより frozen / unfrozen / neckTfm / bert_tfm / extfusion / downstream の
既存条件と同じ表に ZiRa 行を並べられる。逐次学習（Rep+ が効く設定）は
別実験として設計する。

## 8. 相談事項の裁定記録

- 実行範囲: 全ドメイン（ユーザー指定 2026-07-10）。
- dn の扱い: 有効のまま（5.1 の理由。dn は学習時のみの収束安定化テクニックで、
  無効化すると既存基準線と学習条件が揃わなくなるため）。
- ZiL のノルム: SmoothL1（公式コードの既定。論文付録でも hAP 最良）。
