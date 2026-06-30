# exp_010: Image/Text Backbone も学習可能にした単独ドメインFT（凍結範囲 ablation の拡張）

## 位置づけ・目的
これまでの実験は Image Backbone(Swin) と Text Backbone(BERT) を**凍結**してきた（exp_004/005）。
しかしこの凍結は「選択であって制約ではない」。本実験は **exp_004/005（凍結FT）に対し、Swin と BERT も
学習可能（lr_mult=0.1）に戻した**単独ドメインFTを行い、**backbone まで学習させたときの
適応（C）と事前学習知識の忘却（A=COCO）**を測る。

- 継続学習の能力軸との対応：本実験が触るのは **C（新ドメイン適応）と A（事前学習知識=COCO保持）**。
  B（過去ドメイン保持）は単独学習のため対象外（系列実験は別途）。
- 関連：exp_003 は underwater のみ backbone/言語を 0.1× で学習した「全体FT」。本実験は **その 0.1× 設定を
  全ドメインへ拡張する**形でもある（→ underwater は exp_003 とほぼ一致する想定で、sanity check に使える）。

## 唯一の変更点（vs exp_004/005）
| 項目 | exp_004/005（凍結FT） | exp_010（本実験） |
|---|---|---|
| backbone(Swin) lr_mult | 0.0（凍結） | **0.1（学習・実効1e-5相当）** |
| language_model(BERT) lr_mult | 0.0（凍結） | **0.1（学習・実効1e-5相当）** |
| 学習対象 | head/enc/dec/neck のみ | **+ Swin + BERT** |

それ以外（base lr=1e-4 / 実効バッチ64=per-GPU8×4GPU×累積2 / 20epochs milestone[15] /
AdamW wd=1e-4 / clip_grad / データ・拡張 / 評価プロトコル(800,1333) / seed=0）は **exp_004/005 と完全同一**。

## 設定
- 初期重み: 事前学習 `grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_...pth`。
- config: `experiments/exp_010/configs/<domain>_finetune_unfrozen.py`
  - 標準ドメインFT config（`configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_<domain>.py`）を継承し、
    **paramwise の backbone・language_model の lr_mult を 0.1 に上書きするのみ**。
- 実行: GPU 4枚 分散（`tools/dist_train.sh ... 4`）。

## seed（重要・2026-06-27 修正）
- **全実験 seed=0 固定**（`randomness=dict(deterministic=False, seed=0)`）。研究計画 §5 と整合。
- 経緯: 初回 underwater(unfrozen) は config の randomness 未設定により**ランダム seed=1187114715** で学習されてしまった
  （適応0.359/ZCOCO0.404、参考値として保持＝`underwater_work_dir`、以後**破棄扱い**）。seed=0 で取り直す。
- 補足: frozen underwater も従来 seed=0 版が無かった（exp_004=ランダム seed、exp_005 は underwater を含まない）。
  exp_005 は他5ドメインを frozen×seed=0 で保持 → 本実験で **frozen×seed=0 underwater を補完**し、6ドメインの frozen×seed=0 を揃える。

## 対象ドメインと順序（seed=0）
1. **underwater を 2本**（同一 seed=0 で frozen vs unfrozen を公平比較）:
   - `underwater_finetune_unfrozen.py`（backbone/BERT lr_mult=0.1）→ `underwater_unfrozen_work_dir`
   - `underwater_finetune_frozen.py`（backbone/BERT lr_mult=0.0）→ `underwater_frozen_work_dir`
   学習の安定性（特に BERT 発散）と 適応/COCO を確認。
2. 問題なければ **残り5ドメイン単独・unfrozen×seed=0**（aerial / microscopic / videogames / documents / electromagnetic）。real_world は除外。
   （frozen×seed=0 の他5ドメインは exp_005 を流用可）

## 実行コマンド（underwater 例）
```bash
CKPT_DIR=./experiments/exp_010/underwater_work_dir
bash tools/dist_train.sh experiments/exp_010/configs/underwater_finetune_unfrozen.py 4 --work-dir $CKPT_DIR
BEST=$(ls $CKPT_DIR/best_coco_bbox_mAP_epoch_*.pth | tail -1)
# 適応評価（自ドメイン）
bash tools/dist_test.sh configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_underwater.py "$BEST" 4 --work-dir ./experiments/exp_010/underwater_eval
# 忘却評価（COCO=ZCOCO）
bash tools/dist_test.sh configs/mm_grounding_dino/eval_base_coco.py "$BEST" 4 --work-dir ./experiments/exp_010/underwater_coco
```

## 評価・比較
各ドメインで **適応 mAP（自ドメイン）** と **COCO mAP（=A 忘却の指標 ZCOCO）** を取得し、次と比較：
| 設定 | backbone/言語 | 比較の意味 |
|---|---|---|
| full FT (exp_003, underwaterのみ) | 0.1×学習 | exp_010-underwater と一致するか（sanity check） |
| frozen FT (exp_004/005) | 凍結 | backbone学習の効果（適応↑/忘却の変化） |
| 単純LoRA (exp_008) | 凍結+LoRA | 参考 |

## 期待される所見・問い
- backbone を学習させると **適応（自ドメインmAP）が上がるか**、その代償に **COCO 忘却（A）が増えるか**。
- これにより「凍結 vs 0.1×学習」の適応/忘却トレードオフ上の位置が分かり、手法設計（どこまで学習させてよいか）の根拠になる。

## 想定される懸念
1. **BERT 学習の不安定/発散**：lr_mult=0.1（実効1e-5）で安定範囲を狙うが、grad_norm を監視し、
   発散（inf/NaN, loss上昇）時は当該本を止め lr_mult を見直す。**underwater 1本での先行確認はこのため**。
2. num_classes 不一致の COCO 評価警告（既知・検出に影響なし）。
3. backbone 学習で学習パラメータ増 → メモリ。per-GPU8 で実績内だが OOM 時は調整。

## 成功基準
- 学習が安定して回り（base 重みが意図通り学習・発散なし）、6ドメインの適応 mAP と COCO 忘却を取得。
- frozen FT(exp_004/005)・full FT(exp_003) と比較し、**「Swin/BERT も学習させること」が適応と忘却(A)に与える効果**を定量化。

## 承認ゲート
本実験は新規学習のため、本 design.md の承認後に config 作成・実行する（行動原理①②）。

## 関連
- 凍結は選択: [[../../setup_and_code_analysis/code_analysis/mmgrounding_dino_architecture]]（§5/§6）
- 元実験: exp_003(全体FT 0.1×) / exp_004(underwater凍結) / exp_005(全6凍結) / exp_008(単純LoRA)
- 能力軸の議論: [[../exp_009/minutes]]
