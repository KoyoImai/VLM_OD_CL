# exp_050 実行手順（DGS / RF100 6 ドメイン・クラスタ）

design: [[design]]。移植の正本: [[../exp_049/implementation_plan]]。
**実行は design.md の承認後（2026-08-22 承認済み。本走は DTG 割当の報告後に確認）。**

## 0. 状態

| | 状態 |
|---|---|
| 学習 config 6 本（stage1）・評価 config 7 本・ドライバ・sbatch・検証 | 生成済み |
| 実行前検証（check_exp050_setup.py --gpu） | 5/5 相当（項目5はスモークで全量統計により確認） |
| スモーク（実ドライバ手順 t=1..3・1 epoch・本環境） | **合格（2026-08-22。エラーゼロ）** |
| DTG ドライラン（全量統計・6 ドメイン） | **6 グループ・併合なし → stage1 のみで本走可** |
| DTG 用特徴の全量抽出（本環境でゲート判定用） | §2 |
| クラスタへの push / clone / 投入 | 未実施（DTG 割当の報告 → 本走確認後） |

条件の骨子: RF100 共通枠（20 epoch・lr 1e-4・batch 4/GPU×4・ODVG・dn 有効・seed 0・
AMP 無し）＋ DGS 公式値（r16/α32・enhancer 両側 FFN・τ=150・ood 500/ZCOCO 200・
マージ 0.2・案A の 'inter-class'）。ckpt は **epoch_20 のみ・optimizer 状態なし**。

## 1. 実装の要点（exp_049 からの追加）

- **fast_chunked_predict の実装**（`projects/dgs_cl/detectors/gdino_dgs_base.py`）:
  公式リリースは predict() から呼ぶのに未定義（公式実験は chunked 不要なクラス数のみ）。
  上流 GroundingDINO.predict の chunked 分岐をそのままメソッド化した。
  videogames（87 クラス・chunked_size=40）の評価に必須。
- stage2 config は**生成していない**（design §2.3: DTG で併合が出たらドライバが
  exit 2 で停止し、ODVG 用 stage2 の追加実装を別途承認）。

## 2. 事前準備: DTG 用特徴の全量抽出（θ0・決定的）

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/exp_050/extract_feats_rf100.py
# -> experiments/exp_050/feats/<domain>/train/*.pt（約 80,300 件）
```

クラスタのジョブ内でも同じスクリプトが走る（ドライバ手順 0。決定的なので
どちらで作っても同一。クラスタ側で作り直すため feats の転送は不要）。

## 3. 実行前検証（再現手順）

```bash
python experiments/exp_050/check_exp050_setup.py                    # 静的
CUDA_VISIBLE_DEVICES=0 python experiments/exp_050/check_exp050_setup.py --gpu \
    --feats experiments/exp_050/feats                               # 全量統計で 5 項目
```

注意: --feats を省略すると 30 枚/ドメインのサブセットで検証するが、
統計が少なすぎてルーティングが OOD に倒れる（Appendix の 260 枚要件）。
項目 5 は全量統計で確認すること。

## 4. クラスタ投入（本走の確認後）

```bash
# 本環境: push
cd /workspace/kouyou/mmdetection
git add experiments/exp_050/ projects/dgs_cl/ experiments/exp_049/ papers/DGS_summary.md papers/DGS_supplemental.pdf
git commit -m "add exp_050 (DGS RF100) + exp_049 results + dgs_cl chunked predict"
git push

# クラスタ: 専用クローン（実行中の他クローンには触れない。PULL_WHILE_RUNNING.md）
ssh kouyou@192.168.170.100
git clone https://github.com/KoyoImai/VLM_OD_CL.git /home/kouyou/VLM_OD_CL_exp050
cd /home/kouyou/VLM_OD_CL_exp050
git log --oneline -1
ls experiments/exp_050/configs/ | wc -l     # 13
sbatch experiments/exp_050/sbatch_dgs.sh
squeue -u kouyou
```

- ジョブは a6000_ada・4 GPU・`--time=96:00:00`。特徴抽出（約 1.5 h）→ 6 ドメイン逐次。
- **DTG で併合が出るとドライバは exit 2 で停止する**（design §2.3。ゲート判定で
  併合なしを確認してから投入するので通常は起きない）。
- 再開は再投入（学習は epoch_20.pth、評価はディレクトリ有無でスキップ。
  **json の無い評価ディレクトリは消してから**）。

### 投入後 10 分の確認

```bash
LOG=/home/kouyou/logs/result_exp050_dgs_<JOBID>.txt
grep -m3 "\[t=1/6\]\|stage1 で学習" $LOG
grep -m3 "Epoch(train)" $LOG        # grad_norm が有限であること
ls /home/kouyou/VLM_OD_CL_exp050/experiments/exp_050/work_dirs/task_id_mapping.yaml
#   ↑ mapping が生成されていること（書き込み失敗は ERROR ログのみで続行される実装のため必ず確認）
```

## 5. 結果の転送（クラスタ → 本環境）

```bash
cd /workspace/kouyou/mmdetection
REMOTE=kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL_exp050/experiments/exp_050
LOCAL=experiments/exp_050
rsync -avh --progress \
  --include='*/' --include='*eval*/***' --exclude='*' "$REMOTE/" "$LOCAL/"
rsync -avh --progress \
  --include='*/' --include='*.log' --include='scalars.json' --include='config.py' \
  --include='task_id_mapping.yaml' --exclude='*' "$REMOTE/" "$LOCAL/"
rsync -avh --progress --partial --partial-dir=.rsync-partial \
  --include='*/' --include='epoch_20.pth' --exclude='*' "$REMOTE/" "$LOCAL/"
# 容量: ckpt 約 0.7 GB × 6 ≒ 4.2 GB（optimizer なし）
```

## 6. 完了後

結果は `experiments/exp_050/results/summary.md` に事実のみ記録（行動原理8）。
判定材料は design §3（各 t の mAP・ZCOCO、バッファ不使用系列との対照、
DTG 割当と ZCOCO の OOD 率）。
