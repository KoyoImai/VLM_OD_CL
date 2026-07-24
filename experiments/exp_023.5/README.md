# exp_023.5 実行手順 — クラスタ移行のデバッグ

クラスタで本環境と同じ挙動が再現されるかを確認するデバッグ実験の実行手順。
設計・判定の考え方は [design.md](./design.md)、クラスタ運用の全体像は [../../README_cluster.md](../../README_cluster.md) を参照。

すべて**クラスタのマスターノード**で操作する（計算ノードへの直接 SSH 不可）。

## 構成（案A）

| ファイル | 役割 |
|---|---|
| `sbatch.sh` | 器。`#SBATCH`＋rf100 の local_cache ステージング＋Singularity 起動。`train_val.sh` を呼ぶ |
| `train_val.sh` | 中身。Tier2-0 ゼロショット／2-1 1ep学習／2-2 評価。既存 config を無変更で実行 |
| `tier1_plumbing.sh` | Tier1。import・bind・COCO件数・config解決・θ0 存在をインタラクティブに確認 |

---

## 0. 事前準備（初回1回だけ）

```bash
mkdir -p /home/kouyou/logs /home/kouyou/ckpt

# θ0（事前学習重み）を取得して配置（計算ノードはオフライン想定 → マスターノードで wget）
cd /home/kouyou/ckpt
wget https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth
```

前提（未了なら先に）:
- データ転送（[README_cluster.md](../../README_cluster.md) §4）: `rf100_domain`, `o365v1_stage`
- コード同期（§3）: `cd /home/kouyou/VLM_OD_CL && git pull`
- `.sif` 配置（§2）: `/home/kouyou/sif/docker-image-of-mmdetection4singularity.sif`

---

## 1. Tier 1: プラミング確認（インタラクティブ, 数分）

学習せずに、環境・バインド・パス解決を確認する。

```bash
srun --partition=a6000_ada_interactive --gres=gpu:4 --pty bash -c '
  singularity exec --nv \
    --bind /home/kouyou/VLM_OD_CL:/workspace/kouyou/mmdetection \
    --bind /home/kouyou/datasets/rf100_domain:/workspace/kouyou/datasets/rf100_domain \
    --bind /home/kouyou/datasets/o365v1_stage:/workspace/kouyou/datasets/o365v1_stage \
    --bind /dataset01/MSCOCO:/workspace/kouyou/datasets/coco2017 \
    --bind /home/kouyou/ckpt:/workspace/kouyou/ckpt \
    /home/kouyou/sif/docker-image-of-mmdetection4singularity.sif \
    bash /workspace/kouyou/mmdetection/experiments/exp_023.5/tier1_plumbing.sh'
```

**合格条件**: 出力の各行が `OK`。特に —
- `import mmdet/mmcv` が通り `cuda gpus = 4`
- 5つの bind パスが `OK`（`o365v1_stage` の入れ子 bind 含む）
- COCO 件数が `5000 80 36781`
- train データソースのパスが `OK`
- θ0 が存在

`NG` が出たら、その項目（多くは bind 設定か θ0 配置）を直してから次へ。

---

## 2. Tier 2: バッチ実行（`sbatch.sh`）

`RUN_STAGE` 環境変数で範囲を選ぶ（`sbatch` は既定で環境変数をジョブへ渡す）。

```bash
# ゼロショットのみ（最速・数分。θ0ロード＋COCOバインド＋評価経路。期待 mAP ≈0.504）
RUN_STAGE=zeroshot sbatch experiments/exp_023.5/sbatch.sh

# 1エポック学習のみ（checkpoint 保存まで確認）
RUN_STAGE=train sbatch experiments/exp_023.5/sbatch.sh

# 評価のみ（直前の 1ep 学習 ckpt を 適応 + ZCOCO 評価）
RUN_STAGE=eval sbatch experiments/exp_023.5/sbatch.sh

# 全部（ゼロショット → 1ep学習 → 評価。RUN_STAGE 省略時も all）
sbatch experiments/exp_023.5/sbatch.sh
```

状態確認・ログ:
```bash
squeue -u kouyou
tail -f /home/kouyou/logs/result_exp0235_debug_<JOBID>.txt
```

`train_val.sh` は `sbatch.sh` からコンテナ内で呼ばれる実体。通常は直接触らない。
インタラクティブに単体実行したいときは、Tier1 の `srun … singularity exec …` の末尾を
`bash …/experiments/exp_023.5/train_val.sh zeroshot` に変えて使う。

---

## 3. 推奨の実行順

1. `tier1_plumbing.sh`（インタラクティブ）で全 `OK`
2. `RUN_STAGE=zeroshot sbatch …` → ゼロショット COCO が **≈0.504** か確認（数値パリティ本命）
3. `sbatch …`（all）で 1ep 学習＋評価まで通し、出力が出ることを確認

---

## 4. 結果の見方

出力はホーム側 `experiments/exp_023.5/` に残る（`--bind` 経由でホームに書かれる）。

```bash
cd /home/kouyou/VLM_OD_CL

# ゼロショット COCO（期待 ≈0.504）
grep -oE "coco/bbox_mAP: [0-9.]+" experiments/exp_023.5/zeroshot_coco/*/*.log | tail -1

# 1ep 学習後の適応（on underwater）と ZCOCO
grep -oE "coco/bbox_mAP: [0-9.]+" experiments/exp_023.5/eval_after_debug/on_underwater/*/*.log | tail -1
grep -oE "coco/bbox_mAP: [0-9.]+" experiments/exp_023.5/eval_after_debug/zcoco/*/*.log | tail -1
```

判定の考え方（[design.md](./design.md) と一致）:
- **数値パリティは Tier2-0 のゼロショット 0.504 で見る**（学習不要・θ0同一・評価決定的）。
- 1ep 学習後の値は「評価が最後まで回り数値が出るか」の確認用で、数値の一致は見ない（本環境の1epアンカーは後日、4GPU で取得予定）。

---

## 5. 成果物の場所

- ゼロショット: `experiments/exp_023.5/zeroshot_coco/`
- 1ep 学習: `experiments/exp_023.5/debug_underwater_work_dir/`
- 1ep 後の評価: `experiments/exp_023.5/eval_after_debug/{on_underwater, zcoco}/`
- ジョブログ: `/home/kouyou/logs/{result,error}_exp0235_debug_<JOBID>.txt`
