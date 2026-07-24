#!/bin/bash
# =============================================================================
# exp_023.5 tier1_plumbing.sh — Tier 1 プラミング確認（コンテナ内で実行）。
#   インタラクティブジョブでコンテナに入り、本スクリプトを走らせる。学習・評価はしない。
#
#   例（マスターノードで）:
#     srun --partition=a6000_ada_interactive --gres=gpu:4 --pty bash -c '
#       singularity exec --nv \
#         --bind /home/kouyou/VLM_OD_CL:/workspace/kouyou/mmdetection \
#         --bind /home/kouyou/datasets/rf100_domain:/workspace/kouyou/datasets/rf100_domain \
#         --bind /home/kouyou/datasets/o365v1_stage:/workspace/kouyou/datasets/o365v1_stage \
#         --bind /dataset01/MSCOCO:/workspace/kouyou/datasets/coco2017 \
#         --bind /home/kouyou/ckpt:/workspace/kouyou/ckpt \
#         /home/kouyou/sif/docker-image-of-mmdetection4singularity.sif \
#         bash /workspace/kouyou/mmdetection/experiments/exp_023.5/tier1_plumbing.sh'
# =============================================================================
set -uo pipefail
cd /workspace/kouyou/mmdetection

echo "== [1] import & CUDA =="
python3 -c "import mmdet,mmcv,torch; print('mmdet',mmdet.__version__,'mmcv',mmcv.__version__,'torch',torch.__version__); print('cuda gpus =',torch.cuda.device_count())"

echo "== [2] bind mountpoints（OK/NG） =="
for p in /workspace/kouyou/mmdetection \
         /workspace/kouyou/datasets/rf100_domain/underwater \
         /workspace/kouyou/datasets/o365v1_stage/Objects365_v1/2019-08-02/train \
         /workspace/kouyou/datasets/coco2017/annotations/instances_val2017.json \
         /workspace/kouyou/ckpt ; do
  if [ -e "$p" ]; then echo "OK  $p"; else echo "NG  $p"; fi
done

echo "== [3] COCO val 件数（期待: 5000 80 36781） =="
python3 -c "import json;d=json.load(open('/workspace/kouyou/datasets/coco2017/annotations/instances_val2017.json'));print(len(d['images']),len(d['categories']),len(d['annotations']))"

echo "== [4] config build & train データソースのパス解決（OK/NG） =="
python3 -c "
import os
from mmengine.config import Config
c=Config.fromfile('experiments/exp_023/configs/fullft_replay_underwater.py')
for d in c.train_dataloader['dataset']['datasets']:
    root=d.get('data_root',''); ann=d.get('ann_file','')
    p=ann if os.path.isabs(ann) else os.path.join(root,ann)
    print(('OK ' if os.path.exists(p) else 'NG '), p)
"

echo "== [5] θ0（事前学習重み）存在確認 =="
ls -la /workspace/kouyou/ckpt/*.pth 2>/dev/null || echo "NG  θ0 が /workspace/kouyou/ckpt に無い"

echo "== Tier1 plumbing 完了 =="
