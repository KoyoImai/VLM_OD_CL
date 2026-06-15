# exp_003 結果：underwater 単一ドメイン fine-tune（適応 ＋ 忘却）

- 日時: 2026-06-14〜15
- モデル: MM-Grounding DINO Swin-T、事前学習重みから fine-tune
- 学習: underwater train、20 epochs、AdamW lr=1e-4、実効バッチ64（per-GPU8×4GPU×累積2）、
  backbone・language_model lr_mult=0.1（全体 fine-tune）、4GPU 分散
- best チェックポイント: `best_coco_bbox_mAP_epoch_18.pth`
- 評価: (800,1333) プロトコル、4GPU 分散

## 適応 ＆ 忘却（zero-shot 比較）

| 指標 | zero-shot (exp_001) | fine-tune 後 (exp_003) | 変化 |
|---|---:|---:|---:|
| **underwater mAP（適応）** | 0.051 | **0.340** | **+0.289（約6.7倍）** |
| **COCO mAP（忘却/ZCOCO）** | 0.504 | **0.438** | **−0.066（−13.1%）** |

## 詳細指標

### underwater valid（適応, best epoch18）
| mAP | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|---:|---:|---:|---:|---:|---:|
| 0.340 | 0.518 | 0.365 | 0.169 | 0.339 | 0.452 |
- copypaste: `0.340 0.518 0.365 0.169 0.339 0.452`

### COCO2017 val（忘却）
| mAP | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|---:|---:|---:|---:|---:|---:|
| 0.438 | 0.601 | 0.475 | 0.293 | 0.469 | 0.591 |
- copypaste: `0.438 0.601 0.475 0.293 0.469 0.591`

## 学習中の val mAP 推移（underwater）
0.217(ep1) → 0.282(ep4) → 0.300(ep6) → 0.325(ep12) → 0.337(ep17) → **0.340(ep18)** → 0.339(ep20)
- ep6 で 0.30 到達後は緩やかに改善。ep18 で頭打ち（過学習の兆候は軽微）。

## OOM 対応の記録
- 初回 per-GPU16（累積なし・実効64）は epoch1 iter40-45 で GPU3 が CUDA OOM → 全ランク NCCL ハング。
- 多スケール拡張のピークで 40GB 超過が原因。per-GPU8 + 累積2（実効64維持）に変更し再実行→完走。
- 旧ログは `experiments/exp_003/_underwater_work_dir_oom_run1/` に退避。
