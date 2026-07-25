# exp_027 が `/local_cache` で落ちる件：原因と対処

## 結論：node03 の問題（2026-07-25 特定）

`sacct` の結果から、**node03 に割り当てられたジョブだけが失敗**している。

| ジョブ | 名前 | ノード | 結果 |
|---|---|---|---|
| 20510 | exp027_replay | **node03** | FAILED |
| 20521 | exp027_replay | **node03** | FAILED |
| 20480/20481/20482 | exp0235_debug | node04 | COMPLETED |
| 20483/20489/20490 | exp026_refpoints | node04 | COMPLETED |
| 20484 | exp024_ftA_replayfree | node05 | COMPLETED |
| 20485 | exp025_ftB_replayfree | node06 | COMPLETED |
| 20491 | exp026_refpoints | node06 | RUNNING |
| 20492 | exp026_refpoints | node05 | RUNNING |

**node03 は成功したジョブに一度も現れない。** node04/05/06 では同じ `mkdir -p /local_cache/${SLURM_JOB_ID}/datasets`
が通っている（exp_023.5 は `sbatch experiments/exp_023.5/sbatch.sh` で実行され成功）。

したがってスクリプトの問題ではなく、**node03 で `/local_cache` が用意されていない**。
`/local_cache/<ジョブID>` は Slurm の prolog が作り（`doc_cluster_setup/pages/SettingSlurmWorker.md`）、
`/local_cache` 自体は `/dev/sdb1` のマウント（`doc_cluster_manual/pages/Usage_MountDataset.md`）。
node03 でそのどちらかが欠けている。

---

## 対処：`sbatch.sh` に node03 の除外を入れた（2026-07-25）

`experiments/exp_027/sbatch.sh` に以下の1行を追加済み。投入時に毎回指定する必要はない。

```bash
#SBATCH --exclude=node03
```

**node03 が復旧したらこの行を削除すること。**

### 投入手順

クラスタのマスターノードで、まずコードを同期する。

```bash
cd /home/kouyou/VLM_OD_CL
git pull
```

デバッグ実行（数分〜十数分）。

```bash
RUN_STAGE=debug sbatch experiments/exp_027/sbatch.sh
```

ログの末尾に「全項目 OK」が出たら本実行へ。

```bash
tail -30 /home/kouyou/logs/result_exp027_replay_<JOBID>.txt
```

```bash
RUN_STAGE=fullft sbatch experiments/exp_027/sbatch.sh
RUN_STAGE=condB  sbatch experiments/exp_027/sbatch.sh
```

---

## 管理者への報告（任意）

node03 の状態を確認してから報告するとよい。

```bash
sinfo -p a6000_ada -o "%n %t %f %m %e"
scontrol show node node03
```

報告内容の要点。

- node03 に割り当てられたジョブで `mkdir: cannot create directory ‘/local_cache’: Permission denied` が発生
- 該当ジョブ: 20510, 20521（いずれも node03）
- node04/05/06 では同一スクリプトが正常動作
- `/local_cache`（`/dev/sdb1`）のマウント、または prolog の設定を確認いただきたい
