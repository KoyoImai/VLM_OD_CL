# exp_026 〜 exp_029 の結果をクラスタから本環境へ持ち帰る手順

作成: 2026-08-06

対象: **exp_026**（個別学習・オラクル）／**exp_027**（リプレイあり逐次FT）／
**exp_028**（条件A + L2 蒸留 + リプレイ）／**exp_029**（条件B + L2 蒸留 + リプレイ）

方向: クラスタ `kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL/experiments/`
　　→ 本環境 `/workspace/kouyou/mmdetection/experiments/`

**結果の転送は rsync のみで行う。git は使わない**（2026-07-26 の方針）。
汎用手順は [RETRIEVE_FROM_CLUSTER.md](../RETRIEVE_FROM_CLUSTER.md)、
exp_028/029 の既存手順は [RETRIEVE_EXP028_029.md](../RETRIEVE_EXP028_029.md) にもある。

---

## 0. 事前確認

### 0.1 転送元のクローンを確かめる

exp_026〜029 は原クローン `/home/kouyou/VLM_OD_CL` から投入した（exp_033 だけが
別クローン `VLM_OD_CL_new`）。念のため確認する。

```bash
ssh kouyou@192.168.170.100 'ls -d /home/kouyou/VLM_OD_CL*/experiments/exp_02{6,7,8,9}/*_work_dir 2>/dev/null | sed "s|/experiments.*||" | sort -u'
```

### 0.2 ジョブが終わっているか（**最優先**）

**実行中に rsync すると書き込み途中のファイルを掴む。**

```bash
ssh kouyou@192.168.170.100 'squeue -u kouyou'
```

対象のジョブ名は次のとおり。これらが残っていないことを確認する。

```
exp026_refpoints
exp027_replay
exp028a_kdAB_condA_l2      exp028b_kdDE_condA_l2
exp029a_kdAB_condB_l2      exp029b_kdDE_condB_l2
```

途中経過だけ見たい場合は転送せずログを読む。

```bash
ssh kouyou@192.168.170.100 'tail -30 /home/kouyou/logs/result_exp029b_kdDE_condB_l2_*.txt'
```

### 0.3 本環境に既にあるもの（2026-08-06 時点）

| 実験 | 本環境の ckpt 数 | 本環境の容量 | 備考 |
|---|---:|---:|---|
| exp_026 | 143 | 275 GB | indiv（condA / condB / zira）＋ oracle |
| exp_027 | 120 | 234 GB | fullft_replay ＋ condB_replay（2条件 × 3ドメイン × 20 = 120 で完了） |
| exp_028 | 240 | 467 GB | 4条件 × 3ドメイン × 20 = 240 で完了 |
| exp_029 | 149 | 290 GB | kdA / kdB / kdD のみ。**kdE と kdD の一部が未取得** |

**rsync は差分のみを転送する**ので、上表で欠けている分だけが流れる。

### 0.4 容量

```bash
df -h /workspace/kouyou
ssh kouyou@192.168.170.100 'du -sh /home/kouyou/VLM_OD_CL/experiments/exp_02{6,7,8,9}'
```

本環境 `/workspace/kouyou` の空きは **1.8 TB**（2026-08-06 時点）。
4実験すべてを完全に揃えると合計 **約 1.7 TB** の見込み（既に持っている分を含む）。
差分だけなら数百 GB で収まるが、**転送前に必ず上のコマンドで実測する**。

---

## 1. 転送（実験ごと）

```bash
REMOTE=kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL/experiments
LOCAL=/workspace/kouyou/mmdetection/experiments

for e in exp_026 exp_027 exp_028 exp_029; do
  rsync -avh --progress --partial --partial-dir=.rsync-partial \
    --exclude='/configs/' --exclude='/*.sh' --exclude='/*.py' --exclude='/*.md' \
    "$REMOTE/$e/" "$LOCAL/$e/"
done
```

`--exclude` の先頭の `/` は転送元のルート（＝各 `exp_0NN/` 直下）を指す。**直下のコードだけを
外し**、`*_work_dir/` 配下の `.py`（実行時に保存された config の複製）は残す。コードは本環境が正本。

`--partial-dir=.rsync-partial` は必須。付けないと中断時に途中までのファイルが最終的な名前のまま
残り、**完全なファイルに見えてしまう**（2026-07-27 に 1.9GB の一時ファイルを git が巻き込んで
push が失敗した実例がある）。

## 2. 評価結果だけ先に取る（軽量・数十 MB）

数値だけ先に見たい場合はこれで足りる。ckpt を含まないので数分で終わる。

```bash
REMOTE=kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL/experiments
LOCAL=/workspace/kouyou/mmdetection/experiments

for e in exp_026 exp_027 exp_028 exp_029; do
  rsync -avh --progress \
    --include='*/' --include='*eval*/***' --exclude='*' \
    "$REMOTE/$e/" "$LOCAL/$e/"
done
```

exp_026 は評価ディレクトリ名が `indiv_*_eval` / `oracle_condA_eval`、exp_027〜029 は
`*_eval_after_*` なので、上の `*eval*` で両方拾える。

## 3. 学習ログ（scalars.json）だけ取る

学習曲線を見たいだけならこれで足りる。

```bash
REMOTE=kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL/experiments
LOCAL=/workspace/kouyou/mmdetection/experiments

for e in exp_026 exp_027 exp_028 exp_029; do
  rsync -avh --progress \
    --include='*/' --include='*.log' --include='scalars.json' --include='config.py' \
    --exclude='*' "$REMOTE/$e/" "$LOCAL/$e/"
done
```

## 4. 長時間になるので切断に強い形で

ckpt を含む転送は数時間かかる。

```bash
cd /workspace/kouyou/mmdetection
nohup bash -c '
REMOTE=kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL/experiments
LOCAL=/workspace/kouyou/mmdetection/experiments
for e in exp_026 exp_027 exp_028 exp_029; do
  rsync -avh --progress --partial --partial-dir=.rsync-partial \
    --exclude="/configs/" --exclude="/*.sh" --exclude="/*.py" --exclude="/*.md" \
    "$REMOTE/$e/" "$LOCAL/$e/"
done
' > experiments/retrieve_exp026_029.log 2>&1 &

tail -f experiments/retrieve_exp026_029.log
```

**中断したら同じコマンドを再実行すればよい。** 転送済みはスキップされ、途中のファイルは
`.rsync-partial` から続きを再開する。

---

## 5. 転送後の確認

### 5.1 個数の照合

```bash
for e in exp_026 exp_027 exp_028 exp_029; do
  echo -n "$e  クラスタ: "
  ssh kouyou@192.168.170.100 "find /home/kouyou/VLM_OD_CL/experiments/$e -name '*.pth' | wc -l"
  echo -n "$e  本環境  : "
  find /workspace/kouyou/mmdetection/experiments/$e -name '*.pth' | wc -l
done
du -sh /workspace/kouyou/mmdetection/experiments/exp_02{6,7,8,9}
```

完走していれば ckpt は exp_027 が 120 個（2条件 × 3ドメイン × 20 epoch）、
exp_028 が 240 個（4条件）、exp_029 が 240 個（4条件）。

### 5.2 途中ファイルが残っていないか

```bash
find /workspace/kouyou/mmdetection/experiments/exp_02{6,7,8,9} \
  -name '.*.pth.*' -o -name '.rsync-partial' -o -name '*.pth.*' | head
```

何か出たら転送が完了していない。**この状態で `git add` しない**（1.9GB の一時ファイルを
巻き込んで push が失敗した実例がある。`.gitignore` に `.*.pth.*` / `.*.pth` /
`.rsync-partial/` を追加済み）。

### 5.3 評価値が読めるか

```bash
grep -h "coco/bbox_mAP:" \
  /workspace/kouyou/mmdetection/experiments/exp_028/kdA_condA_l2_eval_after_videogames/*/*/*.log \
  | tail -4
```

---

## 6. 注意

- **`*_work_dir` 配下は編集しない**（禁止則）。rsync で上書きするのは構わないが、手で直さない。
- 転送方向は常に **クラスタ → 本環境**。逆向き（本環境 → クラスタ）にコードを送るのは
  `git push` / `git pull` で行う。
- exp_029 は kdE が未完の可能性がある。0.2 のジョブ確認で `exp029b_kdDE_condB_l2` が
  残っていないかを先に見る。
