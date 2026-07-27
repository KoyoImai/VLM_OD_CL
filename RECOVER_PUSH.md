# git push 失敗からの復旧手順（2026-07-27）

## 何が起きたか

`git add -A` が **rsync の転送中一時ファイル**を巻き込んだ。

```
experiments/exp_027/condB_replay_videogames_work_dir/.epoch_12.pth.NNQjeZ   1917.2 MB
```

rsync は転送中のファイルを `.<元の名前>.<ランダム6文字>` という名前で作る。`.gitignore` には
`*.pth` があるが、この名前は **`.pth` で終わらない**ため無視されず、コミットに入った。
GitHub は 100MB を超えるファイルを拒否するため push が弾かれた。

## 現状

- コミット `84354b0 backup` は**ローカルのみ**（`origin/main` より1コミット先行、push は未完了）。
- 100MB を超えるファイルはこの1つだけ。
- 当該ファイルは rsync が転送を終えて `epoch_12.pth` にリネームしたため、**作業ツリーには既に存在しない**。
- **rsync はまだ実行中**なので、別の一時ファイルが再び現れる可能性がある。

リモートには何も送られていないので、履歴を書き換えても他への影響はない。

---

## 復旧手順

### 1. 一時ファイルをコミットから外す

```bash
cd /workspace/kouyou/mmdetection
git rm --cached --ignore-unmatch "experiments/exp_027/condB_replay_videogames_work_dir/.epoch_12.pth.NNQjeZ"
git commit --amend --no-edit
```

`--amend` で `84354b0` を作り直す。push していないので安全。

### 2. 100MB 超のファイルが残っていないか確認

```bash
git ls-tree -r -l HEAD | awk '$4 > 100000000 {printf "%10.1f MB  %s\n", $4/1048576, $5}'
```

**何も出力されなければ OK。**

### 3. 再発防止（.gitignore に追記）

```bash
cat >> .gitignore <<'EOF'

# rsync 転送中の一時ファイル（.<元のファイル名>.<ランダム6文字>）
# *.pth では拾えないため個別に指定する（2026-07-27）
.*.pth.*
.*.pth
.rsync-partial/
EOF
git add .gitignore
git commit -m "gitignore: rsync の一時ファイルを除外"
```

効いているかの確認：

```bash
printf '%s\n' ".epoch_12.pth.NNQjeZ" | git check-ignore --stdin --no-index -v
```

行が出力されれば無視されている。

### 4. push

```bash
git push
```

---

## 補足：ディスクの回収（任意）

外した 1.9GB のオブジェクトは `.git` に残る（現在 `.git` は 4.8GB）。不要なら回収できる。

```bash
git reflog expire --expire-unreachable=now --all
git gc --prune=now
du -sh .git
```

**注意**: これは到達不能なオブジェクトを完全に削除する。上の手順1〜4を終えて push が成功したことを
確認してから実行すること。

---

## 今後の注意

**rsync 実行中に `git add -A` を使わない。** 転送中の一時ファイルを巻き込む。
手順3の `.gitignore` で今回の形は防げるが、確実なのは rsync の完了を待ってからコミットすること。

rsync の進行状況：

```bash
ps aux | grep [r]sync
tail -f /workspace/kouyou/mmdetection/experiments/retrieve_from_cluster.log
```
