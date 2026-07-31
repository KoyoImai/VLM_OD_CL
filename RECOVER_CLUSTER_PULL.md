# クラスタ側 `git pull` が中断した件の復旧手順（2026-07-27）

## 何が起きたか

```
error: The following untracked working tree files would be overwritten by merge:
        experiments/exp_024/eval_after_electromagnetic/.../20260725_121158.json
        ...
Aborting
```

**クラスタ側でジョブが生成した出力ファイル**（`vis_data/*.json`、`config.py`、`last_checkpoint` など）が
未追跡のまま存在している。一方、これらは本環境へ rsync で回収したあと `git add -A` で
コミットされ、リモートに入っている。git は未追跡ファイルを黙って上書きしないので中断した。

**データが壊れたわけではない。**どちらも同じジョブが生んだ同一内容のはずだが、
確認せずに消すのは避ける。

---

## 復旧手順（クラスタのマスターノードで実行）

### 1. 衝突するファイルを列挙する

```bash
cd /home/kouyou/VLM_OD_CL
git fetch origin

git ls-files --others --exclude-standard -z \
  | while IFS= read -r -d '' f; do
      git cat-file -e "origin/main:$f" 2>/dev/null && printf '%s\0' "$f"
    done > /tmp/conflicts.z

tr -d -c '\0' < /tmp/conflicts.z | wc -c   # 件数
```

### 2. 内容が同一かを確認する（重要）

```bash
: > /tmp/diff_files.txt
while IFS= read -r -d '' f; do
  git show "origin/main:$f" | cmp -s - "$f" || echo "$f" >> /tmp/diff_files.txt
done < /tmp/conflicts.z

wc -l /tmp/diff_files.txt
```

**`0` なら、すべて同一内容。**手順3へ進む。
**`0` でないなら、手順5へ。**

### 3. 念のためバックアップを取る

```bash
tar --null -czf ~/untracked_backup_$(date +%Y%m%d_%H%M%S).tar.gz -T /tmp/conflicts.z
ls -lh ~/untracked_backup_*.tar.gz
```

### 4. 衝突ファイルを削除して pull

```bash
xargs -0 rm -f < /tmp/conflicts.z
git pull
git log --oneline -1
```

pull が通れば復旧完了。手順3のバックアップは、内容を確認したうえで削除してよい。

---

## 5. 内容が異なるファイルがあった場合

クラスタ側のほうが新しい（rsync 後に生成された）可能性がある。**消さずに中身を比べる。**

```bash
head -20 /tmp/diff_files.txt
f=$(head -1 /tmp/diff_files.txt)
diff <(git show "origin/main:$f") "$f" | head -40
```

判断の目安。

| 状況 | 対処 |
|---|---|
| クラスタ側が新しい（rsync 後に走ったジョブの出力） | 退避してから pull し、**pull 後に戻す**（`tar` で退避 → `git pull` → 展開） |
| リモート側が新しい | クラスタ側を消して pull（手順4） |
| 判断がつかない | 両方を退避したうえで報告してください |

なお `last_checkpoint` は**絶対パスを保存している**が、クラスタのジョブは Singularity 内で
リポジトリを `/workspace/kouyou/mmdetection` に bind しており、本環境のパスと一致するため
内容は同じになるはずである。もしここに差分が出た場合は、パスの前提が崩れている合図なので
報告してほしい。

---

## 再発防止（判断が必要）

この衝突は、**実験の出力ファイルをリポジトリにコミットしている**限り繰り返し起きる。
クラスタが出力を生み、本環境が rsync で回収してコミットし、それをクラスタが pull しようとして
自分が生んだファイルとぶつかる、という循環になっている。

選択肢は2つ。

### 案1: 出力をコミットしない（推奨）

2026-07-26 に「クラスタ→本環境の結果転送は rsync のみ、git は使わない」と決めている。
その方針を徹底し、`.gitignore` に次を足すと衝突は起きなくなる。

```
# 実験の出力（結果の受け渡しは rsync で行う。2026-07-26 の方針）
experiments/*/*_work_dir/
experiments/*/*eval*/
```

ただし **exp_020〜exp_027 の出力は既にコミット済み**なので、それらは追跡され続ける
（`git rm --cached` で外すかどうかは別途判断）。

### 案2: 現状のまま、pull のたびに本手順を実行する

出力をリポジトリに残したい場合はこちら。衝突は毎回起きるが、手順1〜4で解消できる。

**どちらにするかは判断が必要。** 案1に進む場合は `.gitignore` の追記と、既存の追跡分を
どう扱うかを決める必要がある。
