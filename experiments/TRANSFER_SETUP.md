# クラスタ → 本環境 の転送方法

## 調べた事実（2026-07-26）

| 項目 | 結果 |
|---|---|
| 本環境の sshd | **動いていない**（22番ポートで待ち受けなし） |
| 本環境の IP | **172.17.0.2**（Docker のブリッジ内アドレス） |
| 本環境 → クラスタ の到達性 | **TCP は到達する**（SSH ハンドシェイクまで進む） |
| 本環境 → クラスタ の認証 | **失敗**（`Permission denied (publickey,password)`） |
| 本環境の SSH 鍵 | `~/.ssh/id_ed25519` が**存在する**（クラスタ側に未登録） |

**結論**: クラスタから本環境へ直接 push することは、現状ではできない。
本環境は Docker コンテナ内で sshd が無く、IP も Docker 内部の 172.17.0.2 なので、
クラスタ（192.168.170.x）からは到達できない。

---

## 選択肢A：本環境から pull する（推奨・すぐ使える）

鍵をクラスタに登録するだけで動く。転送の向きは「本環境が取りに行く」。

### A-1. 本環境の公開鍵をクラスタに登録する

本環境の公開鍵は以下。

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIB9lkuJOcuWnd2wixz3eRphLkGRZplZVFXXUCin9SZZD imaikoyo0314@gmail.com
```

クラスタのマスターノードで実行する。

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIB9lkuJOcuWnd2wixz3eRphLkGRZplZVFXXUCin9SZZD imaikoyo0314@gmail.com' >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

### A-2. 本環境で疎通を確認する

```bash
ssh -o StrictHostKeyChecking=accept-new kouyou@192.168.170.100 'hostname; echo OK'
```

`OK` が出れば、[RETRIEVE_FROM_CLUSTER.md](./RETRIEVE_FROM_CLUSTER.md) の rsync がそのまま使える。

### A-3. 自動で取りに行く（任意）

ジョブの完了を待って自動転送したい場合は、本環境側でポーリングする。

```bash
cd /workspace/kouyou/mmdetection
nohup bash -c '
REMOTE=kouyou@192.168.170.100
LOCAL=/workspace/kouyou/mmdetection/experiments
while true; do
  # 実行中・待機中のジョブが無くなったら転送して終了
  n=$(ssh "$REMOTE" "squeue -u kouyou -h | wc -l")
  echo "$(date +%F\ %T)  残ジョブ: $n"
  if [ "$n" -eq 0 ]; then
    for e in exp_024 exp_025 exp_026 exp_027; do
      rsync -avh --progress --partial \
        --include="*/" --include="*_work_dir/***" --include="*eval*/***" --exclude="*" \
        "$REMOTE:/home/kouyou/VLM_OD_CL/experiments/$e/" "$LOCAL/$e/"
    done
    echo "転送完了"
    break
  fi
  sleep 1800
done
' > experiments/auto_retrieve.log 2>&1 &
```

30分ごとにジョブ数を確認し、全ジョブが終わったら転送して終了する。

---

## 選択肢B：クラスタから push する

ご希望の向き。実現するには本環境側の準備が要る。

### 必要なもの

1. **本環境（またはそのホストマシン）がクラスタから到達できる IP を持つこと**
   現在の 172.17.0.2 は Docker 内部アドレスなので不可。Docker ホスト側のラボ内 LAN アドレスが必要。
2. **sshd が動いていること**
   コンテナ内には無い。Docker ホストで動いているなら、そちらを転送先にする。
3. **クラスタ → 本環境 の SSH 鍵**
   クラスタ側で鍵を作り、本環境（またはホスト）の `authorized_keys` に登録する。
4. **ホスト側のパスと `/workspace/kouyou` の対応**
   `/workspace/kouyou` は `/dev/sda1`（7.3T）のマウント。ホスト上でこのディスクがどこに
   マウントされているかを指定する必要がある。

### 実現した場合の使い方

ジョブの最後に転送を挟む形にできる。`sbatch.sh` の末尾に以下を追加する。

```bash
# 学習・評価が終わったら本環境へ push
rsync -avh --partial \
  --include='*/' --include='*_work_dir/***' --include='*eval*/***' --exclude='*' \
  /home/kouyou/VLM_OD_CL/experiments/exp_027/ \
  <ユーザー>@<本環境ホストのIP>:<ホスト側のパス>/experiments/exp_027/
```

これならジョブ完了と同時に結果が本環境へ届く。

---

## 判断していただきたいこと

上記1〜4を満たせるか（特に **Docker ホストのラボ内 IP と sshd の有無**）。

- 満たせる → 選択肢B。`sbatch.sh` に転送処理を組み込む
- 満たせない／手間が大きい → 選択肢A。鍵の登録だけで今日から使える。A-3 を使えば
  「ジョブ完了後に自動で届く」という結果は B と同じになる
