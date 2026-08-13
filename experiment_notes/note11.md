# 実験11

## 位置付け
リプレイフリー，蒸留，リプレイの後半タスクの学習と評価

## 目的
exp_024（リプレイフリー），exp_023（リプレイ），exp_035（lambda=10の蒸留ありリプレイフルファインチューニング）の3タス目終了時のパラメータを初期値として，残りの3ドメインを学習する．


## 実験条件
- ドメイン順序：Aerial → Microscopic → Documents
- 初期パラメータ（リプレイフリー）：`experiments/exp_024/fullft_replayfree_videogames_work_dir/fullft_replayfree_videogames.py`
- 初期パラメータ（リプレイ）：`experiments/exp_027/fullft_replay_videogames_work_dir/epoch_20.pth`
- 初期パラメータ（λ=10蒸留+リプレイ）：``
- バッファデータ：`experiments/exp_023/buffer`



