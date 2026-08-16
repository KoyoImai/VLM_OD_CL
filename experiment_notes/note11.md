# 実験11

## 位置付け
リプレイフリー，蒸留，リプレイの後半タスクの学習と評価

## 目的
exp_024（リプレイフリー），exp_027（リプレイ），exp_035（lambda=10の蒸留E+リプレイフルファインチューニング）の3タス目終了時のパラメータを初期値として，残りの3ドメインを学習する．


## 実験条件
- ドメイン順序：Aerial → Microscopic → Documents
- 初期パラメータ（リプレイフリー）：`experiments/exp_024/fullft_replayfree_videogames_work_dir/epoch_20.pth`
- 初期パラメータ（リプレイ）：`experiments/exp_027/fullft_replay_videogames_work_dir/epoch_20.pth`
- 初期パラメータ（λ=10蒸留E+リプレイ）：`experiments/exp_035/kdE_condA_l2w100_videogames_work_dir/epoch_20.pth`
- 初期パラメータ（ZiRa）：`experiments/exp_039/zira_replayfree_merged/merged_after_t3_videogames.pth`
- 初期パラメータ（ZiRa+リプレイ）：`experiments/exp_039/zira_replay_merged/merged_after_t3_videogames.pth`
- 初期パラメータ（DitHub）：`experiments/exp_039/dithub_replayfree_merged/merged_after_t3_videogames.pth`
- 初期パラメータ（DitHub+リプレイ）：`experiments/exp_039/dithub_replay_merged/merged_after_t3_videogames.pth`
- バッファデータ：`experiments/exp_023/buffer`


## 結果

**t=4: aerial 学習後**
|                    | underwater | electromagnetic | videogames | aerial |  zcoco |
|--------------------|------------|-----------------|------------|--------|--------|
| zero-shot          |            |                 |            |        |        |
| replay free        | 0.052      | 0.139           | 0.248      | 0.480  | 0.059  |
| replay             | 0.235      | 0.406           | 0.631      | 0.475  | 0.365  |
| distill E          | 0.251      | 0.423           | 0.663      | 0.474  | 0.376  |
| ZiRa               | 0.093      | 0.124           | 0.082      | 0.338  | 0.417  |
| ZiRa+replay        | 0.174      | 0.165           | 0.077      | 0.334  | 0.483  |
| DitHub             | 0.023      | 0.197           | 0.119      | 0.347  | 0.418  |
| DitHub+replay      | 0.032      | 0.146           | 0.090      | 0.327  | 0.482  |



**t=5: microscopic 学習後**
|                    | underwater | electromagnetic | videogames | aerial | microscopic |  zcoco |
|--------------------|------------|-----------------|------------|--------|-------------|--------|
| zero-shot          |            |                 |            |        |             |        |
| replay free        | 0.043      | 0.005           | 0.053      | 0.120  | 0.533       | 0.065  |
| replay             | 0.224      | 0.378           | 0.558      | 0.457  | 0.521       | 0.352  |
| distill E          | 0.239      | 0.368           | 0.631      | 0.451  | 0.521       | 0.370  |
| ZiRa               | 0.092      | 0.078           | 0.052      | 0.266  | 0.270       | 0.391  |
| ZiRa+replay        | 0.165      | 0.141           | 0.069      | 0.311  | 0.278       | 0.480  |
| DitHub             | 0.007      | 0.163           | 0.116      | 0.357  | 0.257       | 0.367  |
| DitHub+replay      |            |                 |            |        |             |        |


**t=6: documents 学習後**
|                    | underwater | electromagnetic | videogames | aerial | microscopic | documents | zcoco |
|--------------------|------------|-----------------|------------|--------|-------------|-----------|--------|
| zero-shot          |            |                 |            |        |             |           |        |
| replay free        | 0.015      | 0.003           | 0.001      | 0.023  | 0.150       | 0.533     | 0.018  |
| replay             | 0.208      | 0.344           | 0.534      | 0.436  | 0.457       | 0.503     | 0.342  |
| distill E          | 0.239      | 0.351           | 0.610      | 0.446  | 0.458       | 0.502     | 0.356  |
| ZiRa               | 0.071      | 0.035           | 0.025      | 0.183  | 0.160       | 0.191     | 0.321  |
| ZiRa+replay        |            |                 |            |        |             |           |        |
| DitHub             | 0.001      | 0.126           | 0.103      | 0.330  | 0.242       | 0.219     | 0.350  |
| DitHub+replay      |            |                 |            |        |             |           |        |




## 備考
### 実験結果
- リプレイフリー：exp_040
- リプレイ：exp_040
- 蒸留：exp_040
- ZiRa：exp_041
- ZiRa+リプレイ：exp_041
- DitHub：exp_041
- DitHub+リプレイ：exp_041

