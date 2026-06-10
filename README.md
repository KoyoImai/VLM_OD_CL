# VLM_OD_CL
事前学習済みVision Language Modelを用いた物体検出の継続学習

## 構築手順
事前学習済みVision Language ModelのMM-Grouding DINOをも対象に，学習と評価を動かす環境を整えることを目的とします．


### Dockerコンテナの作成
Dockerfileを使用してDockerイメージを作成します．
Dockerfileはこのリポジトリにあるものを使用しています．
以下のコマンドを実行してDickerイメージを作成してください．
```
docker build ./ --force-rm --no-cache -t mmdet-grounding:cuda12.8-torch2.1
```
上記コマンドを実行することで，イメージ名`mmdet-grounding`，タグ名`cuda12.8-torch2.1`としています．
Dockerイメージを作成後，Dockerコンテナを作成します．
以下のコマンドを実行してください．
```
docker run -it --gpus all --shm-size=128G -v /data1/kouyou:/workspace/kouyou -p 7979:7979 --name mmdet-grounding mmdet-grounding:cuda12.8-torch2.1
```
ホスト側の`/data1/kouyou`ディレクトリをコンテナ側の`/workspace/kouyou`ディレクトリにマウントします．
コンテナ名は`mmdet-grounding`としています．
 