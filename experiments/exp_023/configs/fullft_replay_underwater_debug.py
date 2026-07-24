# =============================================================================
# exp_023 デバッグ用: full finetuning + リプレイ / underwater を 1 epoch だけ実行。
#   目的: 学習パイプラインが端から端まで（分散・混合サンプラー・forward/backward・
#   optimizer・checkpoint 保存）正常に回るかの動作確認。既存ファイルは無変更。
# =============================================================================
_base_ = './fullft_replay_underwater.py'

# 1 epoch だけ回す（val は行わない）。ログを頻繁に出す。
train_cfg = dict(max_epochs=1, val_interval=2)
default_hooks = dict(logger=dict(type='LoggerHook', interval=20))
