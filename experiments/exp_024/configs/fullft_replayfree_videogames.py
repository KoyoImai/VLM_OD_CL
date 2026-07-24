# =============================================================================
# exp_024: リプレイ無し逐次FT（条件A＝全モジュール）/ videogames（逐次 t=3）
#   逐次順 underwater -> electromagnetic -> videogames の3番目（最終）。
#   重みは前ドメイン（electromagnetic）の last をドライバが load_from で渡す。
#   リプレイ無しのため学習データは現在ドメインのみ。batch_size=4/GPU。
#
#   注: videogames は 87 クラスだが、負例サンプリング（max_tokens=256）により
#   キャプションは常に 256 トークン以内に収まるため、finetune で使っていた
#   max_text_len=512 への引き上げは不要（contrastive_cfg も 256 のままで整合）。
# =============================================================================
_base_ = './fullft_replayfree_base.py'

train_pipeline = _base_.train_pipeline

_vg_root = '/workspace/kouyou/datasets/rf100_domain/videogames/'

# 現在ドメイン（videogames、OD-ODVG）
current_videogames = dict(
    type='ODVGDataset',
    data_root=_vg_root,
    ann_file='videogames_train_od.json',
    label_map_file='videogames_label_map.json',
    data_prefix=dict(img='train/'),
    filter_cfg=dict(filter_empty_gt=False),
    return_classes=True,
    pipeline=train_pipeline)

train_dataloader = dict(
    _delete_=True,
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=current_videogames)
