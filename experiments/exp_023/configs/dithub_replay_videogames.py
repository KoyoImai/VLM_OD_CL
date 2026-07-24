# =============================================================================
# exp_023 比較手法: DitHub（リプレイ有り・論文準拠）/ videogames（逐次 t=3・最終）
#   dithub_replay_underwater.py と同一方針（データと dithub_classes を差し替え）。
#   過去2ドメインを入れ子 ConcatDataset で1つの「過去プール」source に束ねる。
#   [現在, 参照, 過去プール] の3ソース、source_ratio [4,1,1]、batch_size=6。
#   load_from はドライバが前ドメイン(electromagnetic)の last に差し替える。既存コード無変更。
# =============================================================================
_base_ = './dithub_replayfree_base.py'

custom_imports = dict(
    imports=[
        'exp023_np_compat',
        'mmdet.models.layers.dithub_layers',
        'mmdet.models.detectors.dithub_grounding_dino',
        'mmdet.engine.hooks.dithub_phase_hook',
        'mmdet.engine.hooks.dithub_seq_phase_hook',
        'mmdet.models.detectors.dithub_replay_grounding_dino',
    ],
    allow_failed_imports=False)

# t=3: 現ドメイン(videogames)∩過去(underwater,electromagnetic) = {person, car}。
# specialization 切替時に式3 fetch+merge を効かせる（案Y・公式忠実）。
custom_hooks = [
    dict(
        type='DitHubSeqPhaseHook',
        warmup_iters=1500,
        trained_classes=['person', 'car'])
]

model = dict(
    type='DitHubReplayGroundingDINO',
    dithub_classes=(
        'avatar', 'object', 'assassin', 'atv', 'car', 'gun', 'gun menu',
        'healthbar', 'horse', 'hud', 'map', 'person', 'surroundings', 'CT',
        'T', 'Character', 'enemy', 'enemy-head', 'friendly', 'friendly-head',
        'Akali', 'Blitzcrank', 'Braum', 'Caitlyn', 'Camille', 'Cho-Gath',
        'Darius', 'Dr- Mundo', 'Ekko', 'Ezreal', 'Fiora', 'Galio',
        'Gankplank', 'Garen', 'Graves', 'Heimerdinger', 'Illaoi', 'Janna',
        'Jayce', 'Jhin', 'Jinx', 'Kai-Sa', 'Kassadin', 'Katarina', 'Kog-Maw',
        'Leona', 'Lissandra', 'Lulu', 'Lux', 'Malzahar', 'Miss Fortune',
        'Orianna', 'Poppy', 'Quinn', 'Samira', 'Seraphine', 'Shaco', 'Singed',
        'Sion', 'Swain', 'Tahm Kench', 'Talon', 'Taric', 'Tristana', 'Trundle',
        'Twisted Fate', 'Twitch', 'Urgot', 'Veigar', 'Vex', 'Vi', 'Viktor',
        'Warwick', 'Yone', 'Yuumi', 'Zac', 'Ziggs', 'Zilean', 'Zyra', 'armor',
        'base', 'rune', 'rune-blue', 'rune-gray', 'rune-grey', 'rune-red',
        'watcher'))

train_pipeline = _base_.train_pipeline

_vg_root = '/workspace/kouyou/datasets/rf100_domain/videogames/'
_uw_root = '/workspace/kouyou/datasets/rf100_domain/underwater/'
_em_root = '/workspace/kouyou/datasets/rf100_domain/electromagnetic/'
_o365_root = '/workspace/kouyou/datasets/o365v1_stage/Objects365_v1/2019-08-02/'
_buf = '/workspace/kouyou/mmdetection/experiments/exp_023/buffer/'

current_videogames = dict(
    type='ODVGDataset',
    data_root=_vg_root,
    ann_file='videogames_train_od.json',
    label_map_file='videogames_label_map.json',
    data_prefix=dict(img='train/'),
    filter_cfg=dict(filter_empty_gt=False),
    return_classes=True,
    pipeline=train_pipeline)

reference_buffer = dict(
    type='ODVGDataset',
    data_root=_o365_root,
    ann_file=_buf + 'reference_o365v1_1000.odvg.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img='train/'),
    filter_cfg=dict(filter_empty_gt=False),
    return_classes=True,
    pipeline=train_pipeline)

past_pool = dict(
    type='ConcatDataset',
    datasets=[
        dict(
            type='ODVGDataset',
            data_root=_uw_root,
            ann_file=_buf + 'underwater_500.odvg.json',
            label_map_file='underwater_label_map.json',
            data_prefix=dict(img='train/'),
            filter_cfg=dict(filter_empty_gt=False),
            return_classes=True,
            pipeline=train_pipeline),
        dict(
            type='ODVGDataset',
            data_root=_em_root,
            ann_file=_buf + 'electromagnetic_500.odvg.json',
            label_map_file='electromagnetic_label_map.json',
            data_prefix=dict(img='train/'),
            filter_cfg=dict(filter_empty_gt=False),
            return_classes=True,
            pipeline=train_pipeline),
    ])

train_dataloader = dict(
    _delete_=True,
    batch_size=6,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(
        type='MultiSourceSampler', batch_size=6, source_ratio=[4, 1, 1]),
    dataset=dict(
        type='ConcatDataset',
        ignore_keys=['cumulative_sizes'],
        datasets=[current_videogames, reference_buffer, past_pool]))
