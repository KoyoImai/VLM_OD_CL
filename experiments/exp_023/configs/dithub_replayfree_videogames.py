# =============================================================================
# exp_023 比較手法: DitHub（リプレイフリー・論文準拠）/ videogames（逐次 t=3）
#   dithub_replayfree_underwater.py と同一（dithub_classes とデータのみ差し替え）。
#   87 クラスだが学習は RandomSamplingNegPos（max_tokens=256）で収まる。
#   全クラス一括評価は eval_videogames.py（chunked_size）で対処。既存コード無変更。
# =============================================================================
_base_ = './dithub_replayfree_base.py'

# t=3: 現ドメイン(videogames)∩過去(underwater,electromagnetic) = {person, car}。
# specialization 切替時に式3 fetch+merge を効かせる（案Y・公式忠実）。
custom_hooks = [
    dict(
        type='DitHubSeqPhaseHook',
        warmup_iters=1500,
        trained_classes=['person', 'car'])
]

# クラス別 LoRA 用のクラス名（videogames_label_map.json の index 順）
model = dict(
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
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=current_videogames)
