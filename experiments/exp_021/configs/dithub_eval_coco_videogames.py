# exp_021: DitHub checkpoint (videogames) 用の ZCOCO 評価 config
# per_class_lora_A のキー集合を学習時と一致させるため、ドメインごとに
# クラス一覧を静的に埋め込む (生成元: 学習 config の class_name)。
_base_ = '../../../configs/mm_grounding_dino/eval_base_coco.py'

custom_imports = dict(
    imports=[
        'mmdet.models.layers.dithub_layers',
        'mmdet.models.detectors.dithub_grounding_dino',
        'mmdet.engine.hooks.dithub_phase_hook',
    ],
    allow_failed_imports=False)

model = dict(
    type='DitHubGroundingDINO',
    dithub_classes=('avatar', 'object', 'assassin', 'atv', 'car', 'gun', 'gun menu', 'healthbar', 'horse', 'hud', 'map', 'person', 'surroundings', 'CT', 'T', 'Character', 'enemy', 'enemy-head', 'friendly', 'friendly-head', 'Akali', 'Blitzcrank', 'Braum', 'Caitlyn', 'Camille', 'Cho-Gath', 'Darius', 'Dr- Mundo', 'Ekko', 'Ezreal', 'Fiora', 'Galio', 'Gankplank', 'Garen', 'Graves', 'Heimerdinger', 'Illaoi', 'Janna', 'Jayce', 'Jhin', 'Jinx', 'Kai-Sa', 'Kassadin', 'Katarina', 'Kog-Maw', 'Leona', 'Lissandra', 'Lulu', 'Lux', 'Malzahar', 'Miss Fortune', 'Orianna', 'Poppy', 'Quinn', 'Samira', 'Seraphine', 'Shaco', 'Singed', 'Sion', 'Swain', 'Tahm Kench', 'Talon', 'Taric', 'Tristana', 'Trundle', 'Twisted Fate', 'Twitch', 'Urgot', 'Veigar', 'Vex', 'Vi', 'Viktor', 'Warwick', 'Yone', 'Yuumi', 'Zac', 'Ziggs', 'Zilean', 'Zyra', 'armor', 'base', 'rune', 'rune-blue', 'rune-gray', 'rune-grey', 'rune-red', 'watcher'))
