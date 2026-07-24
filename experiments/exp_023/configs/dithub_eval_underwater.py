# =============================================================================
# exp_023 DitHub 評価用（逐次ライブラリ対応）/ underwater valid
#   逐次学習で育てたライブラリ checkpoint（全ドメインの per_class_lora_A）を読み、
#   過去ドメインも学習済み A を当てて評価するための config。
#   dithub_classes を全ドメインの和集合（152クラス）にしてスロットを全確保し、
#   ライブラリ ckpt の A を漏れなくロードする。評価時は DitHub のフィルタにより
#   プロンプト中のクラスのうちモジュールを持つものだけに A が当たる。
#   データ・num_classes・（vg の chunked）は eval_underwater.py を継承。既存コード無変更。
# =============================================================================
_base_ = './eval_underwater.py'

custom_imports = dict(
    imports=[
        'exp023_np_compat',
        'mmdet.models.layers.dithub_layers',
        'mmdet.models.detectors.dithub_grounding_dino',
    ],
    allow_failed_imports=False)

# 全ドメイン和集合（152クラス）。評価時 checkpointing は不要（encoder_cp=0）。
_dithub_all_classes = (
        'pipe', 'fish', 'jellyfish', 'penguin',
        'puffin', 'shark', 'starfish', 'stingray',
        'peix', 'taca', 'echinus', 'holothurian',
        'scallop', 'waterweeds', 'Arborescent', 'Caespitose-a',
        'Caespitose-b', 'Columnar', 'Corymbose', 'Digitate',
        'Encrusting', 'Foliose', 'Massive-Faviidae', 'Massive-Merulinidae',
        'Massive-Mussidae', 'Massive-Poritidae', 'Solitary', 'Tabular',
        'dog', 'person', 'Cell', 'Cell-Multi',
        'No-Anomaly', 'Shadowing', 'Unclassified', 'stray',
        'target', 'cheetah', 'human', 'artefact',
        'distal phalanges', 'fifth metacarpal bone', 'first metacarpal bone', 'fourth metacarpal bone',
        'intermediate phalanges', 'proximal phalanges', 'radius', 'second metacarpal bone',
        'soft tissue calcination', 'third metacarpal bone', 'ulna', 'acl',
        '0', 'negative', 'positive', '6W',
        '7W', 'EH', 'label0', 'label1',
        'label2', 'angle', 'fracture', 'line',
        'messed_up_angle', 'bicycle', 'car', 'avatar',
        'object', 'assassin', 'atv', 'gun',
        'gun menu', 'healthbar', 'horse', 'hud',
        'map', 'surroundings', 'CT', 'T',
        'Character', 'enemy', 'enemy-head', 'friendly',
        'friendly-head', 'Akali', 'Blitzcrank', 'Braum',
        'Caitlyn', 'Camille', 'Cho-Gath', 'Darius',
        'Dr- Mundo', 'Ekko', 'Ezreal', 'Fiora',
        'Galio', 'Gankplank', 'Garen', 'Graves',
        'Heimerdinger', 'Illaoi', 'Janna', 'Jayce',
        'Jhin', 'Jinx', 'Kai-Sa', 'Kassadin',
        'Katarina', 'Kog-Maw', 'Leona', 'Lissandra',
        'Lulu', 'Lux', 'Malzahar', 'Miss Fortune',
        'Orianna', 'Poppy', 'Quinn', 'Samira',
        'Seraphine', 'Shaco', 'Singed', 'Sion',
        'Swain', 'Tahm Kench', 'Talon', 'Taric',
        'Tristana', 'Trundle', 'Twisted Fate', 'Twitch',
        'Urgot', 'Veigar', 'Vex', 'Vi',
        'Viktor', 'Warwick', 'Yone', 'Yuumi',
        'Zac', 'Ziggs', 'Zilean', 'Zyra',
        'armor', 'base', 'rune', 'rune-blue',
        'rune-gray', 'rune-grey', 'rune-red', 'watcher',
    )

model = dict(
    type='DitHubGroundingDINO',
    dithub_classes=_dithub_all_classes,
    encoder=dict(num_cp=0),
    encoder_cp=0)
