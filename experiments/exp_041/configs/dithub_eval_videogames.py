# =============================================================================
# exp_041 DitHub 評価用（6ドメイン逐次ライブラリ対応）/ videogames
#
# 【自動生成】experiments/exp_041/gen_configs.py。直接編集しない。
#
#   逐次学習で育てたライブラリ ckpt（per_class_lora_A）を漏れなくロードするため、
#   dithub_classes を**6ドメイン全クラスの和集合（260 クラス）**にしてスロットを全確保する。
#   既存の exp_023 dithub_eval_*（152 クラス）は前半3ドメインの和集合しか持たず、
#   t>=4 のライブラリの後半クラス A を捨ててしまう（design.md §4）。
#   評価時は DitHub のフィルタにより、プロンプト中のクラスのうちモジュールを持つもの
#   だけに A が当たるため、和集合の拡大は評価値を変えない（check_exp041_setup.py の
#   項目7で実測確認）。データ・num_classes は exp_023 の eval_videogames.py を継承。既存コード無変更。
# =============================================================================
_base_ = '../../exp_023/configs/eval_videogames.py'

custom_imports = dict(
    imports=[
        'exp023_np_compat',
        'mmdet.models.layers.dithub_layers',
        'mmdet.models.detectors.dithub_grounding_dino',
    ],
    allow_failed_imports=False)

# 6ドメイン和集合（260 クラス）。評価時 checkpointing は不要（encoder_cp=0）。
_dithub_all_classes = ('pipe', 'fish', 'jellyfish', 'penguin', 'puffin', 'shark',
        'starfish', 'stingray', 'peix', 'taca', 'echinus', 'holothurian',
        'scallop', 'waterweeds', 'Arborescent', 'Caespitose-a',
        'Caespitose-b', 'Columnar', 'Corymbose', 'Digitate',
        'Encrusting', 'Foliose', 'Massive-Faviidae',
        'Massive-Merulinidae', 'Massive-Mussidae', 'Massive-Poritidae',
        'Solitary', 'Tabular', 'dog', 'person', 'Cell', 'Cell-Multi',
        'No-Anomaly', 'Shadowing', 'Unclassified', 'stray', 'target',
        'cheetah', 'human', 'artefact', 'distal phalanges',
        'fifth metacarpal bone', 'first metacarpal bone',
        'fourth metacarpal bone', 'intermediate phalanges',
        'proximal phalanges', 'radius', 'second metacarpal bone',
        'soft tissue calcination', 'third metacarpal bone', 'ulna',
        'acl', '0', 'negative', 'positive', '6W', '7W', 'EH', 'label0',
        'label1', 'label2', 'angle', 'fracture', 'line',
        'messed_up_angle', 'bicycle', 'car', 'avatar', 'object',
        'assassin', 'atv', 'gun', 'gun menu', 'healthbar', 'horse',
        'hud', 'map', 'surroundings', 'CT', 'T', 'Character', 'enemy',
        'enemy-head', 'friendly', 'friendly-head', 'Akali', 'Blitzcrank',
        'Braum', 'Caitlyn', 'Camille', 'Cho-Gath', 'Darius', 'Dr- Mundo',
        'Ekko', 'Ezreal', 'Fiora', 'Galio', 'Gankplank', 'Garen',
        'Graves', 'Heimerdinger', 'Illaoi', 'Janna', 'Jayce', 'Jhin',
        'Jinx', 'Kai-Sa', 'Kassadin', 'Katarina', 'Kog-Maw', 'Leona',
        'Lissandra', 'Lulu', 'Lux', 'Malzahar', 'Miss Fortune',
        'Orianna', 'Poppy', 'Quinn', 'Samira', 'Seraphine', 'Shaco',
        'Singed', 'Sion', 'Swain', 'Tahm Kench', 'Talon', 'Taric',
        'Tristana', 'Trundle', 'Twisted Fate', 'Twitch', 'Urgot',
        'Veigar', 'Vex', 'Vi', 'Viktor', 'Warwick', 'Yone', 'Yuumi',
        'Zac', 'Ziggs', 'Zilean', 'Zyra', 'armor', 'base', 'rune',
        'rune-blue', 'rune-gray', 'rune-grey', 'rune-red', 'watcher',
        'black-hat', 'bodysurface', 'bodyunder', 'umpire', 'white-hat',
        'chain', 'green_sphero', 'orange-sphero', 'orange_sphero',
        'purple_sphero', 'red_sphero', 'yellow_sphero', 'football',
        'player', 'referee', 'crop', 'weed', 'cow', 'Fish', 'Flower',
        'Gravel', 'Sugar', 'close', 'open', 'Platelets', 'RBC', 'WBC',
        'Ancylostoma Spp', 'Ascaris Lumbricoides',
        'Enterobius Vermicularis', 'Fasciola Hepatica', 'Hymenolepis',
        'Schistosoma', 'Taenia Sp', 'Trichuris Trichiura', 'celula',
        '4-fold defect', 'Str_pne', 'dc', 'Mitosis', 'activated',
        'non-activated', 'ballooning', 'fibrosis', 'inflammation',
        'steatosis', 'thick-dark-mark', 'thick-light-mark',
        'thin-dark-mark', 'thin-light-mark', 'caption', 'tweet',
        'profile_info', 'table', 'title', 'action', 'activity',
        'commeent', 'control_flow', 'control_flowcontrol_flow',
        'decision_node', 'exit_node', 'final_flow_node', 'final_node',
        'fork', 'merge', 'merge_noode', 'null', 'object_flow',
        'signal_recept', 'signal_send', 'start_node', 'text',
        'signature', 'author', 'chapter', 'equation', 'equation number',
        'figure', 'figure caption', 'footnote',
        'list of content heading', 'list of content text', 'page number',
        'paragraph', 'reference text', 'section', 'subsection',
        'subsubsection', 'table caption', 'table of contents text', '-',
        'bold_parent_row', 'bold_row', 'closure_row', 'column',
        'direct_children', 'non_bold_parent_row', 'non_bold_row',
        'parent_column', 'prime_parent', 'sub_row', 'g', 'g1', 'g3', 'h',
        'm', 'n')

model = dict(
    type='DitHubGroundingDINO',
    dithub_classes=_dithub_all_classes,
    encoder=dict(num_cp=0),
    encoder_cp=0)
