# exp_021: DitHub checkpoint (documents) 用の ZCOCO 評価 config
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
    dithub_classes=('caption', 'tweet', 'profile_info', 'table', 'title', 'action', 'activity', 'commeent', 'control_flow', 'control_flowcontrol_flow', 'decision_node', 'exit_node', 'final_flow_node', 'final_node', 'fork', 'merge', 'merge_noode', 'null', 'object', 'object_flow', 'signal_recept', 'signal_send', 'start_node', 'text', 'signature', 'author', 'chapter', 'equation', 'equation number', 'figure', 'figure caption', 'footnote', 'list of content heading', 'list of content text', 'page number', 'paragraph', 'reference text', 'section', 'subsection', 'subsubsection', 'table caption', 'table of contents text', '-', 'bold_parent_row', 'bold_row', 'closure_row', 'column', 'direct_children', 'non_bold_parent_row', 'non_bold_row', 'parent_column', 'prime_parent', 'sub_row', 'g', 'g1', 'g3', 'h', 'm', 'n'))
