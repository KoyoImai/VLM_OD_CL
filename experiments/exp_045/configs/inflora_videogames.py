# =============================================================================
# exp_045: InfLoRA（232 層・r=16） / videogames（逐次 t=3）
#
# 【自動生成】experiments/exp_045/gen_configs.py。直接編集しない。
#
# データ・スケジュール（バッファ不使用・batch 4/GPU・DefaultSampler・ODVG・dn 有効）は
# ../../exp_024/configs/fullft_replayfree_videogames.py を継承し、
# 本 config が上書きするのはモデルだけ（design.md §2.1）。
# 逐次学習では前タスクの θ_{t-1} を load_from に上書きする（ドライバが実施）。
# =============================================================================
_base_ = '../../exp_024/configs/fullft_replayfree_videogames.py'

custom_imports = dict(
    imports=[
        'exp023_np_compat',
        'projects.lora_cl',
    ],
    allow_failed_imports=False)

# InfLoRA（design.md §2.3）: 挿入 232 層（Swin・BERT・feature enhancer・text_feat_map）、
# r=16 / alpha=16（scaling=1 = 公式の ΔW=B·A）。encoder テキスト側 self-attention は
# MHA を q/k/v/o に分解して挿入する。design_path（設計済み A）はドライバが
# --cfg-options model.inflora.design_path= で渡す。dn は RF100 枠に従い有効のまま。
model = dict(
    type='GroundingDINOInfLoRA',
    lora=dict(
        r=16,
        alpha=16,
        exclude_components=[],
        decompose_mha=[r'^encoder\.text_layers\.\d+\.self_attn\.attn$'],
        include=['backbone', 'language_model', 'text_feat_map', 'encoder']),
    inflora=dict(design_path=None))
