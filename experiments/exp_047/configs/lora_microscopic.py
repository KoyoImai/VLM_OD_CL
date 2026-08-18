# =============================================================================
# exp_047: 素の LoRA（232 層・r=16・alpha=16）/ microscopic（逐次 t=5）
#
# 【自動生成】experiments/exp_047/gen_configs.py。直接編集しない。
#
# データ・スケジュール（バッファ不使用・batch 4/GPU・DefaultSampler・ODVG・dn 有効）は
# ../../exp_040/configs/replayfree_microscopic.py を継承し、
# 本 config が上書きするのはモデルだけ（design.md §2.1）。LoRA 設定は exp_045 の
# InfLoRA と完全に同一（差分は A の扱いのみ）。
# 逐次学習では前タスクのマージ済み θ_{t-1} を load_from に上書きする（ドライバが実施。
# LoRA は毎タスク B=0 から引き直すので開始時点は θ_{t-1} と厳密に等価）。
# =============================================================================
_base_ = '../../exp_040/configs/replayfree_microscopic.py'

custom_imports = dict(
    imports=[
        'exp023_np_compat',
        'projects.lora_cl',
    ],
    allow_failed_imports=False)

# 素の LoRA（design.md §2.2）: 挿入 232 層、r=16 / alpha=16（scaling=1）。
# encoder テキスト側 self-attention は MHA を q/k/v/o に分解して挿入する。
# dn は RF100 枠に従い有効のまま（use_dn を渡さない = 既定 True）。
model = dict(
    type='GroundingDINOLoRA',
    lora=dict(
        r=16,
        alpha=16,
        exclude_components=[],
        decompose_mha=[r'^encoder\.text_layers\.\d+\.self_attn\.attn$'],
        include=['backbone', 'language_model', 'text_feat_map', 'encoder']))
