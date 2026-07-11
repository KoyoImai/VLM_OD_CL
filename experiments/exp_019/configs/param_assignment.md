# exp_019 パラメータ帰属対応表（自動生成: check_param_assignment.py）

- 総パラメータテンソル数: 896
- 条件A(extfusion)で学習: 667 / 条件B(downstream)で学習: 229
- 相補性・意図一致の検査: OK（違反なし）

## 帰属（A=抽出・融合部 / B=下流）

| parameter | side | shape |
|---|---|---|
| `level_embed` | A | (4, 256) |
| `backbone.patch_embed.projection.weight` | A | (96, 3, 4, 4) |
| `backbone.patch_embed.projection.bias` | A | (96,) |
| `backbone.patch_embed.norm.weight` | A | (96,) |
| `backbone.patch_embed.norm.bias` | A | (96,) |
| `backbone.stages.0.blocks.0.norm1.weight` | A | (96,) |
| `backbone.stages.0.blocks.0.norm1.bias` | A | (96,) |
| `backbone.stages.0.blocks.0.attn.w_msa.relative_position_bias_table` | A | (169, 3) |
| `backbone.stages.0.blocks.0.attn.w_msa.qkv.weight` | A | (288, 96) |
| `backbone.stages.0.blocks.0.attn.w_msa.qkv.bias` | A | (288,) |
| `backbone.stages.0.blocks.0.attn.w_msa.proj.weight` | A | (96, 96) |
| `backbone.stages.0.blocks.0.attn.w_msa.proj.bias` | A | (96,) |
| `backbone.stages.0.blocks.0.norm2.weight` | A | (96,) |
| `backbone.stages.0.blocks.0.norm2.bias` | A | (96,) |
| `backbone.stages.0.blocks.0.ffn.layers.0.0.weight` | A | (384, 96) |
| `backbone.stages.0.blocks.0.ffn.layers.0.0.bias` | A | (384,) |
| `backbone.stages.0.blocks.0.ffn.layers.1.weight` | A | (96, 384) |
| `backbone.stages.0.blocks.0.ffn.layers.1.bias` | A | (96,) |
| `backbone.stages.0.blocks.1.norm1.weight` | A | (96,) |
| `backbone.stages.0.blocks.1.norm1.bias` | A | (96,) |
| `backbone.stages.0.blocks.1.attn.w_msa.relative_position_bias_table` | A | (169, 3) |
| `backbone.stages.0.blocks.1.attn.w_msa.qkv.weight` | A | (288, 96) |
| `backbone.stages.0.blocks.1.attn.w_msa.qkv.bias` | A | (288,) |
| `backbone.stages.0.blocks.1.attn.w_msa.proj.weight` | A | (96, 96) |
| `backbone.stages.0.blocks.1.attn.w_msa.proj.bias` | A | (96,) |
| `backbone.stages.0.blocks.1.norm2.weight` | A | (96,) |
| `backbone.stages.0.blocks.1.norm2.bias` | A | (96,) |
| `backbone.stages.0.blocks.1.ffn.layers.0.0.weight` | A | (384, 96) |
| `backbone.stages.0.blocks.1.ffn.layers.0.0.bias` | A | (384,) |
| `backbone.stages.0.blocks.1.ffn.layers.1.weight` | A | (96, 384) |
| `backbone.stages.0.blocks.1.ffn.layers.1.bias` | A | (96,) |
| `backbone.stages.0.downsample.norm.weight` | A | (384,) |
| `backbone.stages.0.downsample.norm.bias` | A | (384,) |
| `backbone.stages.0.downsample.reduction.weight` | A | (192, 384) |
| `backbone.stages.1.blocks.0.norm1.weight` | A | (192,) |
| `backbone.stages.1.blocks.0.norm1.bias` | A | (192,) |
| `backbone.stages.1.blocks.0.attn.w_msa.relative_position_bias_table` | A | (169, 6) |
| `backbone.stages.1.blocks.0.attn.w_msa.qkv.weight` | A | (576, 192) |
| `backbone.stages.1.blocks.0.attn.w_msa.qkv.bias` | A | (576,) |
| `backbone.stages.1.blocks.0.attn.w_msa.proj.weight` | A | (192, 192) |
| `backbone.stages.1.blocks.0.attn.w_msa.proj.bias` | A | (192,) |
| `backbone.stages.1.blocks.0.norm2.weight` | A | (192,) |
| `backbone.stages.1.blocks.0.norm2.bias` | A | (192,) |
| `backbone.stages.1.blocks.0.ffn.layers.0.0.weight` | A | (768, 192) |
| `backbone.stages.1.blocks.0.ffn.layers.0.0.bias` | A | (768,) |
| `backbone.stages.1.blocks.0.ffn.layers.1.weight` | A | (192, 768) |
| `backbone.stages.1.blocks.0.ffn.layers.1.bias` | A | (192,) |
| `backbone.stages.1.blocks.1.norm1.weight` | A | (192,) |
| `backbone.stages.1.blocks.1.norm1.bias` | A | (192,) |
| `backbone.stages.1.blocks.1.attn.w_msa.relative_position_bias_table` | A | (169, 6) |
| `backbone.stages.1.blocks.1.attn.w_msa.qkv.weight` | A | (576, 192) |
| `backbone.stages.1.blocks.1.attn.w_msa.qkv.bias` | A | (576,) |
| `backbone.stages.1.blocks.1.attn.w_msa.proj.weight` | A | (192, 192) |
| `backbone.stages.1.blocks.1.attn.w_msa.proj.bias` | A | (192,) |
| `backbone.stages.1.blocks.1.norm2.weight` | A | (192,) |
| `backbone.stages.1.blocks.1.norm2.bias` | A | (192,) |
| `backbone.stages.1.blocks.1.ffn.layers.0.0.weight` | A | (768, 192) |
| `backbone.stages.1.blocks.1.ffn.layers.0.0.bias` | A | (768,) |
| `backbone.stages.1.blocks.1.ffn.layers.1.weight` | A | (192, 768) |
| `backbone.stages.1.blocks.1.ffn.layers.1.bias` | A | (192,) |
| `backbone.stages.1.downsample.norm.weight` | A | (768,) |
| `backbone.stages.1.downsample.norm.bias` | A | (768,) |
| `backbone.stages.1.downsample.reduction.weight` | A | (384, 768) |
| `backbone.stages.2.blocks.0.norm1.weight` | A | (384,) |
| `backbone.stages.2.blocks.0.norm1.bias` | A | (384,) |
| `backbone.stages.2.blocks.0.attn.w_msa.relative_position_bias_table` | A | (169, 12) |
| `backbone.stages.2.blocks.0.attn.w_msa.qkv.weight` | A | (1152, 384) |
| `backbone.stages.2.blocks.0.attn.w_msa.qkv.bias` | A | (1152,) |
| `backbone.stages.2.blocks.0.attn.w_msa.proj.weight` | A | (384, 384) |
| `backbone.stages.2.blocks.0.attn.w_msa.proj.bias` | A | (384,) |
| `backbone.stages.2.blocks.0.norm2.weight` | A | (384,) |
| `backbone.stages.2.blocks.0.norm2.bias` | A | (384,) |
| `backbone.stages.2.blocks.0.ffn.layers.0.0.weight` | A | (1536, 384) |
| `backbone.stages.2.blocks.0.ffn.layers.0.0.bias` | A | (1536,) |
| `backbone.stages.2.blocks.0.ffn.layers.1.weight` | A | (384, 1536) |
| `backbone.stages.2.blocks.0.ffn.layers.1.bias` | A | (384,) |
| `backbone.stages.2.blocks.1.norm1.weight` | A | (384,) |
| `backbone.stages.2.blocks.1.norm1.bias` | A | (384,) |
| `backbone.stages.2.blocks.1.attn.w_msa.relative_position_bias_table` | A | (169, 12) |
| `backbone.stages.2.blocks.1.attn.w_msa.qkv.weight` | A | (1152, 384) |
| `backbone.stages.2.blocks.1.attn.w_msa.qkv.bias` | A | (1152,) |
| `backbone.stages.2.blocks.1.attn.w_msa.proj.weight` | A | (384, 384) |
| `backbone.stages.2.blocks.1.attn.w_msa.proj.bias` | A | (384,) |
| `backbone.stages.2.blocks.1.norm2.weight` | A | (384,) |
| `backbone.stages.2.blocks.1.norm2.bias` | A | (384,) |
| `backbone.stages.2.blocks.1.ffn.layers.0.0.weight` | A | (1536, 384) |
| `backbone.stages.2.blocks.1.ffn.layers.0.0.bias` | A | (1536,) |
| `backbone.stages.2.blocks.1.ffn.layers.1.weight` | A | (384, 1536) |
| `backbone.stages.2.blocks.1.ffn.layers.1.bias` | A | (384,) |
| `backbone.stages.2.blocks.2.norm1.weight` | A | (384,) |
| `backbone.stages.2.blocks.2.norm1.bias` | A | (384,) |
| `backbone.stages.2.blocks.2.attn.w_msa.relative_position_bias_table` | A | (169, 12) |
| `backbone.stages.2.blocks.2.attn.w_msa.qkv.weight` | A | (1152, 384) |
| `backbone.stages.2.blocks.2.attn.w_msa.qkv.bias` | A | (1152,) |
| `backbone.stages.2.blocks.2.attn.w_msa.proj.weight` | A | (384, 384) |
| `backbone.stages.2.blocks.2.attn.w_msa.proj.bias` | A | (384,) |
| `backbone.stages.2.blocks.2.norm2.weight` | A | (384,) |
| `backbone.stages.2.blocks.2.norm2.bias` | A | (384,) |
| `backbone.stages.2.blocks.2.ffn.layers.0.0.weight` | A | (1536, 384) |
| `backbone.stages.2.blocks.2.ffn.layers.0.0.bias` | A | (1536,) |
| `backbone.stages.2.blocks.2.ffn.layers.1.weight` | A | (384, 1536) |
| `backbone.stages.2.blocks.2.ffn.layers.1.bias` | A | (384,) |
| `backbone.stages.2.blocks.3.norm1.weight` | A | (384,) |
| `backbone.stages.2.blocks.3.norm1.bias` | A | (384,) |
| `backbone.stages.2.blocks.3.attn.w_msa.relative_position_bias_table` | A | (169, 12) |
| `backbone.stages.2.blocks.3.attn.w_msa.qkv.weight` | A | (1152, 384) |
| `backbone.stages.2.blocks.3.attn.w_msa.qkv.bias` | A | (1152,) |
| `backbone.stages.2.blocks.3.attn.w_msa.proj.weight` | A | (384, 384) |
| `backbone.stages.2.blocks.3.attn.w_msa.proj.bias` | A | (384,) |
| `backbone.stages.2.blocks.3.norm2.weight` | A | (384,) |
| `backbone.stages.2.blocks.3.norm2.bias` | A | (384,) |
| `backbone.stages.2.blocks.3.ffn.layers.0.0.weight` | A | (1536, 384) |
| `backbone.stages.2.blocks.3.ffn.layers.0.0.bias` | A | (1536,) |
| `backbone.stages.2.blocks.3.ffn.layers.1.weight` | A | (384, 1536) |
| `backbone.stages.2.blocks.3.ffn.layers.1.bias` | A | (384,) |
| `backbone.stages.2.blocks.4.norm1.weight` | A | (384,) |
| `backbone.stages.2.blocks.4.norm1.bias` | A | (384,) |
| `backbone.stages.2.blocks.4.attn.w_msa.relative_position_bias_table` | A | (169, 12) |
| `backbone.stages.2.blocks.4.attn.w_msa.qkv.weight` | A | (1152, 384) |
| `backbone.stages.2.blocks.4.attn.w_msa.qkv.bias` | A | (1152,) |
| `backbone.stages.2.blocks.4.attn.w_msa.proj.weight` | A | (384, 384) |
| `backbone.stages.2.blocks.4.attn.w_msa.proj.bias` | A | (384,) |
| `backbone.stages.2.blocks.4.norm2.weight` | A | (384,) |
| `backbone.stages.2.blocks.4.norm2.bias` | A | (384,) |
| `backbone.stages.2.blocks.4.ffn.layers.0.0.weight` | A | (1536, 384) |
| `backbone.stages.2.blocks.4.ffn.layers.0.0.bias` | A | (1536,) |
| `backbone.stages.2.blocks.4.ffn.layers.1.weight` | A | (384, 1536) |
| `backbone.stages.2.blocks.4.ffn.layers.1.bias` | A | (384,) |
| `backbone.stages.2.blocks.5.norm1.weight` | A | (384,) |
| `backbone.stages.2.blocks.5.norm1.bias` | A | (384,) |
| `backbone.stages.2.blocks.5.attn.w_msa.relative_position_bias_table` | A | (169, 12) |
| `backbone.stages.2.blocks.5.attn.w_msa.qkv.weight` | A | (1152, 384) |
| `backbone.stages.2.blocks.5.attn.w_msa.qkv.bias` | A | (1152,) |
| `backbone.stages.2.blocks.5.attn.w_msa.proj.weight` | A | (384, 384) |
| `backbone.stages.2.blocks.5.attn.w_msa.proj.bias` | A | (384,) |
| `backbone.stages.2.blocks.5.norm2.weight` | A | (384,) |
| `backbone.stages.2.blocks.5.norm2.bias` | A | (384,) |
| `backbone.stages.2.blocks.5.ffn.layers.0.0.weight` | A | (1536, 384) |
| `backbone.stages.2.blocks.5.ffn.layers.0.0.bias` | A | (1536,) |
| `backbone.stages.2.blocks.5.ffn.layers.1.weight` | A | (384, 1536) |
| `backbone.stages.2.blocks.5.ffn.layers.1.bias` | A | (384,) |
| `backbone.stages.2.downsample.norm.weight` | A | (1536,) |
| `backbone.stages.2.downsample.norm.bias` | A | (1536,) |
| `backbone.stages.2.downsample.reduction.weight` | A | (768, 1536) |
| `backbone.stages.3.blocks.0.norm1.weight` | A | (768,) |
| `backbone.stages.3.blocks.0.norm1.bias` | A | (768,) |
| `backbone.stages.3.blocks.0.attn.w_msa.relative_position_bias_table` | A | (169, 24) |
| `backbone.stages.3.blocks.0.attn.w_msa.qkv.weight` | A | (2304, 768) |
| `backbone.stages.3.blocks.0.attn.w_msa.qkv.bias` | A | (2304,) |
| `backbone.stages.3.blocks.0.attn.w_msa.proj.weight` | A | (768, 768) |
| `backbone.stages.3.blocks.0.attn.w_msa.proj.bias` | A | (768,) |
| `backbone.stages.3.blocks.0.norm2.weight` | A | (768,) |
| `backbone.stages.3.blocks.0.norm2.bias` | A | (768,) |
| `backbone.stages.3.blocks.0.ffn.layers.0.0.weight` | A | (3072, 768) |
| `backbone.stages.3.blocks.0.ffn.layers.0.0.bias` | A | (3072,) |
| `backbone.stages.3.blocks.0.ffn.layers.1.weight` | A | (768, 3072) |
| `backbone.stages.3.blocks.0.ffn.layers.1.bias` | A | (768,) |
| `backbone.stages.3.blocks.1.norm1.weight` | A | (768,) |
| `backbone.stages.3.blocks.1.norm1.bias` | A | (768,) |
| `backbone.stages.3.blocks.1.attn.w_msa.relative_position_bias_table` | A | (169, 24) |
| `backbone.stages.3.blocks.1.attn.w_msa.qkv.weight` | A | (2304, 768) |
| `backbone.stages.3.blocks.1.attn.w_msa.qkv.bias` | A | (2304,) |
| `backbone.stages.3.blocks.1.attn.w_msa.proj.weight` | A | (768, 768) |
| `backbone.stages.3.blocks.1.attn.w_msa.proj.bias` | A | (768,) |
| `backbone.stages.3.blocks.1.norm2.weight` | A | (768,) |
| `backbone.stages.3.blocks.1.norm2.bias` | A | (768,) |
| `backbone.stages.3.blocks.1.ffn.layers.0.0.weight` | A | (3072, 768) |
| `backbone.stages.3.blocks.1.ffn.layers.0.0.bias` | A | (3072,) |
| `backbone.stages.3.blocks.1.ffn.layers.1.weight` | A | (768, 3072) |
| `backbone.stages.3.blocks.1.ffn.layers.1.bias` | A | (768,) |
| `backbone.norm1.weight` | A | (192,) |
| `backbone.norm1.bias` | A | (192,) |
| `backbone.norm2.weight` | A | (384,) |
| `backbone.norm2.bias` | A | (384,) |
| `backbone.norm3.weight` | A | (768,) |
| `backbone.norm3.bias` | A | (768,) |
| `neck.convs.0.conv.weight` | A | (256, 192, 1, 1) |
| `neck.convs.0.conv.bias` | A | (256,) |
| `neck.convs.0.gn.weight` | A | (256,) |
| `neck.convs.0.gn.bias` | A | (256,) |
| `neck.convs.1.conv.weight` | A | (256, 384, 1, 1) |
| `neck.convs.1.conv.bias` | A | (256,) |
| `neck.convs.1.gn.weight` | A | (256,) |
| `neck.convs.1.gn.bias` | A | (256,) |
| `neck.convs.2.conv.weight` | A | (256, 768, 1, 1) |
| `neck.convs.2.conv.bias` | A | (256,) |
| `neck.convs.2.gn.weight` | A | (256,) |
| `neck.convs.2.gn.bias` | A | (256,) |
| `neck.extra_convs.0.conv.weight` | A | (256, 768, 3, 3) |
| `neck.extra_convs.0.conv.bias` | A | (256,) |
| `neck.extra_convs.0.gn.weight` | A | (256,) |
| `neck.extra_convs.0.gn.bias` | A | (256,) |
| `bbox_head.cls_branches.0.bias` | B | (1,) |
| `bbox_head.cls_branches.1.bias` | B | (1,) |
| `bbox_head.cls_branches.2.bias` | B | (1,) |
| `bbox_head.cls_branches.3.bias` | B | (1,) |
| `bbox_head.cls_branches.4.bias` | B | (1,) |
| `bbox_head.cls_branches.5.bias` | B | (1,) |
| `bbox_head.cls_branches.6.bias` | B | (1,) |
| `bbox_head.reg_branches.0.0.weight` | B | (256, 256) |
| `bbox_head.reg_branches.0.0.bias` | B | (256,) |
| `bbox_head.reg_branches.0.2.weight` | B | (256, 256) |
| `bbox_head.reg_branches.0.2.bias` | B | (256,) |
| `bbox_head.reg_branches.0.4.weight` | B | (4, 256) |
| `bbox_head.reg_branches.0.4.bias` | B | (4,) |
| `bbox_head.reg_branches.1.0.weight` | B | (256, 256) |
| `bbox_head.reg_branches.1.0.bias` | B | (256,) |
| `bbox_head.reg_branches.1.2.weight` | B | (256, 256) |
| `bbox_head.reg_branches.1.2.bias` | B | (256,) |
| `bbox_head.reg_branches.1.4.weight` | B | (4, 256) |
| `bbox_head.reg_branches.1.4.bias` | B | (4,) |
| `bbox_head.reg_branches.2.0.weight` | B | (256, 256) |
| `bbox_head.reg_branches.2.0.bias` | B | (256,) |
| `bbox_head.reg_branches.2.2.weight` | B | (256, 256) |
| `bbox_head.reg_branches.2.2.bias` | B | (256,) |
| `bbox_head.reg_branches.2.4.weight` | B | (4, 256) |
| `bbox_head.reg_branches.2.4.bias` | B | (4,) |
| `bbox_head.reg_branches.3.0.weight` | B | (256, 256) |
| `bbox_head.reg_branches.3.0.bias` | B | (256,) |
| `bbox_head.reg_branches.3.2.weight` | B | (256, 256) |
| `bbox_head.reg_branches.3.2.bias` | B | (256,) |
| `bbox_head.reg_branches.3.4.weight` | B | (4, 256) |
| `bbox_head.reg_branches.3.4.bias` | B | (4,) |
| `bbox_head.reg_branches.4.0.weight` | B | (256, 256) |
| `bbox_head.reg_branches.4.0.bias` | B | (256,) |
| `bbox_head.reg_branches.4.2.weight` | B | (256, 256) |
| `bbox_head.reg_branches.4.2.bias` | B | (256,) |
| `bbox_head.reg_branches.4.4.weight` | B | (4, 256) |
| `bbox_head.reg_branches.4.4.bias` | B | (4,) |
| `bbox_head.reg_branches.5.0.weight` | B | (256, 256) |
| `bbox_head.reg_branches.5.0.bias` | B | (256,) |
| `bbox_head.reg_branches.5.2.weight` | B | (256, 256) |
| `bbox_head.reg_branches.5.2.bias` | B | (256,) |
| `bbox_head.reg_branches.5.4.weight` | B | (4, 256) |
| `bbox_head.reg_branches.5.4.bias` | B | (4,) |
| `bbox_head.reg_branches.6.0.weight` | B | (256, 256) |
| `bbox_head.reg_branches.6.0.bias` | B | (256,) |
| `bbox_head.reg_branches.6.2.weight` | B | (256, 256) |
| `bbox_head.reg_branches.6.2.bias` | B | (256,) |
| `bbox_head.reg_branches.6.4.weight` | B | (4, 256) |
| `bbox_head.reg_branches.6.4.bias` | B | (4,) |
| `encoder.layers.0.self_attn.sampling_offsets.weight` | A | (256, 256) |
| `encoder.layers.0.self_attn.sampling_offsets.bias` | A | (256,) |
| `encoder.layers.0.self_attn.attention_weights.weight` | A | (128, 256) |
| `encoder.layers.0.self_attn.attention_weights.bias` | A | (128,) |
| `encoder.layers.0.self_attn.value_proj.weight` | A | (256, 256) |
| `encoder.layers.0.self_attn.value_proj.bias` | A | (256,) |
| `encoder.layers.0.self_attn.output_proj.weight` | A | (256, 256) |
| `encoder.layers.0.self_attn.output_proj.bias` | A | (256,) |
| `encoder.layers.0.ffn.layers.0.0.weight` | A | (2048, 256) |
| `encoder.layers.0.ffn.layers.0.0.bias` | A | (2048,) |
| `encoder.layers.0.ffn.layers.1.weight` | A | (256, 2048) |
| `encoder.layers.0.ffn.layers.1.bias` | A | (256,) |
| `encoder.layers.0.norms.0.weight` | A | (256,) |
| `encoder.layers.0.norms.0.bias` | A | (256,) |
| `encoder.layers.0.norms.1.weight` | A | (256,) |
| `encoder.layers.0.norms.1.bias` | A | (256,) |
| `encoder.layers.1.self_attn.sampling_offsets.weight` | A | (256, 256) |
| `encoder.layers.1.self_attn.sampling_offsets.bias` | A | (256,) |
| `encoder.layers.1.self_attn.attention_weights.weight` | A | (128, 256) |
| `encoder.layers.1.self_attn.attention_weights.bias` | A | (128,) |
| `encoder.layers.1.self_attn.value_proj.weight` | A | (256, 256) |
| `encoder.layers.1.self_attn.value_proj.bias` | A | (256,) |
| `encoder.layers.1.self_attn.output_proj.weight` | A | (256, 256) |
| `encoder.layers.1.self_attn.output_proj.bias` | A | (256,) |
| `encoder.layers.1.ffn.layers.0.0.weight` | A | (2048, 256) |
| `encoder.layers.1.ffn.layers.0.0.bias` | A | (2048,) |
| `encoder.layers.1.ffn.layers.1.weight` | A | (256, 2048) |
| `encoder.layers.1.ffn.layers.1.bias` | A | (256,) |
| `encoder.layers.1.norms.0.weight` | A | (256,) |
| `encoder.layers.1.norms.0.bias` | A | (256,) |
| `encoder.layers.1.norms.1.weight` | A | (256,) |
| `encoder.layers.1.norms.1.bias` | A | (256,) |
| `encoder.layers.2.self_attn.sampling_offsets.weight` | A | (256, 256) |
| `encoder.layers.2.self_attn.sampling_offsets.bias` | A | (256,) |
| `encoder.layers.2.self_attn.attention_weights.weight` | A | (128, 256) |
| `encoder.layers.2.self_attn.attention_weights.bias` | A | (128,) |
| `encoder.layers.2.self_attn.value_proj.weight` | A | (256, 256) |
| `encoder.layers.2.self_attn.value_proj.bias` | A | (256,) |
| `encoder.layers.2.self_attn.output_proj.weight` | A | (256, 256) |
| `encoder.layers.2.self_attn.output_proj.bias` | A | (256,) |
| `encoder.layers.2.ffn.layers.0.0.weight` | A | (2048, 256) |
| `encoder.layers.2.ffn.layers.0.0.bias` | A | (2048,) |
| `encoder.layers.2.ffn.layers.1.weight` | A | (256, 2048) |
| `encoder.layers.2.ffn.layers.1.bias` | A | (256,) |
| `encoder.layers.2.norms.0.weight` | A | (256,) |
| `encoder.layers.2.norms.0.bias` | A | (256,) |
| `encoder.layers.2.norms.1.weight` | A | (256,) |
| `encoder.layers.2.norms.1.bias` | A | (256,) |
| `encoder.layers.3.self_attn.sampling_offsets.weight` | A | (256, 256) |
| `encoder.layers.3.self_attn.sampling_offsets.bias` | A | (256,) |
| `encoder.layers.3.self_attn.attention_weights.weight` | A | (128, 256) |
| `encoder.layers.3.self_attn.attention_weights.bias` | A | (128,) |
| `encoder.layers.3.self_attn.value_proj.weight` | A | (256, 256) |
| `encoder.layers.3.self_attn.value_proj.bias` | A | (256,) |
| `encoder.layers.3.self_attn.output_proj.weight` | A | (256, 256) |
| `encoder.layers.3.self_attn.output_proj.bias` | A | (256,) |
| `encoder.layers.3.ffn.layers.0.0.weight` | A | (2048, 256) |
| `encoder.layers.3.ffn.layers.0.0.bias` | A | (2048,) |
| `encoder.layers.3.ffn.layers.1.weight` | A | (256, 2048) |
| `encoder.layers.3.ffn.layers.1.bias` | A | (256,) |
| `encoder.layers.3.norms.0.weight` | A | (256,) |
| `encoder.layers.3.norms.0.bias` | A | (256,) |
| `encoder.layers.3.norms.1.weight` | A | (256,) |
| `encoder.layers.3.norms.1.bias` | A | (256,) |
| `encoder.layers.4.self_attn.sampling_offsets.weight` | A | (256, 256) |
| `encoder.layers.4.self_attn.sampling_offsets.bias` | A | (256,) |
| `encoder.layers.4.self_attn.attention_weights.weight` | A | (128, 256) |
| `encoder.layers.4.self_attn.attention_weights.bias` | A | (128,) |
| `encoder.layers.4.self_attn.value_proj.weight` | A | (256, 256) |
| `encoder.layers.4.self_attn.value_proj.bias` | A | (256,) |
| `encoder.layers.4.self_attn.output_proj.weight` | A | (256, 256) |
| `encoder.layers.4.self_attn.output_proj.bias` | A | (256,) |
| `encoder.layers.4.ffn.layers.0.0.weight` | A | (2048, 256) |
| `encoder.layers.4.ffn.layers.0.0.bias` | A | (2048,) |
| `encoder.layers.4.ffn.layers.1.weight` | A | (256, 2048) |
| `encoder.layers.4.ffn.layers.1.bias` | A | (256,) |
| `encoder.layers.4.norms.0.weight` | A | (256,) |
| `encoder.layers.4.norms.0.bias` | A | (256,) |
| `encoder.layers.4.norms.1.weight` | A | (256,) |
| `encoder.layers.4.norms.1.bias` | A | (256,) |
| `encoder.layers.5.self_attn.sampling_offsets.weight` | A | (256, 256) |
| `encoder.layers.5.self_attn.sampling_offsets.bias` | A | (256,) |
| `encoder.layers.5.self_attn.attention_weights.weight` | A | (128, 256) |
| `encoder.layers.5.self_attn.attention_weights.bias` | A | (128,) |
| `encoder.layers.5.self_attn.value_proj.weight` | A | (256, 256) |
| `encoder.layers.5.self_attn.value_proj.bias` | A | (256,) |
| `encoder.layers.5.self_attn.output_proj.weight` | A | (256, 256) |
| `encoder.layers.5.self_attn.output_proj.bias` | A | (256,) |
| `encoder.layers.5.ffn.layers.0.0.weight` | A | (2048, 256) |
| `encoder.layers.5.ffn.layers.0.0.bias` | A | (2048,) |
| `encoder.layers.5.ffn.layers.1.weight` | A | (256, 2048) |
| `encoder.layers.5.ffn.layers.1.bias` | A | (256,) |
| `encoder.layers.5.norms.0.weight` | A | (256,) |
| `encoder.layers.5.norms.0.bias` | A | (256,) |
| `encoder.layers.5.norms.1.weight` | A | (256,) |
| `encoder.layers.5.norms.1.bias` | A | (256,) |
| `encoder.text_layers.0.self_attn.attn.in_proj_weight` | A | (768, 256) |
| `encoder.text_layers.0.self_attn.attn.in_proj_bias` | A | (768,) |
| `encoder.text_layers.0.self_attn.attn.out_proj.weight` | A | (256, 256) |
| `encoder.text_layers.0.self_attn.attn.out_proj.bias` | A | (256,) |
| `encoder.text_layers.0.ffn.layers.0.0.weight` | A | (1024, 256) |
| `encoder.text_layers.0.ffn.layers.0.0.bias` | A | (1024,) |
| `encoder.text_layers.0.ffn.layers.1.weight` | A | (256, 1024) |
| `encoder.text_layers.0.ffn.layers.1.bias` | A | (256,) |
| `encoder.text_layers.0.norms.0.weight` | A | (256,) |
| `encoder.text_layers.0.norms.0.bias` | A | (256,) |
| `encoder.text_layers.0.norms.1.weight` | A | (256,) |
| `encoder.text_layers.0.norms.1.bias` | A | (256,) |
| `encoder.text_layers.1.self_attn.attn.in_proj_weight` | A | (768, 256) |
| `encoder.text_layers.1.self_attn.attn.in_proj_bias` | A | (768,) |
| `encoder.text_layers.1.self_attn.attn.out_proj.weight` | A | (256, 256) |
| `encoder.text_layers.1.self_attn.attn.out_proj.bias` | A | (256,) |
| `encoder.text_layers.1.ffn.layers.0.0.weight` | A | (1024, 256) |
| `encoder.text_layers.1.ffn.layers.0.0.bias` | A | (1024,) |
| `encoder.text_layers.1.ffn.layers.1.weight` | A | (256, 1024) |
| `encoder.text_layers.1.ffn.layers.1.bias` | A | (256,) |
| `encoder.text_layers.1.norms.0.weight` | A | (256,) |
| `encoder.text_layers.1.norms.0.bias` | A | (256,) |
| `encoder.text_layers.1.norms.1.weight` | A | (256,) |
| `encoder.text_layers.1.norms.1.bias` | A | (256,) |
| `encoder.text_layers.2.self_attn.attn.in_proj_weight` | A | (768, 256) |
| `encoder.text_layers.2.self_attn.attn.in_proj_bias` | A | (768,) |
| `encoder.text_layers.2.self_attn.attn.out_proj.weight` | A | (256, 256) |
| `encoder.text_layers.2.self_attn.attn.out_proj.bias` | A | (256,) |
| `encoder.text_layers.2.ffn.layers.0.0.weight` | A | (1024, 256) |
| `encoder.text_layers.2.ffn.layers.0.0.bias` | A | (1024,) |
| `encoder.text_layers.2.ffn.layers.1.weight` | A | (256, 1024) |
| `encoder.text_layers.2.ffn.layers.1.bias` | A | (256,) |
| `encoder.text_layers.2.norms.0.weight` | A | (256,) |
| `encoder.text_layers.2.norms.0.bias` | A | (256,) |
| `encoder.text_layers.2.norms.1.weight` | A | (256,) |
| `encoder.text_layers.2.norms.1.bias` | A | (256,) |
| `encoder.text_layers.3.self_attn.attn.in_proj_weight` | A | (768, 256) |
| `encoder.text_layers.3.self_attn.attn.in_proj_bias` | A | (768,) |
| `encoder.text_layers.3.self_attn.attn.out_proj.weight` | A | (256, 256) |
| `encoder.text_layers.3.self_attn.attn.out_proj.bias` | A | (256,) |
| `encoder.text_layers.3.ffn.layers.0.0.weight` | A | (1024, 256) |
| `encoder.text_layers.3.ffn.layers.0.0.bias` | A | (1024,) |
| `encoder.text_layers.3.ffn.layers.1.weight` | A | (256, 1024) |
| `encoder.text_layers.3.ffn.layers.1.bias` | A | (256,) |
| `encoder.text_layers.3.norms.0.weight` | A | (256,) |
| `encoder.text_layers.3.norms.0.bias` | A | (256,) |
| `encoder.text_layers.3.norms.1.weight` | A | (256,) |
| `encoder.text_layers.3.norms.1.bias` | A | (256,) |
| `encoder.text_layers.4.self_attn.attn.in_proj_weight` | A | (768, 256) |
| `encoder.text_layers.4.self_attn.attn.in_proj_bias` | A | (768,) |
| `encoder.text_layers.4.self_attn.attn.out_proj.weight` | A | (256, 256) |
| `encoder.text_layers.4.self_attn.attn.out_proj.bias` | A | (256,) |
| `encoder.text_layers.4.ffn.layers.0.0.weight` | A | (1024, 256) |
| `encoder.text_layers.4.ffn.layers.0.0.bias` | A | (1024,) |
| `encoder.text_layers.4.ffn.layers.1.weight` | A | (256, 1024) |
| `encoder.text_layers.4.ffn.layers.1.bias` | A | (256,) |
| `encoder.text_layers.4.norms.0.weight` | A | (256,) |
| `encoder.text_layers.4.norms.0.bias` | A | (256,) |
| `encoder.text_layers.4.norms.1.weight` | A | (256,) |
| `encoder.text_layers.4.norms.1.bias` | A | (256,) |
| `encoder.text_layers.5.self_attn.attn.in_proj_weight` | A | (768, 256) |
| `encoder.text_layers.5.self_attn.attn.in_proj_bias` | A | (768,) |
| `encoder.text_layers.5.self_attn.attn.out_proj.weight` | A | (256, 256) |
| `encoder.text_layers.5.self_attn.attn.out_proj.bias` | A | (256,) |
| `encoder.text_layers.5.ffn.layers.0.0.weight` | A | (1024, 256) |
| `encoder.text_layers.5.ffn.layers.0.0.bias` | A | (1024,) |
| `encoder.text_layers.5.ffn.layers.1.weight` | A | (256, 1024) |
| `encoder.text_layers.5.ffn.layers.1.bias` | A | (256,) |
| `encoder.text_layers.5.norms.0.weight` | A | (256,) |
| `encoder.text_layers.5.norms.0.bias` | A | (256,) |
| `encoder.text_layers.5.norms.1.weight` | A | (256,) |
| `encoder.text_layers.5.norms.1.bias` | A | (256,) |
| `encoder.fusion_layers.0.gamma_v` | A | (256,) |
| `encoder.fusion_layers.0.gamma_l` | A | (256,) |
| `encoder.fusion_layers.0.layer_norm_v.weight` | A | (256,) |
| `encoder.fusion_layers.0.layer_norm_v.bias` | A | (256,) |
| `encoder.fusion_layers.0.layer_norm_l.weight` | A | (256,) |
| `encoder.fusion_layers.0.layer_norm_l.bias` | A | (256,) |
| `encoder.fusion_layers.0.attn.v_proj.weight` | A | (1024, 256) |
| `encoder.fusion_layers.0.attn.v_proj.bias` | A | (1024,) |
| `encoder.fusion_layers.0.attn.l_proj.weight` | A | (1024, 256) |
| `encoder.fusion_layers.0.attn.l_proj.bias` | A | (1024,) |
| `encoder.fusion_layers.0.attn.values_v_proj.weight` | A | (1024, 256) |
| `encoder.fusion_layers.0.attn.values_v_proj.bias` | A | (1024,) |
| `encoder.fusion_layers.0.attn.values_l_proj.weight` | A | (1024, 256) |
| `encoder.fusion_layers.0.attn.values_l_proj.bias` | A | (1024,) |
| `encoder.fusion_layers.0.attn.out_v_proj.weight` | A | (256, 1024) |
| `encoder.fusion_layers.0.attn.out_v_proj.bias` | A | (256,) |
| `encoder.fusion_layers.0.attn.out_l_proj.weight` | A | (256, 1024) |
| `encoder.fusion_layers.0.attn.out_l_proj.bias` | A | (256,) |
| `encoder.fusion_layers.1.gamma_v` | A | (256,) |
| `encoder.fusion_layers.1.gamma_l` | A | (256,) |
| `encoder.fusion_layers.1.layer_norm_v.weight` | A | (256,) |
| `encoder.fusion_layers.1.layer_norm_v.bias` | A | (256,) |
| `encoder.fusion_layers.1.layer_norm_l.weight` | A | (256,) |
| `encoder.fusion_layers.1.layer_norm_l.bias` | A | (256,) |
| `encoder.fusion_layers.1.attn.v_proj.weight` | A | (1024, 256) |
| `encoder.fusion_layers.1.attn.v_proj.bias` | A | (1024,) |
| `encoder.fusion_layers.1.attn.l_proj.weight` | A | (1024, 256) |
| `encoder.fusion_layers.1.attn.l_proj.bias` | A | (1024,) |
| `encoder.fusion_layers.1.attn.values_v_proj.weight` | A | (1024, 256) |
| `encoder.fusion_layers.1.attn.values_v_proj.bias` | A | (1024,) |
| `encoder.fusion_layers.1.attn.values_l_proj.weight` | A | (1024, 256) |
| `encoder.fusion_layers.1.attn.values_l_proj.bias` | A | (1024,) |
| `encoder.fusion_layers.1.attn.out_v_proj.weight` | A | (256, 1024) |
| `encoder.fusion_layers.1.attn.out_v_proj.bias` | A | (256,) |
| `encoder.fusion_layers.1.attn.out_l_proj.weight` | A | (256, 1024) |
| `encoder.fusion_layers.1.attn.out_l_proj.bias` | A | (256,) |
| `encoder.fusion_layers.2.gamma_v` | A | (256,) |
| `encoder.fusion_layers.2.gamma_l` | A | (256,) |
| `encoder.fusion_layers.2.layer_norm_v.weight` | A | (256,) |
| `encoder.fusion_layers.2.layer_norm_v.bias` | A | (256,) |
| `encoder.fusion_layers.2.layer_norm_l.weight` | A | (256,) |
| `encoder.fusion_layers.2.layer_norm_l.bias` | A | (256,) |
| `encoder.fusion_layers.2.attn.v_proj.weight` | A | (1024, 256) |
| `encoder.fusion_layers.2.attn.v_proj.bias` | A | (1024,) |
| `encoder.fusion_layers.2.attn.l_proj.weight` | A | (1024, 256) |
| `encoder.fusion_layers.2.attn.l_proj.bias` | A | (1024,) |
| `encoder.fusion_layers.2.attn.values_v_proj.weight` | A | (1024, 256) |
| `encoder.fusion_layers.2.attn.values_v_proj.bias` | A | (1024,) |
| `encoder.fusion_layers.2.attn.values_l_proj.weight` | A | (1024, 256) |
| `encoder.fusion_layers.2.attn.values_l_proj.bias` | A | (1024,) |
| `encoder.fusion_layers.2.attn.out_v_proj.weight` | A | (256, 1024) |
| `encoder.fusion_layers.2.attn.out_v_proj.bias` | A | (256,) |
| `encoder.fusion_layers.2.attn.out_l_proj.weight` | A | (256, 1024) |
| `encoder.fusion_layers.2.attn.out_l_proj.bias` | A | (256,) |
| `encoder.fusion_layers.3.gamma_v` | A | (256,) |
| `encoder.fusion_layers.3.gamma_l` | A | (256,) |
| `encoder.fusion_layers.3.layer_norm_v.weight` | A | (256,) |
| `encoder.fusion_layers.3.layer_norm_v.bias` | A | (256,) |
| `encoder.fusion_layers.3.layer_norm_l.weight` | A | (256,) |
| `encoder.fusion_layers.3.layer_norm_l.bias` | A | (256,) |
| `encoder.fusion_layers.3.attn.v_proj.weight` | A | (1024, 256) |
| `encoder.fusion_layers.3.attn.v_proj.bias` | A | (1024,) |
| `encoder.fusion_layers.3.attn.l_proj.weight` | A | (1024, 256) |
| `encoder.fusion_layers.3.attn.l_proj.bias` | A | (1024,) |
| `encoder.fusion_layers.3.attn.values_v_proj.weight` | A | (1024, 256) |
| `encoder.fusion_layers.3.attn.values_v_proj.bias` | A | (1024,) |
| `encoder.fusion_layers.3.attn.values_l_proj.weight` | A | (1024, 256) |
| `encoder.fusion_layers.3.attn.values_l_proj.bias` | A | (1024,) |
| `encoder.fusion_layers.3.attn.out_v_proj.weight` | A | (256, 1024) |
| `encoder.fusion_layers.3.attn.out_v_proj.bias` | A | (256,) |
| `encoder.fusion_layers.3.attn.out_l_proj.weight` | A | (256, 1024) |
| `encoder.fusion_layers.3.attn.out_l_proj.bias` | A | (256,) |
| `encoder.fusion_layers.4.gamma_v` | A | (256,) |
| `encoder.fusion_layers.4.gamma_l` | A | (256,) |
| `encoder.fusion_layers.4.layer_norm_v.weight` | A | (256,) |
| `encoder.fusion_layers.4.layer_norm_v.bias` | A | (256,) |
| `encoder.fusion_layers.4.layer_norm_l.weight` | A | (256,) |
| `encoder.fusion_layers.4.layer_norm_l.bias` | A | (256,) |
| `encoder.fusion_layers.4.attn.v_proj.weight` | A | (1024, 256) |
| `encoder.fusion_layers.4.attn.v_proj.bias` | A | (1024,) |
| `encoder.fusion_layers.4.attn.l_proj.weight` | A | (1024, 256) |
| `encoder.fusion_layers.4.attn.l_proj.bias` | A | (1024,) |
| `encoder.fusion_layers.4.attn.values_v_proj.weight` | A | (1024, 256) |
| `encoder.fusion_layers.4.attn.values_v_proj.bias` | A | (1024,) |
| `encoder.fusion_layers.4.attn.values_l_proj.weight` | A | (1024, 256) |
| `encoder.fusion_layers.4.attn.values_l_proj.bias` | A | (1024,) |
| `encoder.fusion_layers.4.attn.out_v_proj.weight` | A | (256, 1024) |
| `encoder.fusion_layers.4.attn.out_v_proj.bias` | A | (256,) |
| `encoder.fusion_layers.4.attn.out_l_proj.weight` | A | (256, 1024) |
| `encoder.fusion_layers.4.attn.out_l_proj.bias` | A | (256,) |
| `encoder.fusion_layers.5.gamma_v` | A | (256,) |
| `encoder.fusion_layers.5.gamma_l` | A | (256,) |
| `encoder.fusion_layers.5.layer_norm_v.weight` | A | (256,) |
| `encoder.fusion_layers.5.layer_norm_v.bias` | A | (256,) |
| `encoder.fusion_layers.5.layer_norm_l.weight` | A | (256,) |
| `encoder.fusion_layers.5.layer_norm_l.bias` | A | (256,) |
| `encoder.fusion_layers.5.attn.v_proj.weight` | A | (1024, 256) |
| `encoder.fusion_layers.5.attn.v_proj.bias` | A | (1024,) |
| `encoder.fusion_layers.5.attn.l_proj.weight` | A | (1024, 256) |
| `encoder.fusion_layers.5.attn.l_proj.bias` | A | (1024,) |
| `encoder.fusion_layers.5.attn.values_v_proj.weight` | A | (1024, 256) |
| `encoder.fusion_layers.5.attn.values_v_proj.bias` | A | (1024,) |
| `encoder.fusion_layers.5.attn.values_l_proj.weight` | A | (1024, 256) |
| `encoder.fusion_layers.5.attn.values_l_proj.bias` | A | (1024,) |
| `encoder.fusion_layers.5.attn.out_v_proj.weight` | A | (256, 1024) |
| `encoder.fusion_layers.5.attn.out_v_proj.bias` | A | (256,) |
| `encoder.fusion_layers.5.attn.out_l_proj.weight` | A | (256, 1024) |
| `encoder.fusion_layers.5.attn.out_l_proj.bias` | A | (256,) |
| `decoder.layers.0.self_attn.attn.in_proj_weight` | B | (768, 256) |
| `decoder.layers.0.self_attn.attn.in_proj_bias` | B | (768,) |
| `decoder.layers.0.self_attn.attn.out_proj.weight` | B | (256, 256) |
| `decoder.layers.0.self_attn.attn.out_proj.bias` | B | (256,) |
| `decoder.layers.0.cross_attn_text.attn.in_proj_weight` | B | (768, 256) |
| `decoder.layers.0.cross_attn_text.attn.in_proj_bias` | B | (768,) |
| `decoder.layers.0.cross_attn_text.attn.out_proj.weight` | B | (256, 256) |
| `decoder.layers.0.cross_attn_text.attn.out_proj.bias` | B | (256,) |
| `decoder.layers.0.cross_attn.sampling_offsets.weight` | B | (256, 256) |
| `decoder.layers.0.cross_attn.sampling_offsets.bias` | B | (256,) |
| `decoder.layers.0.cross_attn.attention_weights.weight` | B | (128, 256) |
| `decoder.layers.0.cross_attn.attention_weights.bias` | B | (128,) |
| `decoder.layers.0.cross_attn.value_proj.weight` | B | (256, 256) |
| `decoder.layers.0.cross_attn.value_proj.bias` | B | (256,) |
| `decoder.layers.0.cross_attn.output_proj.weight` | B | (256, 256) |
| `decoder.layers.0.cross_attn.output_proj.bias` | B | (256,) |
| `decoder.layers.0.ffn.layers.0.0.weight` | B | (2048, 256) |
| `decoder.layers.0.ffn.layers.0.0.bias` | B | (2048,) |
| `decoder.layers.0.ffn.layers.1.weight` | B | (256, 2048) |
| `decoder.layers.0.ffn.layers.1.bias` | B | (256,) |
| `decoder.layers.0.norms.0.weight` | B | (256,) |
| `decoder.layers.0.norms.0.bias` | B | (256,) |
| `decoder.layers.0.norms.1.weight` | B | (256,) |
| `decoder.layers.0.norms.1.bias` | B | (256,) |
| `decoder.layers.0.norms.2.weight` | B | (256,) |
| `decoder.layers.0.norms.2.bias` | B | (256,) |
| `decoder.layers.0.norms.3.weight` | B | (256,) |
| `decoder.layers.0.norms.3.bias` | B | (256,) |
| `decoder.layers.1.self_attn.attn.in_proj_weight` | B | (768, 256) |
| `decoder.layers.1.self_attn.attn.in_proj_bias` | B | (768,) |
| `decoder.layers.1.self_attn.attn.out_proj.weight` | B | (256, 256) |
| `decoder.layers.1.self_attn.attn.out_proj.bias` | B | (256,) |
| `decoder.layers.1.cross_attn_text.attn.in_proj_weight` | B | (768, 256) |
| `decoder.layers.1.cross_attn_text.attn.in_proj_bias` | B | (768,) |
| `decoder.layers.1.cross_attn_text.attn.out_proj.weight` | B | (256, 256) |
| `decoder.layers.1.cross_attn_text.attn.out_proj.bias` | B | (256,) |
| `decoder.layers.1.cross_attn.sampling_offsets.weight` | B | (256, 256) |
| `decoder.layers.1.cross_attn.sampling_offsets.bias` | B | (256,) |
| `decoder.layers.1.cross_attn.attention_weights.weight` | B | (128, 256) |
| `decoder.layers.1.cross_attn.attention_weights.bias` | B | (128,) |
| `decoder.layers.1.cross_attn.value_proj.weight` | B | (256, 256) |
| `decoder.layers.1.cross_attn.value_proj.bias` | B | (256,) |
| `decoder.layers.1.cross_attn.output_proj.weight` | B | (256, 256) |
| `decoder.layers.1.cross_attn.output_proj.bias` | B | (256,) |
| `decoder.layers.1.ffn.layers.0.0.weight` | B | (2048, 256) |
| `decoder.layers.1.ffn.layers.0.0.bias` | B | (2048,) |
| `decoder.layers.1.ffn.layers.1.weight` | B | (256, 2048) |
| `decoder.layers.1.ffn.layers.1.bias` | B | (256,) |
| `decoder.layers.1.norms.0.weight` | B | (256,) |
| `decoder.layers.1.norms.0.bias` | B | (256,) |
| `decoder.layers.1.norms.1.weight` | B | (256,) |
| `decoder.layers.1.norms.1.bias` | B | (256,) |
| `decoder.layers.1.norms.2.weight` | B | (256,) |
| `decoder.layers.1.norms.2.bias` | B | (256,) |
| `decoder.layers.1.norms.3.weight` | B | (256,) |
| `decoder.layers.1.norms.3.bias` | B | (256,) |
| `decoder.layers.2.self_attn.attn.in_proj_weight` | B | (768, 256) |
| `decoder.layers.2.self_attn.attn.in_proj_bias` | B | (768,) |
| `decoder.layers.2.self_attn.attn.out_proj.weight` | B | (256, 256) |
| `decoder.layers.2.self_attn.attn.out_proj.bias` | B | (256,) |
| `decoder.layers.2.cross_attn_text.attn.in_proj_weight` | B | (768, 256) |
| `decoder.layers.2.cross_attn_text.attn.in_proj_bias` | B | (768,) |
| `decoder.layers.2.cross_attn_text.attn.out_proj.weight` | B | (256, 256) |
| `decoder.layers.2.cross_attn_text.attn.out_proj.bias` | B | (256,) |
| `decoder.layers.2.cross_attn.sampling_offsets.weight` | B | (256, 256) |
| `decoder.layers.2.cross_attn.sampling_offsets.bias` | B | (256,) |
| `decoder.layers.2.cross_attn.attention_weights.weight` | B | (128, 256) |
| `decoder.layers.2.cross_attn.attention_weights.bias` | B | (128,) |
| `decoder.layers.2.cross_attn.value_proj.weight` | B | (256, 256) |
| `decoder.layers.2.cross_attn.value_proj.bias` | B | (256,) |
| `decoder.layers.2.cross_attn.output_proj.weight` | B | (256, 256) |
| `decoder.layers.2.cross_attn.output_proj.bias` | B | (256,) |
| `decoder.layers.2.ffn.layers.0.0.weight` | B | (2048, 256) |
| `decoder.layers.2.ffn.layers.0.0.bias` | B | (2048,) |
| `decoder.layers.2.ffn.layers.1.weight` | B | (256, 2048) |
| `decoder.layers.2.ffn.layers.1.bias` | B | (256,) |
| `decoder.layers.2.norms.0.weight` | B | (256,) |
| `decoder.layers.2.norms.0.bias` | B | (256,) |
| `decoder.layers.2.norms.1.weight` | B | (256,) |
| `decoder.layers.2.norms.1.bias` | B | (256,) |
| `decoder.layers.2.norms.2.weight` | B | (256,) |
| `decoder.layers.2.norms.2.bias` | B | (256,) |
| `decoder.layers.2.norms.3.weight` | B | (256,) |
| `decoder.layers.2.norms.3.bias` | B | (256,) |
| `decoder.layers.3.self_attn.attn.in_proj_weight` | B | (768, 256) |
| `decoder.layers.3.self_attn.attn.in_proj_bias` | B | (768,) |
| `decoder.layers.3.self_attn.attn.out_proj.weight` | B | (256, 256) |
| `decoder.layers.3.self_attn.attn.out_proj.bias` | B | (256,) |
| `decoder.layers.3.cross_attn_text.attn.in_proj_weight` | B | (768, 256) |
| `decoder.layers.3.cross_attn_text.attn.in_proj_bias` | B | (768,) |
| `decoder.layers.3.cross_attn_text.attn.out_proj.weight` | B | (256, 256) |
| `decoder.layers.3.cross_attn_text.attn.out_proj.bias` | B | (256,) |
| `decoder.layers.3.cross_attn.sampling_offsets.weight` | B | (256, 256) |
| `decoder.layers.3.cross_attn.sampling_offsets.bias` | B | (256,) |
| `decoder.layers.3.cross_attn.attention_weights.weight` | B | (128, 256) |
| `decoder.layers.3.cross_attn.attention_weights.bias` | B | (128,) |
| `decoder.layers.3.cross_attn.value_proj.weight` | B | (256, 256) |
| `decoder.layers.3.cross_attn.value_proj.bias` | B | (256,) |
| `decoder.layers.3.cross_attn.output_proj.weight` | B | (256, 256) |
| `decoder.layers.3.cross_attn.output_proj.bias` | B | (256,) |
| `decoder.layers.3.ffn.layers.0.0.weight` | B | (2048, 256) |
| `decoder.layers.3.ffn.layers.0.0.bias` | B | (2048,) |
| `decoder.layers.3.ffn.layers.1.weight` | B | (256, 2048) |
| `decoder.layers.3.ffn.layers.1.bias` | B | (256,) |
| `decoder.layers.3.norms.0.weight` | B | (256,) |
| `decoder.layers.3.norms.0.bias` | B | (256,) |
| `decoder.layers.3.norms.1.weight` | B | (256,) |
| `decoder.layers.3.norms.1.bias` | B | (256,) |
| `decoder.layers.3.norms.2.weight` | B | (256,) |
| `decoder.layers.3.norms.2.bias` | B | (256,) |
| `decoder.layers.3.norms.3.weight` | B | (256,) |
| `decoder.layers.3.norms.3.bias` | B | (256,) |
| `decoder.layers.4.self_attn.attn.in_proj_weight` | B | (768, 256) |
| `decoder.layers.4.self_attn.attn.in_proj_bias` | B | (768,) |
| `decoder.layers.4.self_attn.attn.out_proj.weight` | B | (256, 256) |
| `decoder.layers.4.self_attn.attn.out_proj.bias` | B | (256,) |
| `decoder.layers.4.cross_attn_text.attn.in_proj_weight` | B | (768, 256) |
| `decoder.layers.4.cross_attn_text.attn.in_proj_bias` | B | (768,) |
| `decoder.layers.4.cross_attn_text.attn.out_proj.weight` | B | (256, 256) |
| `decoder.layers.4.cross_attn_text.attn.out_proj.bias` | B | (256,) |
| `decoder.layers.4.cross_attn.sampling_offsets.weight` | B | (256, 256) |
| `decoder.layers.4.cross_attn.sampling_offsets.bias` | B | (256,) |
| `decoder.layers.4.cross_attn.attention_weights.weight` | B | (128, 256) |
| `decoder.layers.4.cross_attn.attention_weights.bias` | B | (128,) |
| `decoder.layers.4.cross_attn.value_proj.weight` | B | (256, 256) |
| `decoder.layers.4.cross_attn.value_proj.bias` | B | (256,) |
| `decoder.layers.4.cross_attn.output_proj.weight` | B | (256, 256) |
| `decoder.layers.4.cross_attn.output_proj.bias` | B | (256,) |
| `decoder.layers.4.ffn.layers.0.0.weight` | B | (2048, 256) |
| `decoder.layers.4.ffn.layers.0.0.bias` | B | (2048,) |
| `decoder.layers.4.ffn.layers.1.weight` | B | (256, 2048) |
| `decoder.layers.4.ffn.layers.1.bias` | B | (256,) |
| `decoder.layers.4.norms.0.weight` | B | (256,) |
| `decoder.layers.4.norms.0.bias` | B | (256,) |
| `decoder.layers.4.norms.1.weight` | B | (256,) |
| `decoder.layers.4.norms.1.bias` | B | (256,) |
| `decoder.layers.4.norms.2.weight` | B | (256,) |
| `decoder.layers.4.norms.2.bias` | B | (256,) |
| `decoder.layers.4.norms.3.weight` | B | (256,) |
| `decoder.layers.4.norms.3.bias` | B | (256,) |
| `decoder.layers.5.self_attn.attn.in_proj_weight` | B | (768, 256) |
| `decoder.layers.5.self_attn.attn.in_proj_bias` | B | (768,) |
| `decoder.layers.5.self_attn.attn.out_proj.weight` | B | (256, 256) |
| `decoder.layers.5.self_attn.attn.out_proj.bias` | B | (256,) |
| `decoder.layers.5.cross_attn_text.attn.in_proj_weight` | B | (768, 256) |
| `decoder.layers.5.cross_attn_text.attn.in_proj_bias` | B | (768,) |
| `decoder.layers.5.cross_attn_text.attn.out_proj.weight` | B | (256, 256) |
| `decoder.layers.5.cross_attn_text.attn.out_proj.bias` | B | (256,) |
| `decoder.layers.5.cross_attn.sampling_offsets.weight` | B | (256, 256) |
| `decoder.layers.5.cross_attn.sampling_offsets.bias` | B | (256,) |
| `decoder.layers.5.cross_attn.attention_weights.weight` | B | (128, 256) |
| `decoder.layers.5.cross_attn.attention_weights.bias` | B | (128,) |
| `decoder.layers.5.cross_attn.value_proj.weight` | B | (256, 256) |
| `decoder.layers.5.cross_attn.value_proj.bias` | B | (256,) |
| `decoder.layers.5.cross_attn.output_proj.weight` | B | (256, 256) |
| `decoder.layers.5.cross_attn.output_proj.bias` | B | (256,) |
| `decoder.layers.5.ffn.layers.0.0.weight` | B | (2048, 256) |
| `decoder.layers.5.ffn.layers.0.0.bias` | B | (2048,) |
| `decoder.layers.5.ffn.layers.1.weight` | B | (256, 2048) |
| `decoder.layers.5.ffn.layers.1.bias` | B | (256,) |
| `decoder.layers.5.norms.0.weight` | B | (256,) |
| `decoder.layers.5.norms.0.bias` | B | (256,) |
| `decoder.layers.5.norms.1.weight` | B | (256,) |
| `decoder.layers.5.norms.1.bias` | B | (256,) |
| `decoder.layers.5.norms.2.weight` | B | (256,) |
| `decoder.layers.5.norms.2.bias` | B | (256,) |
| `decoder.layers.5.norms.3.weight` | B | (256,) |
| `decoder.layers.5.norms.3.bias` | B | (256,) |
| `decoder.ref_point_head.layers.0.weight` | B | (256, 512) |
| `decoder.ref_point_head.layers.0.bias` | B | (256,) |
| `decoder.ref_point_head.layers.1.weight` | B | (256, 256) |
| `decoder.ref_point_head.layers.1.bias` | B | (256,) |
| `decoder.norm.weight` | B | (256,) |
| `decoder.norm.bias` | B | (256,) |
| `query_embedding.weight` | B | (900, 256) |
| `memory_trans_fc.weight` | B | (256, 256) |
| `memory_trans_fc.bias` | B | (256,) |
| `memory_trans_norm.weight` | B | (256,) |
| `memory_trans_norm.bias` | B | (256,) |
| `language_model.language_backbone.body.model.embeddings.word_embeddings.weight` | A | (30522, 768) |
| `language_model.language_backbone.body.model.embeddings.position_embeddings.weight` | A | (512, 768) |
| `language_model.language_backbone.body.model.embeddings.token_type_embeddings.weight` | A | (2, 768) |
| `language_model.language_backbone.body.model.embeddings.LayerNorm.weight` | A | (768,) |
| `language_model.language_backbone.body.model.embeddings.LayerNorm.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.0.attention.self.query.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.0.attention.self.query.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.0.attention.self.key.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.0.attention.self.key.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.0.attention.self.value.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.0.attention.self.value.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.0.attention.output.dense.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.0.attention.output.dense.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.0.attention.output.LayerNorm.weight` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.0.attention.output.LayerNorm.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.0.intermediate.dense.weight` | A | (3072, 768) |
| `language_model.language_backbone.body.model.encoder.layer.0.intermediate.dense.bias` | A | (3072,) |
| `language_model.language_backbone.body.model.encoder.layer.0.output.dense.weight` | A | (768, 3072) |
| `language_model.language_backbone.body.model.encoder.layer.0.output.dense.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.0.output.LayerNorm.weight` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.0.output.LayerNorm.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.1.attention.self.query.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.1.attention.self.query.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.1.attention.self.key.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.1.attention.self.key.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.1.attention.self.value.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.1.attention.self.value.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.1.attention.output.dense.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.1.attention.output.dense.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.1.attention.output.LayerNorm.weight` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.1.attention.output.LayerNorm.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.1.intermediate.dense.weight` | A | (3072, 768) |
| `language_model.language_backbone.body.model.encoder.layer.1.intermediate.dense.bias` | A | (3072,) |
| `language_model.language_backbone.body.model.encoder.layer.1.output.dense.weight` | A | (768, 3072) |
| `language_model.language_backbone.body.model.encoder.layer.1.output.dense.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.1.output.LayerNorm.weight` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.1.output.LayerNorm.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.2.attention.self.query.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.2.attention.self.query.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.2.attention.self.key.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.2.attention.self.key.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.2.attention.self.value.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.2.attention.self.value.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.2.attention.output.dense.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.2.attention.output.dense.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.2.attention.output.LayerNorm.weight` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.2.attention.output.LayerNorm.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.2.intermediate.dense.weight` | A | (3072, 768) |
| `language_model.language_backbone.body.model.encoder.layer.2.intermediate.dense.bias` | A | (3072,) |
| `language_model.language_backbone.body.model.encoder.layer.2.output.dense.weight` | A | (768, 3072) |
| `language_model.language_backbone.body.model.encoder.layer.2.output.dense.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.2.output.LayerNorm.weight` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.2.output.LayerNorm.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.3.attention.self.query.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.3.attention.self.query.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.3.attention.self.key.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.3.attention.self.key.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.3.attention.self.value.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.3.attention.self.value.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.3.attention.output.dense.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.3.attention.output.dense.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.3.attention.output.LayerNorm.weight` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.3.attention.output.LayerNorm.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.3.intermediate.dense.weight` | A | (3072, 768) |
| `language_model.language_backbone.body.model.encoder.layer.3.intermediate.dense.bias` | A | (3072,) |
| `language_model.language_backbone.body.model.encoder.layer.3.output.dense.weight` | A | (768, 3072) |
| `language_model.language_backbone.body.model.encoder.layer.3.output.dense.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.3.output.LayerNorm.weight` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.3.output.LayerNorm.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.4.attention.self.query.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.4.attention.self.query.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.4.attention.self.key.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.4.attention.self.key.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.4.attention.self.value.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.4.attention.self.value.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.4.attention.output.dense.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.4.attention.output.dense.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.4.attention.output.LayerNorm.weight` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.4.attention.output.LayerNorm.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.4.intermediate.dense.weight` | A | (3072, 768) |
| `language_model.language_backbone.body.model.encoder.layer.4.intermediate.dense.bias` | A | (3072,) |
| `language_model.language_backbone.body.model.encoder.layer.4.output.dense.weight` | A | (768, 3072) |
| `language_model.language_backbone.body.model.encoder.layer.4.output.dense.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.4.output.LayerNorm.weight` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.4.output.LayerNorm.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.5.attention.self.query.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.5.attention.self.query.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.5.attention.self.key.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.5.attention.self.key.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.5.attention.self.value.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.5.attention.self.value.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.5.attention.output.dense.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.5.attention.output.dense.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.5.attention.output.LayerNorm.weight` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.5.attention.output.LayerNorm.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.5.intermediate.dense.weight` | A | (3072, 768) |
| `language_model.language_backbone.body.model.encoder.layer.5.intermediate.dense.bias` | A | (3072,) |
| `language_model.language_backbone.body.model.encoder.layer.5.output.dense.weight` | A | (768, 3072) |
| `language_model.language_backbone.body.model.encoder.layer.5.output.dense.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.5.output.LayerNorm.weight` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.5.output.LayerNorm.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.6.attention.self.query.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.6.attention.self.query.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.6.attention.self.key.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.6.attention.self.key.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.6.attention.self.value.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.6.attention.self.value.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.6.attention.output.dense.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.6.attention.output.dense.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.6.attention.output.LayerNorm.weight` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.6.attention.output.LayerNorm.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.6.intermediate.dense.weight` | A | (3072, 768) |
| `language_model.language_backbone.body.model.encoder.layer.6.intermediate.dense.bias` | A | (3072,) |
| `language_model.language_backbone.body.model.encoder.layer.6.output.dense.weight` | A | (768, 3072) |
| `language_model.language_backbone.body.model.encoder.layer.6.output.dense.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.6.output.LayerNorm.weight` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.6.output.LayerNorm.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.7.attention.self.query.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.7.attention.self.query.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.7.attention.self.key.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.7.attention.self.key.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.7.attention.self.value.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.7.attention.self.value.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.7.attention.output.dense.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.7.attention.output.dense.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.7.attention.output.LayerNorm.weight` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.7.attention.output.LayerNorm.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.7.intermediate.dense.weight` | A | (3072, 768) |
| `language_model.language_backbone.body.model.encoder.layer.7.intermediate.dense.bias` | A | (3072,) |
| `language_model.language_backbone.body.model.encoder.layer.7.output.dense.weight` | A | (768, 3072) |
| `language_model.language_backbone.body.model.encoder.layer.7.output.dense.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.7.output.LayerNorm.weight` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.7.output.LayerNorm.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.8.attention.self.query.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.8.attention.self.query.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.8.attention.self.key.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.8.attention.self.key.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.8.attention.self.value.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.8.attention.self.value.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.8.attention.output.dense.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.8.attention.output.dense.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.8.attention.output.LayerNorm.weight` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.8.attention.output.LayerNorm.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.8.intermediate.dense.weight` | A | (3072, 768) |
| `language_model.language_backbone.body.model.encoder.layer.8.intermediate.dense.bias` | A | (3072,) |
| `language_model.language_backbone.body.model.encoder.layer.8.output.dense.weight` | A | (768, 3072) |
| `language_model.language_backbone.body.model.encoder.layer.8.output.dense.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.8.output.LayerNorm.weight` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.8.output.LayerNorm.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.9.attention.self.query.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.9.attention.self.query.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.9.attention.self.key.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.9.attention.self.key.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.9.attention.self.value.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.9.attention.self.value.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.9.attention.output.dense.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.9.attention.output.dense.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.9.attention.output.LayerNorm.weight` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.9.attention.output.LayerNorm.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.9.intermediate.dense.weight` | A | (3072, 768) |
| `language_model.language_backbone.body.model.encoder.layer.9.intermediate.dense.bias` | A | (3072,) |
| `language_model.language_backbone.body.model.encoder.layer.9.output.dense.weight` | A | (768, 3072) |
| `language_model.language_backbone.body.model.encoder.layer.9.output.dense.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.9.output.LayerNorm.weight` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.9.output.LayerNorm.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.10.attention.self.query.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.10.attention.self.query.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.10.attention.self.key.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.10.attention.self.key.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.10.attention.self.value.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.10.attention.self.value.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.10.attention.output.dense.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.10.attention.output.dense.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.10.attention.output.LayerNorm.weight` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.10.attention.output.LayerNorm.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.10.intermediate.dense.weight` | A | (3072, 768) |
| `language_model.language_backbone.body.model.encoder.layer.10.intermediate.dense.bias` | A | (3072,) |
| `language_model.language_backbone.body.model.encoder.layer.10.output.dense.weight` | A | (768, 3072) |
| `language_model.language_backbone.body.model.encoder.layer.10.output.dense.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.10.output.LayerNorm.weight` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.10.output.LayerNorm.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.11.attention.self.query.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.11.attention.self.query.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.11.attention.self.key.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.11.attention.self.key.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.11.attention.self.value.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.11.attention.self.value.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.11.attention.output.dense.weight` | A | (768, 768) |
| `language_model.language_backbone.body.model.encoder.layer.11.attention.output.dense.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.11.attention.output.LayerNorm.weight` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.11.attention.output.LayerNorm.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.11.intermediate.dense.weight` | A | (3072, 768) |
| `language_model.language_backbone.body.model.encoder.layer.11.intermediate.dense.bias` | A | (3072,) |
| `language_model.language_backbone.body.model.encoder.layer.11.output.dense.weight` | A | (768, 3072) |
| `language_model.language_backbone.body.model.encoder.layer.11.output.dense.bias` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.11.output.LayerNorm.weight` | A | (768,) |
| `language_model.language_backbone.body.model.encoder.layer.11.output.LayerNorm.bias` | A | (768,) |
| `text_feat_map.weight` | A | (256, 768) |
| `text_feat_map.bias` | A | (256,) |
| `dn_query_generator.label_embedding.weight` | B | (28, 256) |
