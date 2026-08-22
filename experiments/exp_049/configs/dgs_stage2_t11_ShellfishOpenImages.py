# =============================================================================
# exp_049: DGS stage2 / t=11 ShellfishOpenImages
# 【自動生成】experiments/exp_049/gen_configs.py。直接編集しない。
# ドライバが渡すのは load_from（θ_{t-1}）だけ。
# =============================================================================
_base_ = './dgs_stage2.py'

model = dict(
    num_tasks=13,
    task_id=10,
    seen_tasks='AerialMaritimeDrone,Aquarium,CottontailRabbits,EgoHands,NorthAmericaMushroom,Packages,PascalVOC,pistols,pothole,Raccoon,ShellfishOpenImages',
    domain_predictor_cfg=dict(task_id_mapping_path='experiments/exp_049/work_dirs/task_id_mapping.yaml'))

train_dataloader = dict(
    dataset=dict(
        metainfo='ShellfishOpenImages',
        data_root='data/odinw/ShellfishOpenImages/416x416/',
        ann_file='train/annotations_without_background.json',
        data_prefix=dict(img='train/'),
        seen_tasks='AerialMaritimeDrone,Aquarium,CottontailRabbits,EgoHands,NorthAmericaMushroom,Packages,PascalVOC,pistols,pothole,Raccoon,ShellfishOpenImages',
        distn_cfg=dict(task_id_mapping_path='experiments/exp_049/work_dirs/task_id_mapping.yaml')))
