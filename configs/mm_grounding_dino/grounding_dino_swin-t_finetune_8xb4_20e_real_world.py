_base_ = 'grounding_dino_swin-t_pretrain_obj365.py'
data_root = '/workspace/kouyou/datasets/rf100_domain/real world/'
class_name = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
     '13', 'smoke', 'Minorrotation', 'Moderaterotation', 'Severerotation',
     'Slippage', 'corrosion', 'crack', 'EXCAVATORS', 'dump truck',
     'wheel loader', 'bishop', 'black-bishop', 'black-king', 'black-knight',
     'black-pawn', 'black-queen', 'black-rook', 'white-bishop', 'white-king',
     'white-knight', 'white-pawn', 'white-queen', 'white-rook', 'bus_stop',
     'do_not_enter', 'do_not_stop', 'do_not_turn_l', 'do_not_turn_r',
     'do_not_u_turn', 'enter_left_lane', 'green_light', 'left_right_lane',
     'no_parking', 'parking', 'ped_crossing', 'ped_zebra_cross',
     'railway_crossing', 'red_light', 'stop', 't_intersection_l',
     'traffic_light', 'u_turn', 'warning', 'yellow_light', 'Cone',
     'Face_Shield', 'Gloves', 'Goggles', 'Head', 'Helmet', 'No glasses',
     'No gloves', 'helmet', 'no-helmet', 'no-vest', 'person', 'vest',
     'bicycles', 'buses', 'crosswalks', 'fire hydrants', 'motorcycles',
     'traffic lights', 'vehicles', 'bathtub', 'c', 'geyser', 'mirror',
     'showerhead', 'sink', 'toilet', 'towel', 'washbasin', 'wc', 'Button',
     'Buzzer', 'Capacitor', 'Capacitor Jumper', 'Capacitor Network', 'Clock',
     'Connector', 'Diode', 'EM', 'Electrolytic Capacitor',
     'Electrolytic capacitor', 'Ferrite Bead', 'Flex Cable', 'Fuse', 'IC',
     'Inductor', 'Jumper', 'Led', 'Pads', 'Pins', 'Potentiometer', 'RP',
     'Resistor', 'Resistor Jumper', 'Resistor Network', 'Switch',
     'Test Point', 'Transducer', 'Transformer', 'Transistor',
     'Unknown Unlabeled', 'mask', 'no-mask', 'Antenne', 'BBS', 'BFU',
     'Batterie', 'DDF', 'PCF', 'PCU AC', 'PCU DC', 'PDU', 'PSU', 'RBS',
     'coca-cola', 'fanta', 'sprite', 'otr_chassis_loaded',
     'otr_chassis_unloaded', 'otr_chassis_working', 'stacker',
     'AlcoholPercentage', 'Appellation AOC DOC AVARegion',
     'Appellation QualityLevel', 'CountryCountry', 'Distinct Logo',
     'Established YearYear', 'Maker-Name', 'Organic', 'Sustainable',
     'Sweetness-Brut-SecSweetness-Brut-Sec', 'TypeWine Type', 'VintageYear',
     'big bus', 'big truck', 'bus-l-', 'bus-s-', 'car', 'mid truck',
     'small bus', 'small truck', 'truck-l-', 'truck-m-', 'truck-s-',
     'truck-xl-', 'with mold', 'without mold', 'iC', 'Agrotis',
     'Athetis lepigone', 'Athetis lineosa', 'Chilo suppressalis',
     'Cnaphalocrocis medinalis Guenee', 'Creatonotus transiens',
     'Diaphania indica', 'Endotricha consocia', 'Euproctis sparsa',
     'Gryllidae', 'Gryllotalpidae', 'Helicoverpa armigera',
     'Holotrichia oblita Faldermann', 'Loxostege sticticalis',
     'Mamestra brassicae', 'Maruca testulalis Geyer', 'Mythimna separata',
     'Naranga aenescens Moore', 'Nilaparvata', 'Paracymoriza taiwanalis',
     'Sesamia inferens', 'Sirthenea flavipes', 'Sogatella furcifera',
     'Spodoptera exigua', 'Spoladea recurvalis', 'Staurophora celsia',
     'Timandra Recompta', 'Trichoptera', 'cavity', 'normal', 'mildew',
     'rose_P01', 'rose_R02', 'red', 'white', 'Cipro 500', 'Ibuphil 600 mg',
     'Ibuphil Cold 400-60', 'Xyzall 5mg', 'blue', 'pink', '59',
     '10 Diamonds', '10 Hearts', '10 Spades', '10 Trefoils', '2 Diamonds',
     '2 Hearts', '2 Spades', '2 Trefoils', '3 Diamonds', '3 Hearts',
     '3 Spades', '3 Trefoils', '4 Diamonds', '4 Hearts', '4 Spades',
     '4 Trefoils', '5 Diamonds', '5 Hearts', '5 Spades', '5 Trefoils',
     '6 Diamonds', '6 Hearts', '6 Spades', '6 Trefoils', '7 Diamonds',
     '7 Hearts', '7 Spades', '7 Trefoils', '8 Diamonds', '8 Hearts',
     '8 Spades', '8 Trefoils', '9 Diamonds', '9 Hearts', '9 Spades',
     '9 Trefoils', 'A Diamonds', 'A Hearts', 'A Spades', 'A Trefoils',
     'J Diamonds', 'J Hearts', 'J Spades', 'J Trefoils', 'K Diamonds',
     'K Hearts', 'K Spades', 'K Trefoils', 'Q Diamonds', 'Q Hearts',
     'Q Spades', 'Q Trefoils', 'div', 'eqv', 'minus', 'mult', 'plus',
     'army worm', 'legume blister beetle', 'red spider', 'rice gall midge',
     'rice leaf roller', 'rice leafhopper', 'rice water weevil',
     'wheat phloeothrips', 'white backed plant hopper', 'yellow rice borer',
     'G-arboreum', 'G-barbadense', 'G-herbaceum', 'G-hirsitum', 'Chair',
     'Sofa', 'Table', 'break', 'thunderbolt', 'cat', 'chicken', 'cow', 'dog',
     'fox', 'goat', 'horse', 'racoon', 'skunk', 'coin', 'nail', 'nut',
     'screw', 'apple', 'damaged_apple', 'Human', 'GND', 'IDC', 'IDC_I', 'R',
     'VDC', 'VDC_I', '14', '0 ridderzuring', 'gauges', 'numbers', 'A', 'B',
     'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
     'Q', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'sees-dark-almond-nougat',
     'sees-dark-almonds', 'sees-dark-bordeaux', 'sees-dark-caramel-patties',
     'sees-dark-chocolate-buttercream', 'sees-dark-marzipan',
     'sees-dark-normandie', 'sees-dark-scotchmallow',
     'sees-dark-walnut-square', 'sees-milk-almond-caramel',
     'sees-milk-almonds', 'sees-milk-beverly', 'sees-milk-bordeaux',
     'sees-milk-butterscotch-square', 'sees-milk-california-brittle',
     'sees-milk-chelsea', 'sees-milk-chocolate-buttercream',
     'sees-milk-coconut-cream', 'sees-milk-mayfair', 'sees-milk-mocha',
     'sees-milk-molasses-chips', 'sees-milk-rum-nougat', 'aair', 'boal',
     'chapila', 'deshi puti', 'foli', 'ilish', 'kal baush', 'katla', 'koi',
     'magur', 'mrigel', 'pabda', 'pangas', 'puti', 'rui', 'shol', 'taki',
     'tara baim', 'telapiya', 'Ready', 'empty_pod', 'germination', 'pod',
     'young', 'Lower', 'Sand Tiger Shark', 'Snaggletooth Shark', 'Upper',
     'bees', 'Cross bedding', 'Low angle', 'Massive', 'Parallel lamination',
     'mud drape', 'Dime', 'Nickel', 'Penny', 'Quarter', 'fifty', 'five',
     'hundred', 'one', 'ten', 'twenty', 'Deer', 'Hog', 'joint', 'side')
num_classes = len(class_name)
metainfo = dict(classes=class_name)
model = dict(bbox_head=dict(num_classes=num_classes))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]
train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        return_classes=True,
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        ann_file='train/_annotations.coco.json',
        data_prefix=dict(img='train/')))
val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/')))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=data_root + 'valid/_annotations.coco.json')
test_evaluator = val_evaluator
max_epoch = 20
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epoch, val_interval=1)
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epoch,
        by_epoch=True,
        milestones=[15],
        gamma=0.1)
]
optim_wrapper = dict(
    optimizer=dict(lr=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.0),
            'language_model': dict(lr_mult=0.0)
        }))
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'  # noqa