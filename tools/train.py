# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()
    # print("args.config: ", args.config)   # args.config:  configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_cat.py

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()


    #--------- 学習設定（config）の確定 ---------
    cfg = Config.fromfile(args.config)          # configファイルの読み込み
    # print("cfg: ", cfg)

    cfg.launcher = args.launcher                # 起動方式を設定に反映
    if args.cfg_options is not None:            # コマンドラインからの上書き指定があれば，configの内容を上書き
        cfg.merge_from_dict(args.cfg_options)


    #--------- 作業ディレクトリ（work_dir）を決定 ---------
    # CLI（コマンドライン引数） ＞ segment in file（configファイル内のwork_dir設定） ＞ filename（configファイルの名前から自動で設定）
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])


    #--------- AMP(Automatic Mixed Precision，自動混合精度)の設定 ---------
    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'      # Optimizerを混合精度対応版に差し替え
        cfg.optim_wrapper.loss_scale = 'dynamic'        # loss scalingを動的に調整


    #--------- auto-scale-lr（学習率の自動スケーリング）の設定 ---------
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True     # 自動スケーリングを有効化
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    #--------- 中断した学習を，チェックポイントから再開（resume）するための設定処理 ---------
    if args.resume == 'auto':           # --resume だけ指定（値なし） →　自動再開
        cfg.resume = True
        cfg.load_from = None            # work_dir 内の最新チェックポイントから再開
    elif args.resume is not None:       # --resume にパス指定 → 指定再開
        cfg.resume = True
        cfg.load_from = args.resume     # 指定したチェックポイントから再開

    #--------- config から Runner（実行器） を構築する処理 ---------
    # print("'runner_type' not in cfg: ", 'runner_type' not in cfg)   # 'runner_type' not in cfg:  True
    
    if 'runner_type' not in cfg:        # config に runner_type の指定がない
        
        # config ファイルを読み込んで Runner を作成
        runner = Runner.from_cfg(cfg)   # 標準の Runner を構築
    else:                               # config に runner_type の指定がある
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)     # 指定された Runner を構築
    
    # print("runner: ", runner)   # runner:  <mmengine.runner.runner.Runner object at 0x7f9afac0b9a0>

    #--------- runner を使って学習を実行 ---------
    # /opt/conda/lib/python3.10/site-packages/mmengine/runner/runner.py
    runner.train()


if __name__ == '__main__':
    main()
