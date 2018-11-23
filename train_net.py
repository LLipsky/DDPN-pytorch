import argparse
import sys
from config.base_config import cfg_from_file,print_cfg
import os.path as osp
import config.base_config as cfg

def get_solver(opts):
    # s = caffe_pb2.SolverParameter()
    train_net =   osp.join(opts.train_net_path)
    snapshot =    0
    snapshot_prefix = cfg.TRAIN.SNAPSHOT_PREFIX
    max_iter =    cfg.TRAIN.MAX_ITERS
    display =     cfg.TRAIN.DISPLAY
    average_loss= 100
    type =        cfg.TRAIN.TYPE
    stepsize =    cfg.TRAIN.STEPSIZE
    gamma =       cfg.TRAIN.GAMMA
    lr_policy =   cfg.TRAIN.LR_POLICY
    base_lr =     cfg.TRAIN.LR
    momentum =    cfg.TRAIN.MOMENTUM
    momentum2 =   cfg.TRAIN.MOMENTUM2
    iter_size =   cfg.TRAIN.ITER_SIZE
    # s.weight_decay = 0.0005
    # s.clip_gradients = 10
    return s

def parse_args():
    """
    Parse input arguments
    """
    #argparse 是 Python 内置的一个用于命令项选项与参数解析的模块，
    #通过在程序中定义好我们需要的参数，argparse 将会从 sys.argv 中解析出这些参数，并自动生成帮助和使用信息。
    parser=argparse.ArgumentParser(description='Train a vg network')#创建 ArgumentParser() 对象

    parser.add_argument('--randomize',help='randomize',default=None,type=int)#调用 add_argument() 方法添加参数
    parser.add_argument('--gpu_id',help='gpu_id', default=0, type=int)
    parser.add_argument('--train_split', help='train_split', default='train', type=str)
    parser.add_argument('--val_split', help='val_split', default='val', type=str)
    parser.add_argument('--vis_pred', help='visualize prediction', default=False, type=bool)
    parser.add_argument(
                '--pretrained_model',
                help='pretrained_model',
                default= None, #osp.join(get_models_dir(''), '_iter_25000.caffemodel'),
                type=str
        )
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='optional config file',
        #default='config/experiments/refcoco-kld-bbox_reg.yaml',
        type=str
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    opts = parser.parse_args()#使用 parse_args() 解析添加的参数
    return opts


if __name__=='__main__':
    opts=parse_args()

    if opts.cfg_file is not None:
        cfg_from_file(opts.cfg_file)

    print_cfg()


