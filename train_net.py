import sys
import argparse
from config.base_config import cfg
from utils.dictionary import Dictionary
#the main function may in the train_net.py
#main function may in the train_net.py,for simple just put in the same file
from networks.model_pytorch import Net
def qdicLoader():
    qdic_dir = cfg.QUERY_DIR
    qdic = Dictionary(qdic_dir)
    qdic.load()

    return qdic

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a vg network')
    parser.add_argument('--randomize', help='randomize', default=None, type=int)
    parser.add_argument('--gpu_id', help='gpu_id', default=0, type=int)
    parser.add_argument('--train_split', help='train_split', default='train', type=str)
    parser.add_argument('--val_split', help='val_split', default='val', type=str)
    parser.add_argument('--vis_pred', help='visualize prediction', default=False, type=bool)
    parser.add_argument(
            '--pretrained_model',
            help='pretrained_model',
            default=None, #osp.join(get_models_dir(''), '_iter_25000.caffemodel'),
            type=str
    )
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='optional config file',
        default='config/experiments/refcoco-kld-bbox_reg.yaml',
        type=str
    )

    # to avoid the len==1,can set the default ==specified number,such as gpu_id 0
    # if len(sys.argv) == 1:  # sys.argv has ????????
    #     parser.print_help()
    #     sys.exit(1)

    opts = parser.parse_args()

    return opts

if __name__ == '__main__':

    qdic = qdicLoader()

    vocab_size = qdic.size()
    print(vocab_size)  # vocab_size supposed to be 9368;

    opts = parse_args()  # cannot return a opts,because len==1 and exit(1)

    print(opts.train_split)

    train_net = Net(opts.train_split, vocab_size)  # train_split value == train