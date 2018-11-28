# import torch
import torch.nn as nn
from config.base_config import cfg
from utils.dictionary import Dictionary
# import numpy as np
import json
from networks.data_layer import DataProviderLayer
import sys
import argparse

#the class maybe in the models.py,include load data,initialize the params,then divide it into many files
class Net(nn.Module):
    def __init__(self, split, vocab_size):
    # def __init__(self, vocab_size):
        super(Net, self).__init__()
        # self.dict_dir = cfg.QUERY_DIR
        # self.qdic = Dictionary(self.dict_dir)
        # self.vocab_size = self.qdic.size()
        # print(vocab_size)  # vocab_size supposed to be 9368;
        #
        # self.query_maxlen = cfg.QUERY_MAXLEN
        #
        # qvec = np.zeros(self.query_maxlen)
        # cvec = np.zeros(self.query_maxlen)

        self.split = split
        self.vocab_size = vocab_size
        print("hahahhahahh")

        top = []
        param_str = json.dumps({'split': self.split, 'batchsize': cfg.BATCHSIZE})

        dataProviderLayer = DataProviderLayer(top, param_str)
        # print("fafdafadfafhasfjasdfkasjfasfas")




        self.qvec, self.cvec, self.img_feat, self.spt_feat, self.query_label, \
        self.query_label_mask, self.query_bbox_targets, self.query_bbox_inside_weights, \
        self.query_bbox_outside_weights = dataProviderLayer()  # don't consider para_str


    def forward(self, *input):
        print()


def imageLoader():
    print()


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

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    opts = parser.parse_args()

    return opts

#the main function may in the train_net.py
#main function may in the train_net.py,for simple just put in the same file
if __name__ == '__main__':

    qdic = qdicLoader()

    vocab_size = qdic.size()
    print(vocab_size)#vocab_size supposed to be 9368;

    opts = parse_args()  # cannot return a opts,because len==1 and exit(1)

    train_net = Net(opts.train_split, vocab_size)







