from easydict import EasyDict as edict#EasyDict可以让你像访问属性一样访问dict里的变量
import os.path as osp#os.path 模块主要用于获取文件的属性。
import yaml
import numpy as np
__C=edict()

cfg=__C


__C.BATCHSIZE=64
__C.BOTTOMUP_FEAT_DIM=2048

__C.QUERY_MAXLEN=15

__C.USE_KLD=True

__C.RPN_TOPN=100
__C.NTHREADS=16
__C.IMDB_NAME='refcoco'# flickr30k, referit, refcoco, refcoco+
__C.PROJ_NAME='genome'
# if __C.USE_KLD:
#     __C.PROJ_NAME+='_kld'
# else:
#     __C.PROJ_NAME+='_soft'

__C.FEAT_TYPE='bottom-up'# ss, wfrpn, wofrpn, vgrpn, bu
__C.ROOT_DIR=osp.abspath(osp.join(osp.dirname(__file__),'..'))#osp.dirname返回文件路径；osp.abspath返回绝对路径;__file__是获取当前脚本的路径
__C.DATA_DIR=osp.join(__C.ROOT_DIR,'data')
__C.SS_BOX_DIR=osp.join(__C.DATA_DIR,'ss_box')
__C.SS_FEAT_DIR=osp.join(__C.SS_BOX_DIR,'ss_feat_vgg_det')
__C.SS_FEAT_DIM=4096

__C.UNK=1#这个参数具体代表什么？？？？？

__C.SPLIT_TOK='+'

__C.IMAGE_DIR=osp.join(__C.DATA_DIR,'mscoco/features/fst-res101-feats/train2014')#这里到时候应该根据自己存放的具体路径进行修改！！！！
__C.FEAT_DIR=osp.join(__C.DATA_DIR,'mscoco/image2014/train2014')
__C.QUERY_DIR=osp.join(__C.DATA_DIR,'query_dict')
__C.ANNO_PATH=osp.join(__C.DATA_DIR,'format_%s.pkl')

__C.THRESHOLD=0.5

__C.BBOX_NORMALIZE_TARGETS_PRECOMPUTED=True
__C.BBOX_NORMALIZE_STDS=(0.1,0.1,0.2,0.2)
__C.BBOX_INSIDE_WEIGHTS=(1.0,1.0,1.0,1.0)

__C.WORD_EMB_SIZE=300
__C.RNN_DIM=1024
__C.DROPOUT_RATIO=0.3
__C.USE_REG=True

def print_cfg():
    print('imdb name: %s'%__C.IMDB_NAME)
    print ('feat type: %s'%__C.FEAT_TYPE)
    print('proj name: %s'%__C.PROJ_NAME)

def merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    with open(filename,'r') as f :
        yaml_cfg=edict(yaml.load(f))

    merge_a_into_b(yaml_cfg, __C)



