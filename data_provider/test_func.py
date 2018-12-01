# import torch
# x=torch.tensor([1,2,3])#axis=0代表列；axis=1代表行
#
# print(x)
# print(x.repeat(4,2))#猜测前面的4代表的是行repeat的次数；后者的2代表的是列repeat的次数。
# print(x.repeat(4,2).size())
#
# print(x.repeat(3,2))
# print(x.repeat(3,2).size())
#
# print(x.repeat(3,1))
# print(x.repeat(3,1).size())
#
# print(x.repeat(4,3))
#
# print(x.repeat(4,2,1))
#
# print('hahahh')
# print(x.repeat(4,1))
# print(x.repeat(4))
# print(x.repeat(1,4))
#
# import numpy as np
# a=np.array([1,2,3])
# print(np.tile(a,2).shape)
# print(x.repeat(2).size())
#
import _pickle as cPickle
import os.path as osp  # os.path 模块主要用于获取文件的属性。
import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
import codecs
def load(file_path):
    with open(file_path, 'rb') as f:
    # with codecs.open(file_path, 'rb', encoding='utf-8') as f:
        print(f)
        return cPickle.load(f, encoding='latin1')  # unicode error solved by latin1


ROOT_DIR = osp.join('/home/lipin/code/DDPN-master')
DATA_DIR = osp.join(ROOT_DIR, 'data')
ANNO_PATH = osp.join(DATA_DIR, 'format_dataset/refcoco/format_train.pkl')
print(ANNO_PATH)
# file_path = '/code/DDPN-master/data/format_dataset/refcoco'
train_img_data = load(ANNO_PATH)

print(train_img_data)