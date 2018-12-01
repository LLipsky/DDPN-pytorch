import torch.nn as nn
from config.base_config import cfg
import json
from data_provider.data_factory import get_data_provider
import numpy as np
import functools
from utils.data_utils import complete_data
class DataProviderLayer(nn.Module):  # succeed the torch.nn.Modules
    def __init__(self, bottom, top, param_str_):  # top means output,bottom means input

        super(DataProviderLayer, self).__init__()
        self.bottomup_feat_dim = cfg.BOTTOMUP_FEAT_DIM
        self.query_maxlen = cfg.QUERY_MAXLEN
        self.split = json.loads(param_str_)['split']
        self.batchsize = json.loads(param_str_)['batchsize']
        self.use_kld = cfg.USE_KLD


        self.top_names = ['qvec', 'cvec', 'img_feat', 'spt_feat', 'query_label', 'query_label_mask',\
                        'query_bbox_targets', 'query_bbox_inside_weights', 'query_bbox_outside_weights']


        '''
        top[0].reshape(self.query_maxlen, self.batchsize)  # reshape相当于重新指定维度
        # print('00000000000000000000000000000has done')
        top[1].reshape(self.query_maxlen, self.batchsize)  # 在pytorch中一般使用view比较多，详情见https://stackoverflow.com/questions/49643225/whats-the-difference-between-reshape-and-view-in-pytorch
        top[2].reshape(self.batchsize, cfg.RPN_TOPN, self.bottomup_feat_dim)
        top[3].reshape(self.batchsize, cfg.RPN_TOPN, 5)

        if self.use_kld:#这里根据use_kld的取值不同，为什么reshape成的形状不同？？？？？?????
            top[4].reshape(self.batchsize, cfg.RPN_TOPN)
        else:
            top[4].reshape(self.batchsize)
        top[5].reshape(self.batchsize)
        top[6].reshape(self.batchsize*cfg.RPN_TOPN, 4)
        top[7].reshape(self.batchsize*cfg.RPN_TOPN, 4)
        top[8].reshape(self.batchsize*cfg.RPN_TOPN, 4)'''

        tmp = np.zeros((self.query_maxlen, self.batchsize))  # qvec
        print(tmp.shape)
        top.append(tmp)  # top[0]
        print(np.array(top[0]).shape)

        tmp = np.zeros((self.query_maxlen, self.batchsize))  # cvec
        top.append(tmp)  # top[1]

        tmp = np.zeros((self.batchsize, cfg.RPN_TOPN, self.bottomup_feat_dim))  # img_feat
        top.append(tmp)  # top[2]

        tmp = np.zeros((self.batchsize, cfg.RPN_TOPN, 5))  # spt_feat
        top.append(tmp)  # top[3]

        if self.use_kld:
            tmp = np.zeros((self.batchsize, cfg.RPN_TOPN, 5))  # query_label
            top.append(tmp)  # top[4]
        else:
            tmp = np.zeros((self.batchsize))
            top.append(tmp)  # top[4]

        tmp = np.zeros((self.batchsize))  #query_label_mask  (positive or negative)
        top.append(tmp)  # top[5]

        tmp = np.zeros((self.batchsize*cfg.RPN_TOPN, 4))  # query_bbox_targets
        top.append(tmp)  # top[6]

        tmp = np.zeros((self.batchsize*cfg.RPN_TOPN, 4))  # query_bbox_inside_weigths
        top.append(tmp)  # top[7]

        tmp = np.zeros((self.batchsize * cfg.RPN_TOPN, 4))  # query_bbox_outside_weights
        top.append(tmp)  # top[8]

        if str(self.phase) == 'TRAIN':
            dp = get_data_provider(data_split=self.split, batchsize=self.batchsize)  # call the MultiDataProvider(data_split, batchsize)
            if cfg.NTHREADS > 1:
                import torch
                self.dataloader = torch.utils.data.DataLoader(dp, batch_size=self.batchsize, shuffle=True, num_workers=int(cfg.NTHREADS))
            else:
                self.dataloader = dp

            self.data_iter = iter(self.dataloader)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):  # input->operation->forward,return output
        if str(self.phase) != 'TRAIN':
            return
        try:
            next_data = self.data_iter.next()
        except:
            self.data_iter = iter(self.dataloader)
            next_data = self.data_iter.next()

        next_data = map(np.array, next_data)
        my_complete_data = functools.partial(complete_data, batchsize=self.batchsize)  # functools.partial通过包装手法，允许我们“重新定义”函数签名,，可以像原始对象一样对待
        #map(function,iterable,...)
        gt_boxes, qvec, cvec, img_feat, bbox, img_shape, spt_feat, query_label, query_label_mask, \
                        query_bbox_targets, query_bbox_inside_weights, query_bbox_outside_weights, valid_data, iid_list = map(my_complete_data,next_data)

        #queries
        qvec = np.transpose(qvec, (1, 0))
        top[0].reshape(*qvec.shape)
        top[0].data[...] = qvec

        cvec = np.transpose(cvec,(1,0))
        top[1].reshape(*cvec.shape)
        top[1].data[...] = cvec

        top[2].reshape(*img_feat.shape)
        top[2].data[...] = img_feat

        top[3].reshape(*spt_feat.shape)
        top[3].data[...] = spt_feat

        top[4].reshape(*query_label.reshape)
        top[4].data[...] = query_label

        #query_label_mask
        top[5].reshape(*query_label_mask.shape)
        top[5].data[...] = query_label_mask

        #bbox regression
        query_bbox_targets = query_bbox_targets.reshape(-1, 4)
        top[6].reshape(*query_bbox_targets.shape)
        top[6].data[...] = query_bbox_targets

        query_bbox_inside_weights = query_bbox_inside_weights.reshape(-1, 4)
        top[7].reshape(*query_bbox_inside_weights.shape)
        top[7].data[...] = query_bbox_inside_weights

        query_bbox_outside_weights = query_bbox_outside_weights.reshape(-1, 4)
        top[8].reshape(*query_bbox_outside_weights.shape)
        top[8].data[...] = query_bbox_outside_weights

    def backward(self, top, propagate_down, bottom):
        pass







