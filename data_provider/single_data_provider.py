import torch.nn as nn
from data_provider.data_provider import DataProvider
# from data_provider.data_factory import get_data_provider
import numpy as np
import random
class SingleDataProvider(DataProvider):
    def __init__(self,data_split,batchsize=1):
        DataProvider.__init__(self,data_split,batchsize)

    def create_batch(self,qid_list):
        gt_bbox=np.zeros(self.batchsize,4)
        qvec=(np.zeros(self.batchsize*self.query_maxlen)).reshape(self.batchsize,self.query_maxlen)#batchsize*query_maxlen矩阵64，15
        cvec=(np.zeros(self.batchsize*self.query_maxlen)).reshape(self.batchsize,self.query_maxlen)#batchsize*query_maxlen矩阵64，15
        img_feat=np.zeros((self.batchsize,self.rpn_topn,self.bottomup_feat_dim))#64，100，2048

        bbox=np.zeros((self.batchsize,self.rpn_topn,4))#64，100，4
        img_shape=np.zeros((self.batchsize,2))#64，2

        spt_feat=np.zeros((self.batchsize,self.rpn_topn,5))#64，100，5
        if self.use_kld:
            query_label=np.zeros((self.batchsize,self.rpn_topn))#64，100
        else:
            query_label=np.zeros((self.batchsize))#64

        query_label_mask=np.zeros((self.batchsize))#64
        query_bbox_targets=np.zeros((self.batchsize,self.rpn_topn,4))#64，100，4
        query_bbox_inside_weights=np.zeros((self.batchsize,self.rpn_topn,4))#64，100，4
        query_bbox_outside_weights=np.zeros((self.batchsize,self.rpn_topn,4))#64，100，4
        valid_data=np.ones(self.batchsize)#64

        for i,qid in enumerate(qid_list):
            t_qstr=self.anno[qid]['qstr']#这个语句的意思是将anno中的数据中的qstr部分提取，是吗？？？？？/?
            t_qvec,t_cvec=self.str2list(t_qstr,self.query_maxlen)
            qvec[i,...]=t_qstr#python的多维切片，三个点是用来省略所有的冒号来用省略号代替a[:,:,None]和a[...,None],注这里的None相当于newaxis
            cvec[i,...]=t_cvec

            try:
                t_gt_bbox=self.anno[qid]['boxes']
                gt_bbox[i,...]=t_gt_bbox[0]
                t_img_feat, t_num_bbox, t_bbox,t_img_shape=self.get_topdown_feat(self.anno[qid]['iid'])
                t_img_feat=t_img_feat.transpose((1,0))#比如是a = np.array([[1, 2], [3, 4]]);a.transpose((1,0))是 array([[1, 3],[2, 4]])；a.transpose(1,0)同上
                t_img_feat=(t_img_feat/np.sqrt((t_img_feat**2).sum()))#将t_img_feat/(t_img_feat的求和再开方)

                img_feat[:,:t_num_bbox,:]=t_img_feat
                bbox[:,:t_num_bbox,:]=t_bbox

                #spt feat
                img_shape[i,:]=np.array(t_img_shape)
                t_spt_feat=self.get_spt_feat(t_bbox,t_img_shape)#5-D
                spt_feat[i,:t_num_bbox,:]=t_spt_feat

                #query label,mask
                t_gt_bbox=np.array(self.anno[qid]['boxes'])
                t_query_label, t_query_label_mask, t_query_bbox_targets, t_query_bbox_inside_weights, t_query_bbox_outside_weights=\
                    self.get_labels(self, t_bbox, t_gt_bbox)

                if self.use_kld:
                    query_label[i, :t_num_bbox] = t_query_label
                    query_label_mask[i] = t_query_label_mask
                else:
                    query_label[i, ...] = t_query_label

                query_bbox_targets[:, :t_num_bbox, :] = t_query_bbox_inside_weights
                t_query_bbox_outside_weights[:, :t_num_bbox, :] = t_query_bbox_outside_weights

            except Exception as e:
                print(e)
                valid_data[i] = 0
                if not self.use_kld:
                    query_label[i] = -1
                query_label_mask[i] = 0
                query_bbox_inside_weights[i, ...] = 0
                query_bbox_outside_weights[i, ...] = 0
                print('data not found for iid: %s' % str(self.anno[qid]['iid']))

        return gt_bbox, qvec, cvec, img_feat, bbox, img_feat, spt_feat, query_label, query_label_mask,\
                    query_bbox_targets, query_bbox_inside_weights, query_bbox_outside_weights, valid_data

    def __getitem__(self, index):
        if self.batch_len is None:
            self.n_skipped=0
            qid_list=self.get_query_ids()
            if self.mode=='train':
                random.shuffle(qid_list)
            self.qid_list=qid_list
            self.batch_len=len(qid_list)
            self.batch_index=0
            self.epoch_counter=0
            print('mode %s has %d data'%(self.mode,self.batch_len))

        if self.mode!='train' and self.epoch_counter>0:
            return None

        counter=0
        t_qid_list=[]
        t_iid_list=[]
        while counter<self.batchsize:
            t_qid=self.qid_list[self.batch_index]
            t_qid_list.append(t_qid)
            t_iid_list.append(self.get_iid(t_qid))
            counter+=1
            if self.batch_index<self.batch_len-1:
                self.batch_index+=1
            else:
                self.epoch_counter+=1
                qid_list=self.get_query_ids()
                random.shuffle(qid_list)
                self.qid_list=qid_list
                self.batch_index=0
                print('a epoch passes')
        t_batch=self.create_batch(t_qid_list)
        return t_batch+(t_iid_list,)

    def __len__(self):#__len__()和len()[the len() function will use the __len__ method if present to query your ohject for it's lenghth]
        return self.num_query#如果一个类表现得像一个list，要获取多少个元素，就得用len（）函数，为了让len（）函数工作正常，则类必须提供__len__()，来返回元素的个数







