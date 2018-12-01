import torch.nn as nn
from config.base_config import cfg
from utils.data_utils import load
from utils.dictionary import Dictionary
import os
import skimage.io
import numpy as np
from utils.bbox_transform import bbox_overlaps_batch
from utils.bbox_transform import bbox_transform
class DataProvider(nn.Module):
    def __init__(self, data_split, batchsize=1):
        print("init DataProvider for %s : %s : %s", cfg.IMDB_NAME, cfg.PROJ_NAME, cfg.data_split)
        self.is_ss = cfg.FEAT_TYPE == 'ss'  # selective search
        self.ss_box_dir = cfg.SS_BOX_DIR
        self.ss_feat_dir = cfg.SS_FEAT_DIR
        self.feat_type = cfg.FEAT_TYPE

        if 'refcoco' in cfg.IMDB_NAME or cfg.IMDB_NAME == 'refclef':  # base_config.py: __C.IMDB_NAME = 'refcoco'
            self.is_mscoco_prefix = True
        else:
            self.is_mscoco_prefix = False

        self.use_kld = cfg.USE_KLD
        self.rpn_topn = cfg.RPN_TOPN

        if self.is_ss:
            self.bottomup_feat_dim = cfg.SS_FEAT_DIM  # 4096
        else:
            self.bottomup_feat_dim = cfg.BOTTOMUP_FEAT_DIM  # 2048

        self.query_maxlen = cfg.QUERY_MAXLEN
        self.image_ext = '.jpg'
        data_splits = data_split.split(cfg.SPLIT_TOK)  # SPLIT_TOK='+'
        if 'train' in data_splits:
            self.mode = 'train'
        else:
            self.mode = 'test'

        self.batchsize = cfg.BATCHSIZE
        self.image_dir = cfg.IMAGE_DIR
        self.feat_dir = cfg.FEAT_TYPE
        self.dict_dir = cfg.QUERY_DIR

        self.anno = self.load_data(data_splits)
        self.qdic = Dictionary(self.dict_dir)  # indx2token.pkl;token2indx.pkl;special_words.pkl;word_freq.pkl
        self.qdic.load()
        self.index = 0
        self.batch_len = None
        self.num_query = len(self.anno)

    def load_data(self, data_splits):
        anno = {}
        for data_split in data_splits:
            data_path = cfg.ANNO_PATH % str(data_split)  # cfg.ANNO_PATH = __C.DATA_DIR + 'format_%s.pkl',/home/lipin/code/DDPN-master/data/format_dataset/../format_%s.pkl

            t_anno = load(data_path)  # /home/lipin/code/DDPN-master/data/format_dataset/refcoco/format_%s.pkl(%s:val,testA,testB,train)
            anno.update(t_anno)  # special dict.update()

        return anno  # store data

    def get_image_ids(self):
        qid_list = self.get_query_ids()
        iid_list = set()
        for qid in qid_list:
            iid_list.add(self.anno[qid]['iid'])
        return list(iid_list)

    def get_query_ids(self):
        return self.anno.keys()

    def get_num_query(self):
        return self.num_query

    def get_vocabsize(self):
        return self.qdic.size()

    def get_iid(self, qid):
        return self.anno[qid]['iid']

    def get_img_path(self, iid):
        if self.is_mscoco_prefix:
            return os.path.join(self.image_dir, 'COCO_train2014_'+str(iid).zfill(12)+self.image_ext)#zfill原str右对齐，前面的填充0
        else:
            return os.path.join(self.image_dir, str(iid)+self.image_ext)

    def load_ss_box(self, ss_box_path):
        boxes = np.loadtxt(ss_box_path)  # loadtxt()是Load data from a text file.
        if len(boxes) == 0:
            raise Exception("boxes is None!")
        boxes = boxes-1  # why？？？？？？
        boxes[:, [0, 1]] = boxes[:, [1, 0]]
        boxes[:, [2, 3]] = boxes[:, [3, 2]]
        return boxes

    def get_topdown_feat(self, iid):  # return img_feat and spt_feat,iid is the name of the image
        try:
            if self.is_ss:  # different feat_type，different operation
                img_path = self.get_img_path(iid)  # due to iid get img path
                im = skimage.io.imread(img_path)
                img_h = im.shape[0]
                img_w = im.shape[1]
                feat_path = os.path.join(self.feat_dir, str(iid)+'.npz')  # npz文件是压缩的二进制文件，load函数自动识别npz文件，并且返回一个类似字典的对象
                ss_box_path = os.path.join(self.ss_box_dir, str(iid)+'.txt')
                bbox = self.load_ss_box(ss_box_path)  # ss:selective search
                num_bbox = bbox.shape[0]
                img_feat = np.transpose(np.load(feat_path)['x'], (1, 0))
            else:
                if self.is_mscoco_prefix:  # self.image_ext = '.jpg', the value of str(iid).zfill(12) or str(iid) is image's id(such as 000000581560)
                    feat_path = os.path.join(self.feat_dir, 'COCO_train2014_'+str(iid).zfill(12)+self.image_ext+'.npz')  # just the path of every image's feature map
                else:
                    feat_path = os.path.join(self.feat_dir, str(iid)+self.image_ext+'.npz')

                feat_dict = np.load(feat_path)  # 想打印出来看看feat_dict是怎么样的？？？？？？
                img_feat = feat_dict['x']
                num_bbox = feat_dict['num_bbox']  # what is the meaning?????
                bbox = feat_dict['bbox']
                img_h = feat_dict['img_h']
                img_w = feat_dict['img_w']

            return img_feat, num_bbox, bbox, (img_h, img_w)
        except Exception as e:#except Exception,e:这个python3.6不支持，解决办法是将‘，’改成‘as’
            print(e)
            raise Exception("UnknownError")


    def get_spt_feat(self,bbox,img_shape):  # due to the bbox's two coordinates,compute the center
        # spt_shape. a 5-D spatial feature vspat = [xtl/W,ytl/H,xbr/W,ybr/H,wh/WH] tl:top left;br:bottom right
        spt_feat = np.zeros((bbox.shape[0], 5), dtype=np.float)

        spt_feat[:, 0] = bbox[:, 0]/float(img_shape[1])  # img_shape[1] W
        spt_feat[:, 1] = bbox[:, 1]/float(img_shape[0])  # img_shape[0] H
        spt_feat[:, 2] = bbox[:, 2]/float(img_shape[1])
        spt_feat[:, 3] = bbox[:, 3]/float(img_shape[0])
        spt_feat[:, 4] = (bbox[:, 2]-bbox[:, 0])*(bbox[:, 3]-bbox[:, 1])/float(img_shape[0]*img_shape[1])
        return spt_feat

    def str2list(self, qstr, query_maxlen):
        q_list = qstr.split()
        qvec = np.zeros(query_maxlen, dtype=np.int64)
        cvec = np.zeros(query_maxlen, dtype=np.int64)
        for i, _ in enumerate(range(query_maxlen)):
            if i < query_maxlen-len(q_list):
                cvec[i] = 0
            else:
                w = q_list[i-(query_maxlen-len(q_list))]
                qvec[i] = self.qdic.lookup(w)  # lookup(word),return index
                cvec[i] = 0 if i == query_maxlen-len(q_list) else 1  # ignore

        return qvec, cvec

    def create_batch_rpn(self, iid):
        img_path = self.get_img_path(iid)
        img_feat, num_bbox, bbox, (img_h, img_w) = self.get_topdown_feat(iid)
        return num_bbox, bbox, img_path

    def create_batch_recall(self, qid):  # qid是query_ids
        iid = self.anno[qid]['iid']
        gt_bbox = self.anno[qid]['boxes']  # 所以想知道anno中的data具体是怎么样的，这就得关注数据集读取的时候的部分了！！！！
        img_path = self.get_img_path(iid)

        img_feat, num_bbox, bbox, (img_h, img_w) = self.get_topdown_feat(iid)
        return num_bbox, bbox, gt_bbox, img_path

    def compute_targets(self, ex_rois, gt_rois, query_label):
        assert ex_rois.shape[1] == 4
        # assert ex_rois.shape[1]==4

        targets = bbox_transform(ex_rois, gt_rois)  # return ctr_x,ctr_y,w,h
        if cfg.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            targets = ((targets-np.array(cfg.BBOX_NORMALIZE_TARGETS_PRECOMPUTED))/np.array(cfg.BBOX_NORMALIZE_STDS))

        query_bbox_target_data = np.hstack((query_label[:, np.newaxis], targets)).astype(np.float32, copy=False)

        return query_bbox_target_data

    def get_query_bbox_regression_labels(self, query_bbox_target_data):  # query_bbox_target_data include query_label and query_bbox targets
        query_label = query_bbox_target_data[:, 0]
        query_bbox_targets = np.zeros((query_label.size, 4), dtype=np.float32)
        query_bbox_inside_weights = np.zeros(query_bbox_targets.shape, dtype=np.float32)
        inds = np.where(query_label > 0)[0]
        if len(inds) != 0:
            for ind in inds:
                query_bbox_targets[ind, :] = query_bbox_target_data[ind, 1:]
                if query_label[ind] == 1:  # 根据不同的label值来设置权重
                    query_bbox_inside_weights[ind, :] = cfg.BBOX_INSIDE_WEIGHTS
                elif query_label[ind] == 2:
                    query_bbox_inside_weights[ind, :] = 0.2

        return query_bbox_targets, query_bbox_inside_weights


    # 获取bbox regression对应的mask，根据scores获取对应的label
    def get_labels(self, rpn_rois, gt_boxes):  # rpn_rois means t_bbox(gotten from DDPN(faster-rcnn) rpn),gt_boxes means t_gt_boxes
        # to get labels(query_label) of the rpn_rois,according to overlaps
        overlaps = bbox_overlaps_batch(np.ascontiguousarray(rpn_rois, dtype=float), np.ascontiguousarray(gt_boxes[:, :4], dtype=float))

        if self.use_kld:
            query_label = np.zeros(rpn_rois.shape[0])

        query_label_mask = 0
        bbox_label = np.zeros(rpn_rois.shape[0])

        # 找出query=1的gt_boxes的index
        query_gt_ind = 0
        query_overlaps = overlaps[:, query_gt_ind].reshape(-1)

        if self.use_kld:
            # kld：根据iou设置权重
            if query_overlaps.max() >= 0.5:
                query_label_mask = 1  # positive example
                query_inds = np.where(query_overlaps >= cfg.THRESHOLD)[0]

                for ind in query_inds:
                    query_label[ind] = query_overlaps[ind]
                if query_label.sum() == 0:
                    print(query_overlaps.max)
                query_label = query_label/float(query_label.sum())

        else:
            #softmax
            if query_overlaps.max() >= 0.5:
                query_label = int(query_overlaps.argmax())
            else:
                query_label = -1
        rois = rpn_rois
        gt_assigment = overlaps.argmax(axis=1)  # 存储overlaps最大时的x
        gt_target_boxes = gt_boxes[gt_assigment, :4]  # 存储overlaps的最大的boxes（anchors），是以两个点的坐标形式存放，左上角和右下角
        bbox_label[np.where(overlaps.max(axis=1) >= 0.5)[0]] = 2  # 将overlaps时大于0.5的boxes的label改为2
        if query_overlaps.max() >= 0.5:
            query_inds = np.where(query_overlaps >= cfg.THRESHOLD)[0]  # cfg.THRESHOLD=0.5
            bbox_label[query_inds] = 1  # 将query_overlaps大于0.5的指定index的bbox的label改成1
            gt_target_boxes[query_inds] = gt_boxes[query_gt_ind, :4]

        bbox_target_data = self.compute_targets(rois, gt_target_boxes, bbox_label)
        query_bbox_targets, query_bbox_inside_weights = self.get_query_bbox_regression_labels(bbox_target_data)  # set weights due to the different query label value
        query_bbox_outside_weights = np.array(query_bbox_inside_weights > 0).astype(np.float32)  # 将大于0的inside_weights挑选出来

        return query_label, query_label_mask, query_bbox_targets, query_bbox_inside_weights, query_bbox_outside_weights

    def str2list(self, qstr, query_maxlen):
        q_list = qstr.split()
        qvec = np.zeros(self.query_maxlen, dtype=np.int64)
        cvec = np.zeros(self.query_maxlen, dtype=np.int64)
        for i, _ in enumerate(range(query_maxlen)):
            if i < query_maxlen-len(q_list):
                cvec[i] = 0
            else:
                w = q_list[i-(query_maxlen-len(q_list))]
                qvec[i] = self.qdic.lookup(w)  # find the word is in the vocabulary?
                cvec[i] = 0 if i == query_maxlen-len(q_list) else 1

        return qvec, cvec










