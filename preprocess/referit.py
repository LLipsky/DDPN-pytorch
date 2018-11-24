#coding:utf-8
import xml.etree.ElementTree as ET
import os, re, sys
import numpy as np
import scipy.io as sio
from config.base_config import cfg
from utils.data_utils import save, load, transform_single
from utils.dictionary import Dictionary

iid_reg = re.compile(r'(\d+)')
ent_reg = re.compile(r'\[/EN#(\d+)/(\S+)\s(.*?)\]')


class Referit:
    def __init__(self):
        self.rebuild = False
        self.num_classes = 1
        self.data_dir = cfg.DATA_DIR
        self.data_paths = cfg.DATA_PATHS
        self.img_dir = cfg.IMAGE_DIR
        self.mask_dir = cfg.MASK_DIR
        self.raw_anno_path = cfg.RAW_ANNO_PATH
        
        self.query_maxlen = cfg.QUERY_MAXLEN
        self.vocab_space = cfg.VOCAB_SPACE

        self.image_ext = '.jpg'
        self.qdic_dir = cfg.DICT_DIR
        if not os.path.exists(self.qdic_dir) or self.rebuild:
            self.qdic = self.make_dict(self.vocab_space, self.raw_anno_path, self.qdic_dir)
        else:
            self.qdic = Dictionary(self.qdic_dir)
            self.qdic.load()

        self.vocab_size = self.qdic.size()

    def vocab_size(self):
        return self.vocab_size

    def make_dict(self, vocab_space, raw_anno_path, dict_dir):
        print('making dict...')
        iid_list = self.load_img_iids(vocab_space)
        dic = Dictionary(dict_dir)
        dic.add_specials(cfg.SP_WORDS, cfg.SP_IDXS)
        with open(raw_anno_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                # 8756_2.jpg~sunray at very top~.33919597989949750~.023411371237458192
                splits = line.strip().split('~', 2)
                # example: 8756_2 (segmentation regions)
                img_name = splits[0].split('.', 1)[0]
                iid = img_name.split('_', 1)[0]
                if iid not in iid_list:
                    continue
                # example: 'sunray at very top'
                dic.add_tokens(splits[1].strip().lower().split())
                # construct imcrop_name - discription list dictionary
                # an image crop can have zero or mutiple annotations
        dic.save()
        return dic

    def load_img_iids(self, data_split):
        image_set_file = self.data_paths[data_split]['id_list_path']
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file, 'r') as f:
            str_list = f.readlines()
        str_list = [s[:-1] for s in str_list]
        return str_list


    def format_data(self, data_split):
        data_store_path = self.data_paths[data_split]['format_data_path']
        if not os.path.exists(data_store_path) or self.rebuild:
            print ('making %s cache ...'%data_split)
            iid_list = self.load_img_iids(data_split)
            anno_dict = self.load_annotation(data_split, iid_list, self.raw_anno_path, self.mask_dir)
            save(anno_dict, data_store_path)
        else:
            print ('%s format data already exists ...'%data_split)

    def get_img_path(self, iid):
        return os.path.join(self.img_dir, iid + self.image_ext)

    def load_annotation(self, data_split, iid_list, raw_anno_path, mask_dir):
        anno_dict = {}
        with open(raw_anno_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                # 8756_2.jpg~sunray at very top~.33919597989949750~.023411371237458192
                splits = line.strip().split('~', 2)
                # example: 8756_2 (segmentation regions)
                imcrop_name = splits[0].split('.', 1)[0]
                iid = imcrop_name.split('_', 1)[0]
                if iid not in iid_list:
                    continue
                # example: 'sunray at very top'
                qstr = splits[1].strip().lower()
                qstr = ' '.join(self.qdic.split_words(qstr))

                # construct imcrop_name - discription list dictionary
                # an image crop can have zero or mutiple annotations
                mask_name = imcrop_name + '.mat'
                mask_path = os.path.join(mask_dir, mask_name)
                mask = sio.loadmat(mask_path)['segimg_t']
                idx = np.nonzero(mask == 0)
                x_min, x_max = np.min(idx[1]), np.max(idx[1])
                y_min, y_max = np.min(idx[0]), np.max(idx[0])
                bbox = [x_min, y_min, x_max, y_max]
                anno_dict[imcrop_name] = {'iid': iid, 'img_path': self.get_img_path(iid), 'boxes': [np.array(bbox)], 'qstr':qstr}

        return anno_dict
        # anno['boxes'] = boxes
        # anno['seg_areas'] = seg_areas
        # anno['gt_classes'] = gt_classes
        # anno['gt_overlaps'] = overlaps
        # anno['height'] = height
        # anno['width'] = width

        # anno['queries'] = queries
        # anno['query_cont'] = query_cont
        # anno['query_label'] = query_label
        # anno['flipped'] = False









