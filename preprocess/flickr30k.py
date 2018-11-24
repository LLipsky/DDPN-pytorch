# coding:utf-8
import xml.etree.ElementTree as ET
import os, re, sys
import numpy as np
from config.base_config import cfg
from utils.data_utils import save, load, transform_single
from utils.dictionary import Dictionary

iid_reg = re.compile(r'(\d+)')
ent_reg = re.compile(r'\[/EN#(\d+)/(\S+)\s(.*?)\]')


class Flickr30k:
    def __init__(self):
        self.rebuild = False
        self.num_classes = 1
        self.data_dir = cfg.DATA_DIR
        self.data_paths = cfg.DATA_PATHS
        self.raw_sen_dir = cfg.RAW_SEN_DIR
        self.raw_img_dir = cfg.RAW_IMG_DIR
        self.raw_anno_dir = cfg.RAW_ANNO_DIR
        self.query_maxlen = cfg.QUERY_MAXLEN
        self.vocab_space = cfg.VOCAB_SPACE

        self.image_ext = '.jpg'
        self.split_tok = cfg.SPLIT_TOK
        self.qdic_dir = cfg.DICT_DIR
        if not os.path.exists(self.qdic_dir) or self.rebuild:
            print ('making query dict...')
            self.qdic = self.make_dict(self.vocab_space, cfg.RAW_SEN_DIR, self.qdic_dir)
        else:
            self.qdic = Dictionary(self.qdic_dir)
            self.qdic.load()

        self.vocab_size = self.qdic.size()

    def vocab_size(self):
        return self.vocab_size

    def make_dict(self, vocab_space, raw_sen_dir, dict_dir):
        iid_list = self.load_img_iids(vocab_space)
        dic = Dictionary(dict_dir)
        dic.add_specials(cfg.SP_WORDS, cfg.SP_IDXS)
        # cls_info = {}

        for iid in iid_list:
            senpath = os.path.join(raw_sen_dir, iid + '.txt')
            with open(senpath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line == '':
                        continue
                    ms = ent_reg.findall(line)
                    for m in ms:
                        ent_id = m[0]
                        ent_type = m[1].strip().lower()
                        ent_str = m[2].strip().lower()
                        dic.add_tokens(ent_str.split())
        dic.save()
        return dic

    def load_img_iids(self, data_split):
        img_iids = []
        for split in data_split.split(self.split_tok):
            image_set_file = self.data_paths[split]['id_list_path']
            assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
            with open(image_set_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line == '':
                        continue
                    iid = iid_reg.search(line).group(1)
                    img_iids.append(iid)
        return img_iids

    def format_data(self, data_split):
        data_store_path = self.data_paths[data_split]['format_data_path']
        if not os.path.exists(data_store_path) or self.rebuild:
            print('making %s cache ...' % data_split)
            iid_list = self.load_img_iids(data_split)
            ent_dict = self.make_ent_dict(iid_list)
            anno_dict = self.load_annotation(data_split, iid_list, ent_dict)
            save(anno_dict, data_store_path)
        else:
            print('%s format data already exists ...' % data_split)

    def make_ent_dict(self, iid_list):
        ent_dict = {}
        for iid in iid_list:
            senpath = os.path.join(self.raw_sen_dir, iid + '.txt')
            with open(senpath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line == '':
                        continue
                    ms = ent_reg.findall(line)
                    for m in ms:
                        ent_id = m[0]
                        ent_type = m[1].strip().lower()
                        ent_str = m[2].strip().lower()
                        if ent_id not in ent_dict:
                            ent_dict[ent_id] = {'ent_type': ent_type, 'ent_str': [ent_str], 'ent_num': 1}
                        else:
                            if ent_str not in ent_dict[ent_id]['ent_str']:
                                ent_dict[ent_id]['ent_str'].append(ent_str)
                                ent_dict[ent_id]['ent_num'] += 1
        return ent_dict

    def mcb_process_boxes(self, bbox_dict):
        for ent_id in bbox_dict:
            bboxes = np.vstack(bbox_dict[ent_id])
            xmin = np.min(bboxes[:, 0])
            ymin = np.min(bboxes[:, 1])
            xmax = np.max(bboxes[:, 2])
            ymax = np.max(bboxes[:, 3])
            bbox_dict[ent_id] = [np.array([xmin, ymin, xmax, ymax])]
        return bbox_dict

    @property
    def process_boxes(self):
        return self.mcb_process_boxes

    def load_annotation(self, mode, iid_list, ent_dict):
        anno_dict = {}
        for iid in iid_list:
            annopath = os.path.join(self.raw_anno_dir, iid + '.xml')
            tree = ET.parse(annopath)
            filename = tree.find('filename')
            size = tree.find('size')
            height = size.find('height').text
            width = size.find('width').text
            objs = tree.findall('object')
            num_objs = len(objs)
            bbox_dict = {}

            for obj in objs:
                ent_ids = [name.text for name in obj.findall('name')]
                bbox = obj.find('bndbox')
                if bbox is None:
                    continue
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1
                bbox = np.array([x1, y1, x2, y2])
                for ent_id in ent_ids:
                    if ent_id not in bbox_dict:
                        bbox_dict[ent_id] = [bbox]
                    else:
                        bbox_dict[ent_id].append(bbox)
            # merge box
            bbox_dict = self.process_boxes(bbox_dict)

            # 处理query
            for ent_id in bbox_dict:
                if not ent_dict.has_key(ent_id):
                    continue
                for i, qstr in enumerate(ent_dict[ent_id]['ent_str']):
                    aid = iid + '_' + ent_id + '_' + str(i)
                    anno_dict[aid] = {'iid': iid, 'boxes': bbox_dict[ent_id], 'qstr': qstr,
                                      'qtype': ent_dict[ent_id]['ent_type']}
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


if __name__ == '__main__':
    flickr = Flickr30k()
    flickr.format_data()
