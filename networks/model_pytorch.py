import torch
import torch.nn as nn
from config.base_config import cfg
import numpy as np

from networks.data_layer import DataProviderLayer

import torch.nn.functional as F

#the class maybe in the models.py,include load data,initialize the params,then divide it into many files
class Net(nn.Module):
    def __init__(self, split, vocab_size):

        super(Net, self).__init__()

        self.split = split
        self.vocab_size = vocab_size


        # Word Embed definition
        self.embedding = nn.Embedding(vocab_size, cfg.WORD_EMB_SIZE)  # vocab_size:9368;cfg.WORD_EMB_SIZE:300

        # LSTM definition
        self.lstm = nn.LSTM(input_size=cfg.WORD_EMB_SIZE, hidden_size=1024, num_layers=1, batch_first=False)  # num_output:1024

        # print("try the test again adfahsdjkfhaskfjashfaskjfasjkfaslkf")


    def forward(self,qvec,img_feat,spt_feat):

        # Word Embed
        embed_ba = self.embedding(qvec)  # qvec :one hot vector input;output_num(embed_ba):300

        # Tanh
        embed = F.tanh(embed_ba)

        # LSTM
        lstm1, _ = self.lstm(embed)  # num_output:1024

        # Slice
        lstm1_out = lstm1[:, 14:-1:1]  # axis=0,column,slice_first14 (extract the feature of the last word from the LSTM network as the output feature)

        # Reshape
        lstm1_reshaped = lstm1_out.view(-1, cfg.RNN_DIM)


        # Dropout
        lstm1_droped = F.dropout(input=lstm1_reshaped, p=cfg.DROPOUT_RATIO, training=self.training)

        # L2 Normalize
        lstm_l2norm = F.normalize(input=lstm1_droped, p=2)  # perform Lp normalization of inputs over specified dimension,这里p=2

        # Reshape
        q_emb = lstm_l2norm.view(0, -1)

        # Concat
        v_spt = self.proc_image(img_feat, spt_feat)

        qv_relu = self.concat(q_emb, v_spt)



        query_score_fc1 = nn.Linear(out_features=1)
        query_score_fc = query_score_fc1(qv_relu)

        query_score_pred = query_score_fc.view(-1, cfg.RPN_TOPN)

        # KLD loss  have backward,predict score
        if cfg.USE_KLD:
            #softmaxKldLoss
            query_score_pred = F.log_softmax(query_score_pred)
            criterion = nn.KLDivLoss()
            loss_query_score = criterion(query_score_pred, query_label)  # query_label_mask function????

        else:
            #softmax and normal loss
            query_score_pred = F.log_softmax(query_score_pred)
            criterion = nn.MSELoss()
            loss_query_score = criterion(query_score_pred,query_label)


        # predict bbox
        query_bbox_pred1 = nn.Linear(out_features=4)
        query_bbox_pred = query_bbox_pred1(qv_relu)

        if cfg.USE_REG:
            loss_query_bbox = F.smooth_l1_loss(query_bbox_pred, query_bbox_targets)  # inside_weights and outside_weights function?????
        else:
            print("not use regression bbox loss")



        return loss_query_score, loss_query_bbox

    def proc_image(self, img_feat_layer, spt_feat_layer):

        # Concat
        v_spt = torch.cat((img_feat_layer, spt_feat_layer), axis=2)

        return v_spt

    def concat(self, q_layer, v_layer):

        # q
        # Reshape
        q_emb_resh1 = q_layer.view(0, 1, cfg.RNN_DIM)


        # Tile
        q_emb_tile = q_emb_resh1.repeat(repeats=cfg.RPN_TOPN, axis=1)

        # Reshape
        q_emb_resh = q_emb_tile.view(-1, cfg.RNN_DIM)

        # v
        v_emb_resh = v_layer.view(-1, cfg.SPT_FEAT_DIM+cfg.BOTTOMUP_FEAT_DIM)

        # q||v
        qv_fuse = torch.cat((q_emb_resh, v_emb_resh), axis=1)

        # FC
        qv_fc = nn.Linear(out_features=512)
        qv_fc1 = qv_fc(qv_fuse)

        #ReLU
        qv_relu = F.relu(qv_fc1)

        return qv_relu

    # def get_top(self):
    #     # top = np.zeros((9,))
    #     top = []
    #     param_str = json.dumps({'split': self.split, 'batchsize': cfg.BATCHSIZE})
    #     dataProviderLayer = DataProviderLayer(top, param_str)
    #
    #     #
    #     # self.qvec, self.cvec, self.img_feat, self.spt_feat, self.query_label, \
    #     # self.query_label_mask, self.query_bbox_targets, self.query_bbox_inside_weights, \
    #     # self.query_bbox_outside_weights = dataProviderLayer()
    #
    #     qvec, cvec, img_feat, spt_feat, query_label, \
    #     query_label_mask, query_bbox_targets, query_bbox_inside_weights, \
    #     query_bbox_outside_weights = dataProviderLayer()  # cvec in the pytorch version may no need,just ignore it
    #
    #     return qvec, cvec, img_feat, spt_feat, query_label, \
    #     query_label_mask, query_bbox_targets, query_bbox_inside_weights, \
    #     query_bbox_outside_weights






