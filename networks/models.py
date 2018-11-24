import json
import config.base_config as cfg
from networks.data_layer import DataProviderLayer
import torch
import torch.nn as nn
import torch.nn.functional as F
def net(split,vocab_size):
    param_str=json.dumps({'split':split,'batchsize':cfg.BATCHSIZE})

    dataProviderLayer=DataProviderLayer()
    top=[]
    #各个参数的意思是什么？？？？？/
    qvec,cvec,img_feat,spt_feat,query_label,query_label_mask,query_bbox_targets,\
    query_bbox_inside_weights,query_bbox_outside_weights=dataProviderLayer(top,param_str)

    #Query部分
    embedding=nn.Embedding(vocab_size,cfg.WORD_EMB_SIZE)#input_size;num_output;there is vocab_size words in vocab,WORD_EMB_SIZE dimensional embeddings
    embed_ba=embedding(qvec)

    embed=nn.Tanh(embed_ba)
    word_emb=embed

    #LSTM部分
    lstm=nn.LSTM(word_emb.shape[0],cfg.RNN_DIM)#input dim;output dim
    lstm_out,hidden=lstm(word_emb,cvec)#inputs,hidden

    lstm1_reshaped=lstm_out.view((-1,cfg.RNN_DIM))
    lstm1_droped=nn.Dropout(lstm1_reshaped,cfg.DROPOUT_RATIO)
    lstm_l2norm=F.normalize(input=lstm1_droped,p=2)#perform Lp normalization of inputs over specified dimension,这里p=2
    q_emb=lstm_l2norm.view((0,-1))
    q_layer=q_emb#(N,1024)

    #Image 部分
    v_layer=proc_img(img_feat,spt_feat)#out:(N,100,2048+5)

    out_layer=concat(q_layer,v_layer)#fuse q and v

    #predict score
    query_score_fc1=nn.Linear(out_features=1)
    query_score_fc=query_score_fc1(out_layer)
    query_score_pred=query_score_fc.view((-1,cfg.RPN_TOPN))

    if cfg.USE_KLD:
        loss_query_score=nn.KLDivLoss()
    """"""
    else:



    #predict bbox
    query_bbox_pred_fc1=nn.Linear(out_features=4)
    query_bbox_pred=query_bbox_pred_fc1(out_layer)

    if cfg.USE_REG:
        loss_query_score=
    else:

    return

def proc_img(img_feat_layer,spt_feat_layer):#就是concat img_feat和spt_feat
    v_spt=torch.cat((img_feat_layer,spt_feat_layer),2)
    out_layer=v_spt
    return out_layer

def concat(q_layer,v_layer):
    #input:q_layer:(N,1024) v_layer:(N,100,2048+5)
    q_emb_resh1=q_layer.view((0,1,cfg.RNN_DIM))
    q_emb_tile=q_emb_resh1.repeat(cfg.RPN_TOPN)#有可能是.repeat(cfg.RPN_TOPN,1)
    q_emb_resh=q_emb_tile.view((-1,cfg.RNN_DIM))

    v_emb_resh=v_layer.view((-1,cfg.SPT_FEAT_DIM+cfg.BOTTOMUP_FEAT_DIM))#5+2048
    qv_fuse=torch.cat((q_emb_resh,v_emb_resh),1)

    qv_fc=nn.Linear(out_features=512)#in_features,out_features
    qv_fc1=qv_fc(qv_fuse)

    qv_relu=nn.ReLU(qv_fc1)
    return qv_relu


















