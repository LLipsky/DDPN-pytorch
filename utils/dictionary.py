import os
from utils.data_utils import save
from utils.data_utils import load
import config.base_config as cfg

stop_words = [',', '<', '.', '>', '/', '?', '\'', '"', '\\', '-', '_', '=', '+', '[', ']', '{', '}', '|', ':', ';', '(', ')', '*', '&', '%', '^', '$', '#', '@', '!', '~', '`']

class Dictionary(object):
    def __init__(self,save_dir):
        self.indx2token={}
        self.token2indx={}
        self.word_freq={}
        self.special=[]
        self.save_dir=save_dir

    def save(self,save_dir=None):
        if not save_dir is None:
            self.save_dir=save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        save(self.indx2token,self.save_dir+'/idx2token.pkl')
        save(self.token2indx,self.save_dir+'/token2idx.pkl')
        save(self.word_freq,self.save_dir+'/word_freq.pkl')
        save(self.special,self.save_dir+'/special_words.pkl')

    def load(self):
        self.indx2token=load(self.save_dir+'/idx2token.pkl')
        self.token2indx=load(self.save_dir+'/token2idx.pkl')
        self.word_freq=load(self.save_dir+'/word_freq.pkl')
        self.special=load(self.save_dir+'special_words.pkl')

    def lookup(self,token,default=cfg.UNK):#UNK=1
        return self.token2indx.get(token,default)#这里的get(key,default)函数返回指定键的值，如果值不在字典中则返回默认值



