import config.base_config as cfg
from data_provider.single_data_provider import SingleDataProvider
def get_data_provider(data_split,batchsize=1):
    if cfg.NTHREADS>1:
        try:
            import torch
            from data_provider.multi_data_provider import MultiDataProvider
            #引入多线程
        except:
            cfg.NTHREADS=1

    if cfg.NTHREADS>1:
        print("多线程的dataprovider函数")# data_provider=#等于多线程的dataProvider
        data_provider=MultiDataProvider(data_split,batchsize)
    else:
        print("单线程dataprovider函数")
        data_provider=SingleDataProvider(data_split,batchsize)#等于单线程的dataProvider

    return data_provider