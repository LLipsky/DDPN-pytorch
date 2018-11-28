import _pickle as cPickle
import numpy as np
def save(data, save_path):#将对象转化为文件保存在磁盘上，在需要的时候再读取并还原（load）。data是要持久化保存的对象
    with open(save_path, 'wb') as f:
        return cPickle.dump(data, f)#dumps和dump的区别是，dump多了一个类似文件指针的参数
def load(file_path):
    with open(file_path, 'rb') as f:
        return cPickle.load(f)

def complete_data(data,batchsize):
    if data.shape[0] == batchsize:
        return data
    if len(data.shape) == 1:
        t_data = np.zeros(batchsize-data.shape[0])
        return np.hstack(data, t_data)
    else:
        shape = (batchsize-data.shape[0],)+data.shape[1:]
        t_data = np.zeros(shape)
        return np.vstack(data, t_data)


#change x1,y1,x2,y2 => x,y,w,h
def transform_single(bbox):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]
    x = (x1 + x2)/ float(2)
    y = (y1 + y2)/ float(2)
    w = x2 - x1
    h = y2 - y1
    return np.hstack((x, y, w, h))
