import _pickle as cPickle
import numpy as np
def save(data, save_path):  # store object in the disk,and save and load when need,data should be saved permanently
    with open(save_path, 'wb') as f:
        return cPickle.dump(data, f)  # dumps VS dump，dump has a param (file pointer
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
