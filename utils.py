from scipy.spatial.distance import cosine
import numpy as np
import torch
import scipy

class ToInt:
    def __call__(self, pic):
        return pic * 255

def NCD(c1, c2, c12):
    dis = (c12-min(c1,c2))/max(c1, c2)
    return dis

def CLM(c1, c2, c12):
    dis = 1 - (c1+c2-c12)/c12
    return dis

def CDM(c1, c2, c12):
    dis = c12/(c1+c2)
    return dis

def MSE(v1, v2):
    return np.sum((v1-v2)**2)/len(v1)

def agg_by_concat_space(t1, t2):
    return t1+' '+t2

    return ''.join(comb)

def agg_by_avg(i1, i2):
    return torch.div(i1+i2, 2, rounding_mode='floor')

def agg_by_stack(i1, i2):
    return torch.stack([i1, i2])
