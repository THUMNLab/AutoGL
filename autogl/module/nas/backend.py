# tackle different backend
from autogl.backend import DependentBackend
_isdgl=DependentBackend.is_dgl()

def is_dgl():
    return _isdgl

def bk_mask(data,mask):
    if is_dgl():
        return data.ndata[f'{mask}_mask']
    else:
        return data[f'{mask}_mask']

def bk_label(data):
    if is_dgl():
        return data.ndata['label']
    else:
        return data.y

def bk_feat(data):
    if is_dgl():
        return data.ndata['feat']
    else:
        return data.x

def bk_gconv(op,data,feat):
    if is_dgl():
        return op(data,feat)
    else:
        return op(feat,data.edge_index)
