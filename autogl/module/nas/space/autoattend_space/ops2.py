from .operations import *
OPS = {
    'ZERO': lambda indim, outdim, head, dropout, concat=False: Zero(indim, outdim),
    'CONST': lambda indim, outdim, head, dropout, concat=False: GeoLayer(indim, outdim, head, concat, att_type='const', dropout=dropout),
    'GCN': lambda indim, outdim, head, dropout, concat=False: GeoLayer(indim, outdim, head, concat, att_type='gcn', dropout=dropout),
    'GAT': lambda indim, outdim, head, dropout, concat=False: GeoLayer(indim, outdim, head, concat, att_type='gat', dropout=dropout),
    'SYM': lambda indim, outdim, head, dropout, concat=False: GeoLayer(indim, outdim, head, concat, att_type='gat_sym', dropout=dropout),
    'COS': lambda indim, outdim, head, dropout, concat=False: GeoLayer(indim, outdim, head, concat, att_type='cos', dropout=dropout),
    'LIN': lambda indim, outdim, head, dropout, concat=False: GeoLayer(indim, outdim, head, concat, att_type='linear', dropout=dropout),
    'GENE': lambda indim, outdim, head, dropout, concat=False: GeoLayer(indim, outdim, head, concat, att_type='generalized_linear', dropout=dropout)
}

PRIMITIVES = list(OPS.keys())
