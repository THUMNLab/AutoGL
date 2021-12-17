from .operations import *

OPS = {
    'ZERO': lambda indim, outdim, dropout, concat=False: Zero(indim, outdim),
    'IDEN': lambda indim, outdim, dropout, concat=False: Identity(),
    'GCN': lambda indim, outdim, dropout, concat=False: GCNConv(indim, outdim, add_self_loops=False),
    'SAGE-MEAN': lambda indim, outdim, dropout, concat=False: SAGEConv(indim, outdim),
    'GAT16': lambda indim, outdim, dropout, concat=False: GATConv(indim, outdim, dropout=dropout, heads=16, concat=False, add_self_loops=False) if not concat else GATConv(indim, outdim // 16, dropout=dropout, heads=16, concat=True, add_self_loops=False),
    'GAT2': lambda indim, outdim, dropout, concat=False: GATConv(indim, outdim, dropout=dropout, heads=2, concat=False, add_self_loops=False) if not concat else GATConv(indim, outdim // 2, dropout=dropout, heads=2, concat=True, add_self_loops=False),
    'GAT4': lambda indim, outdim, dropout, concat=False: GATConv(indim, outdim, dropout=dropout, heads=4, concat=False, add_self_loops=False) if not concat else GATConv(indim, outdim // 4, dropout=dropout, heads=4, concat=True, add_self_loops=False),
    'GAT8': lambda indim, outdim, dropout, concat=False: GATConv(indim, outdim, dropout=dropout, heads=8, concat=False, add_self_loops=False) if not concat else GATConv(indim, outdim // 8, dropout=dropout, heads=8, concat=True, add_self_loops=False),
    'GAT1': lambda indim, outdim, dropout, concat=False: GATConv(indim, outdim, dropout=dropout, heads=1, concat=False, add_self_loops=False),
    'LIN': lambda indim, outdim, dropout, concat=False: Linear(indim, outdim),
    'ARMA': lambda indim, outdim, dropout, concat=False: ARMAConv(indim, outdim),
    'CHEB': lambda indim, outdim, dropout, concat=False: ChebConv(indim, outdim, 2),
    'SGC': lambda indim, outdim, dropout, concat=False: SGConv(indim, outdim, add_self_loops=False)
}

PRIMITIVES = list(OPS.keys())
