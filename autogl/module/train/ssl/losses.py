# NTXent_loss from <https://github.com/Shen-Lab/GraphCL/>
import torch
import torch.nn as nn

def NTXent_loss(zs, tau=0.5, norm=True):
    batch_size, _ = zs[0].size()
    sim_matrix = torch.einsum('ik,jk->ij', zs[0], zs[1])
    if norm:
        z1_abs = zs[0].norm(dim=1)
        z2_abs = zs[1].norm(dim=1)
        sim_matrix = sim_matrix / torch.einsum('i,j->ij', z1_abs, z2_abs)
    sim_matrix = torch.exp(sim_matrix/tau)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()
    
    return loss
