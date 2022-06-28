

import torch
import torch.nn.functional as F
import pdb
from torch import nn
import numpy as np



def _reg_loss(x1,x2, args,margin=0.5):
    
    x1_norm = F.normalize(x1,dim = 0) # batch-wise normalization
    x2_norm = F.normalize(x2,dim = 0)
    
    cor_mat = (x1_norm.t() @ x2_norm).clamp(min=1e-7)


    diag = torch.diagonal(cor_mat)
    margin_mask = (diag>margin).type(torch.uint8)

    loss = (diag*margin_mask).sum()/args.train_batch
    
    return loss



def reg_loss(tok_embeddings, args):

    total_reg_loss = 0
    
    raw_tok_embeddings, aug_tok_embeddings =  tok_embeddings
    
    for i in range(len(raw_tok_embeddings)-1):
        raw_hid = raw_tok_embeddings[i]
        raw_hid_next = raw_tok_embeddings[i+1]

        aug_hid = aug_tok_embeddings[i]
        aug_hid_next = aug_tok_embeddings[i+1]
        
        
        total_reg_loss += (_reg_loss(raw_hid,raw_hid_next, args) + _reg_loss(aug_hid,aug_hid_next, args)) / 2
        
    return total_reg_loss


def get_sim_mat(x):
    x = F.normalize(x, dim=1)
    return (x @ x.t()).clamp(min=1e-7)  # normalized cosine similarity scores

def Supervised_NT_xent(sim_matrix, labels, temperature=0.2, chunk=2, eps=1e-8):
    '''
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    '''

    device = sim_matrix.device
    labels = labels.repeat(2)

    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    sim_matrix = sim_matrix - logits_max.detach()

    B = sim_matrix.size(0) // chunk  # B = B' / chunk

    eye = torch.eye(B * chunk).to(device)  # (B', B')
    sim_matrix = torch.exp(sim_matrix / temperature) * (1 - eye)  # remove diagonal

    denom = torch.sum(sim_matrix, dim=1, keepdim=True)
    sim_matrix = -torch.log(sim_matrix / (denom + eps) + eps)  # loss matrix

    labels = labels.contiguous().view(-1, 1)
    Mask = torch.eq(labels, labels.t()).float().to(device)
    Mask = Mask / (Mask.sum(dim=1, keepdim=True) + eps)

    loss = torch.sum(Mask * sim_matrix) / (2 * B)

    return loss

