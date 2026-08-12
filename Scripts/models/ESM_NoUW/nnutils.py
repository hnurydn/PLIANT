import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torch
from torch.autograd import Variable
import os

# Check for CUDA availability once
USE_CUDA = torch.cuda.is_available()

def create_var(tensor, requires_grad=None):
    # If CUDA is available, use it. Otherwise, keep it on CPU.
    if requires_grad is None:
        if USE_CUDA:
            return Variable(tensor).cuda()
        else:
            return Variable(tensor).cpu()
    else:
        if USE_CUDA:
            return Variable(tensor, requires_grad=requires_grad).cuda()
        else:
            return Variable(tensor, requires_grad=requires_grad).cpu()

# This function is redundant but must be fixed if used elsewhere
def create_var_gpu(tensor, requires_grad=None):
    # This should ONLY be called when intending to use GPU, but we protect it anyway
    # Since your system failed, let's explicitly use the CPU path if CUDA is not found
    if requires_grad is None:
        if USE_CUDA:
            return Variable(tensor).cuda()
        else:
            # Fallback to CPU to prevent crash
            return Variable(tensor).cpu() 
    else:
        if USE_CUDA:
            return Variable(tensor, requires_grad=requires_grad).cuda()
        else:
            # Fallback to CPU to prevent crash
            return Variable(tensor, requires_grad=requires_grad).cpu()

def index_select_ND(source, dim, index):
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source.index_select(dim, index.view(-1))
    return target.view(final_size)

def avg_pool(all_vecs, scope, dim):
    size = torch.tensor([le for _,le in scope], dtype=torch.float32, device=all_vecs.device)
    return all_vecs.sum(dim=dim) / size.unsqueeze(-1)

def stack_pad_tensor(tensor_list):
    max_len = max([t.size(0) for t in tensor_list])
    for i,tensor in enumerate(tensor_list):
        pad_len = max_len - tensor.size(0)
        tensor_list[i] = F.pad( tensor, (0,0,0,pad_len) )
    return torch.stack(tensor_list, dim=0)

#3D padded tensor to 2D matrix, with padded zeros removed
def flatten_tensor(tensor, scope):
    assert tensor.size(0) == len(scope)
    tlist = []
    for i,tup in enumerate(scope):
        le = tup[1]
        tlist.append( tensor[i, 0:le] )
    return torch.cat(tlist, dim=0)

#2D matrix to 3D padded tensor
def inflate_tensor(tensor, scope): 
    max_len = max([le for _,le in scope])
    batch_vecs = []
    for st,le in scope:
        cur_vecs = tensor[st : st + le]
        cur_vecs = F.pad( cur_vecs, (0,0,0,max_len-le) )
        batch_vecs.append( cur_vecs )

    return torch.stack(batch_vecs, dim=0)

def GRU(x, h_nei, W_z, W_r, U_r, W_h):
    hidden_size = x.size()[-1]
    sum_h = h_nei.sum(dim=0,keepdim=True)
    z_input = torch.cat([x,sum_h], dim=-1)
    z = torch.sigmoid(W_z(z_input))

    r_1 = W_r(x).view(-1,hidden_size)
    r_2 = U_r(h_nei)
    r = torch.sigmoid(r_1 + r_2)
    
    gated_h = r * h_nei
    sum_gated_h = gated_h.sum(dim=0,keepdim=True)
    h_input = torch.cat([x,sum_gated_h], dim=-1)
    pre_h = torch.tanh(W_h(h_input))
    new_h = (1.0 - z) * sum_h + z * pre_h
    return new_h