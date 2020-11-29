import torch
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
import math
import functools
import random

def distance(a, b): # L2
    return torch.sqrt(((a - b) ** 2).sum()).unsqueeze(0)

def multiply(x): #to flatten matrix into a vector 
    return functools.reduce(lambda x,y: x*y, x, 1)

def flatten(x):
    """ Flatten matrix into a vector """
    count = multiply(x.size())
    return x.resize_(count)


class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()

    def get_local_similarity(self, feature, memory):
        '''
        return score between patch and corresponding location memory [b, h*w]
        '''
        b, c, h, w = feature.size()
        feature = feature.view(b, h*w, c)
        memory = memory.view(h*w, c)
        similarity = F.softmax(torch.sum(feature*memory, dim=2), dim=1)  # b,hw score
        return similarity

    def read_local_memory(self, feature, memory, similarity):
        '''
        return weighted memory at corresponding location per sample [b, c, h, w]
        '''
        b, c, h, w = feature.size()
        memory = memory.view(c, h, w)
        similarity = similarity.view(b, h, w).unsqueeze(1).expand(-1, c, -1, -1)
        concat_memory = similarity.detach() * memory
        return concat_memory

    def gather_local_loss(self, feature, memory):
        '''
        Loss between each patch and its corresponding location memory
        '''
        # ######## memory loss
        loss_mse = torch.nn.MSELoss(reduction='none')
        loss_mse.zero_grad()
        # ########
        b, c, h, w = feature.size()
        feature = feature.contiguous().view(b,h*w, c)
        memory = memory.contiguous().view(h,w,c).permute(2, 0, 1).unsqueeze(0)
        memory = memory.expand(b, -1, -1, -1).contiguous().view(b,h*w, c)
        memory_loss = loss_mse(feature, memory.detach())
        return memory_loss

    def update_local(self, memory, feature, similarity):
        m, _ = memory.size()
        b, c, h, w = feature.size()
        similarity = similarity.view(b, h, w).unsqueeze(1).expand(-1, c, -1, -1)
        memory_update = torch.mean(similarity * feature, dim=0).view(h*w, c)
        updated_memory = F.normalize(0.01 * memory_update + memory.clone(), dim=1)

        return updated_memory

    def get_update_memory(self, memory, max_indices, softmax_score_query, feature):
        m, _ = memory.size()
        b, c, h, w = feature.size()
        feature = feature.contiguous().view(b * h * w, c)
        feature_update = torch.zeros((m, c)).cuda()
        for i in range(m):
            idx = torch.nonzero(max_indices.squeeze(1) == i, as_tuple=False)
            a, _ = idx.size()
            if a != 0:
                feature_update[i] = torch.sum(((softmax_score_query[idx, i] / torch.max(softmax_score_query[:, i]))
                                               * feature[idx].squeeze(1)), dim=0)
            else:
                feature_update[i] = 0

        return feature_update

    @autocast()
    def forward(self, x, memory, train=True):
        # query [batch X channel X H X W]
        # keys [memory items X channel]
        b, c, h, w = x.size()
        x = F.normalize(x, dim=1)  # channel wise norm

        similarity = self.get_local_similarity(x, memory)
        read_memory = self.read_local_memory(x, memory, similarity)
        memory_loss = self.gather_local_loss(x, memory)

        if train:
            with torch.no_grad():
                memory = self.update_local(memory, x, similarity)

        return read_memory, memory, memory_loss


if __name__ == "__main__":
    model = Memory()
