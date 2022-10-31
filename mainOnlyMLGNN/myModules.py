# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 19:51:10 2020

@author: Luana Ruiz

"""

import torch
import torch.nn as nn
import math

# 2.1

def FilterFunction(h, S, x):
    K = h.shape[0]
    B = x.shape[0]
    N = x.shape[1]

    x = x.reshape([B, 1, N])
    S = S.reshape([1, N, N])
    z = x
    for k in range(1, K):
        x = torch.matmul(x, S)
        xS = x.reshape([B, 1, N])
        z = torch.cat((z, xS), dim=1)
    y = torch.matmul(z.permute(0, 2, 1).reshape([B, N, K]), h)
    return y
    
# 2.2
    
class GraphFilter(nn.Module):
    def __init__(self, gso, k):
        super().__init__()
        self.gso = torch.tensor(gso)
        self.n = gso.shape[0]
        self.k = k
        self.weight = nn.Parameter(torch.randn(self.k))
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.k)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return FilterFunction(self.weight, self.gso, x)
    
# 3.1
        
class GraphPerceptron(nn.Module):
    def __init__(self, gso, k, sigma):
        super().__init__()
        self.gso = torch.tensor(gso)
        self.n = gso.shape[0]
        self.k = k
        self.sigma = sigma
        self.weight = nn.Parameter(torch.randn(self.k))
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.k)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        y = FilterFunction(self.weight, self.gso, x)
        y = self.sigma(y)
        return y      
    
# 3.2
        
class MLGNN(nn.Module):
    def __init__(self, gso, l, k, sigma):
        super().__init__()
        layers = []
        for layer in range(l):
            layers.append(GraphPerceptron(gso, k[layer], sigma))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        y = self.layers(x)
        return y
