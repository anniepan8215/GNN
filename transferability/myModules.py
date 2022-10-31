# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 19:51:10 2020

@author: Luana Ruiz

"""

import torch
import torch.nn as nn
import math

# 4.1

def FilterFunction(h, S, x, b=None):
    F = h.shape[0]
    K = h.shape[1]
    G = h.shape[2]

    N = S.shape[1]
    B = x.shape[0]

    x = x.reshape([B, 1, G, N])
    S = S.reshape([1, N, N])
    z = x
    for k in range(1, K):
        x = torch.matmul(x, S)
        xS = x.reshape([B, 1, G, N])
        z = torch.cat((z, xS), dim=1)
    y = torch.matmul(z.permute(0, 3, 1, 2).reshape([B, N, K*G]), h.reshape([F, K*G]).permute(1, 0)).permute(0, 2, 1)
    
    if b is not None:
        y = y +b
    return y
    
    
class GraphFilter(nn.Module):
    def __init__(self, gso, k, f_in, f_out, bias):
        super().__init__()
        self.gso = torch.tensor(gso)
        self.n = gso.shape[0]
        self.k = k
        self.f_in = f_in
        self.f_out = f_out
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(self.f_out, 1))
        self.weight = nn.Parameter(torch.randn(self.f_out, self.k, self.f_in))
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.f_in * self.k)
        self.weight.data.uniform_(-stdv, stdv)
        
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return FilterFunction(self.weight, self.gso, x, self.bias)
    
    def changeGSO(self, new_gso):
        self.gso = torch.tensor(new_gso)
        self.n = new_gso.shape[0]
    
# 4.2
        
class GNN(nn.Module):
    def __init__(self, gso, l, k, f, sigma, bias):
        super().__init__()
        self.gso = torch.tensor(gso)
        self.n = gso.shape[0]
        self.l = l
        self.k = k
        self.f = f
        self.sigma = sigma
        self.bias = bias
        
        gml = []
        for layer in range(l):
            gml.append(GraphFilter(gso,k[layer],f[layer],f[layer+1], bias))
            gml.append(sigma)
        
        self.gml = nn.Sequential(*gml)
        
    def forward(self, x):
        return self.gml(x)
    
    def changeGSO(self, new_gso):
        self.gso = new_gso
        for layer in range(2*self.l):
            if layer % 2 == 0:
                self.gml[layer].changeGSO(new_gso)
        self.n = new_gso.shape[0]
                
    
    
        
