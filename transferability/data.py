# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 10:51:49 2020

@author: Damian Owerko
"""

import numpy as np


def sbm(n=50, c=5, p_intra=0.6, p_inter=0.2):

    # assign a community to each node
    community = np.repeat(list(range(c)), np.ceil(n / c))
    # make sure community vector has size n
    community = community[0:n]
    # make it a column vector
    community = np.expand_dims(community, 1)

    # generate a boolean matrix indicating whether two nodes
    # are in the same community
    intra = community == community.T
    # generate a boolean matrix indicating whether two nodes
    # are in different communities
    inter = np.logical_not(intra)
    # generate a matrix with random entries between 0 and 1
    random = np.random.random((n, n))
    # generate a triangular matrix with zeros below the main diagonal
    # because the SBM graph is symmetric, we only have to assign weights
    # to the upper triangular part of the adjacency matrix,
    # and later copy them to the lower triangular part
    tri = np.tri(n, k=-1)

    # initialize adjacency matrix
    graph = np.zeros((n, n))
    # assign intra-community edges
    graph[np.logical_and.reduce([tri, intra, random < p_intra])] = 1
    # assign inter-community edges
    graph[np.logical_and.reduce([tri, inter, random < p_inter])] = 1
    # make the adjacency matrix symmetric
    graph += graph.T

    return graph


def normalize_gso(gso):
    eigenvalues, _ = np.linalg.eig(gso)
    return gso / np.max(eigenvalues.real)


def generate_diffusion(gso, n_samples=2100, n_sources=10):

    # get the number of nodes
    n = gso.shape[0]

    # initialize the matrix used to store the samples
    # shape is n_samples x n x time x 1 features
    z = np.zeros((n_samples, n, 5, 1))

    for i in range(n_samples):

        # pick n_sources at random from n nodes
        sources = np.random.choice(n, n_sources, replace=False)

        # define z_0 for each sample
        z[i, sources, 0, 0] = np.random.uniform(0,10)

    # noise mean and variance
    mu = np.zeros(n)
    sigma = np.eye(n) * 1e-3

    for t in range(4):

        # generate noise
        noise = np.random.multivariate_normal(mu, sigma, n_samples)

        # generate z_t
        z[:, :, t + 1] = gso @ z[:, :, t] + np.expand_dims(noise, -1)
        
    # transpose dimensions so shape is n_samples x time x n x 1 feature
    z = z.transpose((0, 2, 1, 3))
    
    return z


def data_from_diffusion(z):
    
    # permute the samples in z
    z = np.random.permutation(z)
    
    # define the output tensor
    y = np.expand_dims(z[:, 0, :], 1)
    
    # initialize the input tensor
    x = np.zeros(y.shape)
    
    # define the input tensor as x = z_4
    for i, sample in enumerate(z):
        x[i] = sample[4]
        
    return x, y


def split_data(x, y, splits=(2000, 100)):
    
    # define the initial index of each set (training/test)
    splits = np.cumsum([0] + list(splits))
    splits = (splits * x.shape[0] / splits[-1]).astype(int)
    
    # return training and test data as tuples
    return ((x[splits[i]:splits[i + 1]], y[splits[i]:splits[i + 1]]) for i in range(len(splits) - 1))
