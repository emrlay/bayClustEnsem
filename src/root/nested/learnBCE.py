import numpy as np
import scipy as sp
import math
import scipy.io as io
import scipy.special as special

from bceEstep import *
from bceMstep import *

def learnBCE(X, oldAlpha, oldBeta, lap, Q):
    #
    # BCE learning
    # 
    #   k = number of ensemble clusters
    #   N = number of base clusterings
    #   M = number of data points
    #
    # Input:
    #   X:          M*N, base clustering results
    #   oldAlpha:   k*1, model parameter for Dirichlet distribution
    #   oldBeta:    cell with N elements for N base clusterings, each element is a
    #               k*q matrix, i.e., k parameters for a q-dimensional discrete distribution
    #   lap:        smoothing parameter
    #   Q:          cell with N elements, each is the number of clusters in base
    #               clustering results         
    # Output:
    #   phiAll:     k*N*M, variational parameters for discrete distributions
    #   gamaAll:    k*M, variational parameters for Dirichlet distributions
    #--------------------------------------------------------------------

    [M, N] = X.shape
    k = np.size(oldAlpha)

    # initial value and variables for iteration
    alpha_t = oldAlpha
    beta_t = oldBeta
    epsilon = 0.01
    time = 500
    e = 100
    t = 1

    # start learning iterations
    print ('learning BCE')
    sample = np.zeros((M, N))
    phiAll = np.zeros((M, k, N))  # FM: added to make it compatible with python
                                  # note that it is assumed that phiAll is a 3D array
    gamaAll = np.zeros((k, M)) 
    while e > epsilon and t < time:
        # E-step
        for s in range(M):
            sample = np.array([X[s, :]])
            estimatedPhi, estimatedGama = bceEstep(alpha_t, beta_t, sample)
            phiAll[s, :, :] = estimatedPhi 
            gamaAll[:, s] = np.reshape(estimatedGama, (3,))

        # M-step
        alpha_tt, beta_tt = bceMstep(alpha_t, phiAll, gamaAll, X, Q, lap)

        # error
        upvalue = 0
        downvalue = 0
        for index in range(np.size(Q)):
            upvalue = upvalue + np.sum(np.sum(abs(beta_t[0, index] - beta_tt[0, index]), 0)) 
            downvalue = downvalue + np.sum(np.sum(beta_t[0, index], 0))
        e = upvalue / downvalue
        print ('t=', t, ', error=', e)

        # update
        alpha_t = alpha_tt
        beta_t = beta_tt

        t = t + 1

    resultAlpha = alpha_t
    resultBeta = beta_t


    return phiAll, gamaAll, resultAlpha, resultBeta
