import numpy as np
import scipy as sp
import math
import scipy.io as io
import scipy.special as special


def bceMstep(alpha, phi, gama, X, Q, lap):
    #
    # M-step of BCE
    #
    #   k = number of ensemble clusters
    #   N = number of base clusterings
    #   M = number of data points
    #
    # Input:
    #   phi:    k*N*M, variational parameters for discrete distributions
    #   gama:   k*M, variational parameters for Dirichlet distributions
    #   alpha:  k*1, model parameters for the Dirichlet distribution
    #   X:      M*N, base clustering results
    #   Q:      cell with N elements, each is the number of clusters in base
    #           clustering results
    #   lap:    laplacian smoothing parameter
    #
    # Output:
    #   alpha:  k*1, model paramter for Dirichlet distribution
    #   beta:   cell with N elements for N base clusterings, each element is a
    #           k*q matrix, i.e., k parameters for a q-dimensional discrete distribution   
    # -------------------------------------------------
    
    [M, k, N] = np.shape(phi) 
    
    beta = np.empty((1, N), np.ndarray)  # FM: added to initialize beta as an array of arrays
    #-------update beta----------
    for ind in range(N):
        beta[0, ind] = np.zeros((k, Q[0, ind]))
    
    
    intVec = np.vectorize(int)  # FM: vectorized version of int
    
    for ind in range(N):
        for q in range(Q[0, ind]):
            temp = np.zeros((k, N))
            for s in range(M):
                x = np.array([X[s, :]])
                fil = np.matmul(np.ones((k, 1)), intVec(x == (q + 1)))   
                temp = temp + phi[s, :, :] * fil 
            beta[0, ind][:, q] = temp[:, ind]
    
    # smoothing
    for ind in range(N):
        beta[0, ind] = beta[0, ind] + lap
        beta[0, ind] = beta[0, ind] / np.matmul(np.reshape(np.sum(beta[0, ind], 1), \
                                                (k, 1)), (np.ones((1, Q[0, ind])))) 
    
    
    # -------update alpha-----------
    
    alpha_t = alpha
    epsilon = 0.001
    time = 500
    
    t = 0
    e = 100
    psiGama = special.psi(gama)
    psiSumGama = special.psi(sum(gama, 0))
    while e > epsilon and t < time:
        g = np.reshape(np.sum((psiGama - np.matmul(np.ones((k, 1)), np.reshape(psiSumGama, (1, M)))), 1), (k, 1)) \
                                + M * (special.psi(np.sum(alpha_t, 0)) - special.psi(alpha_t))
        h = -M * special.polygamma(1, alpha_t)
        z = M * special.polygamma(1, np.sum(alpha_t, 0))
        c = np.sum((g / h), 0) / (1 / z + np.sum((1 / h), 0)) 
        delta = (g - c) / h
    
        # line search
        eta = 1
        alpha_tt = alpha_t - delta
        while (np.size(np.nonzero(intVec(alpha_tt <= 0))[0]) > 0):
            eta = eta / 2
            alpha_tt = alpha_t - eta * delta
        e = np.sum(abs(alpha_tt - alpha_t), 0) / np.sum(alpha_t, 0)
        
        alpha_t = alpha_tt
    
        t = t + 1
    
    alpha = alpha_t
    
    return alpha, beta
