import numpy as np
import scipy as sp
import math
import scipy.io as io
import scipy.special as special


def bceEstep(alpha, beta, x):
    # 
    # E step of BCE
    # 
    #   k = number of ensemble clusters
    #   N = number of base clusterings
    #   M = number of data points
    # 
    # Input:
    #   alpha:  k*1, parameter for Dirichlet distribution
    #   beta:   cell with N elements for N base clusterings, each element is a
    #           k*q matrix, i.e., k parameters for a q-dimensional discrete distribution
    #   x:      1*N, base clustering results for one data point. 0 indicates
    #           missing base clustering results
    # 
    # output:
    #   phi_t:  k*N, variational parameter for discrete distribution
    #   gama_t: k*1, variational parameter for Dirichlet distribution
    #-------------------------------------------------
    
    k = np.size(alpha)
    N = np.size(x)
    V = np.size(np.nonzero(x))
    intVec = np.vectorize(int)  # FM: vectorized version of int function to be applied on an array
    fil = np.matmul(np.ones((k, 1)), intVec((x != 0)))  # FM: matrix product of two arrays
    
    # initial value for variational parameters
    phi_t = (np.ones((k, N)) * fil) / k
    gama_t = alpha + V / k
    
    # variables for iteration
    epsilon = 0.01
    time = 500
    e = 100
    t = 1
    tempBeta = np.zeros((k, N))  # FM: added to initialize tempBeta
    
    for i in range(k):
        for n in range(N):
            if (x[0, n] != 0):
               tempBeta[i, n] = beta[0, n][i, x[0, n] - 1]  
            else:
               tempBeta[i, n] = -1
    
    
    # Continue iteration, if the error is larger than the threshold, or
    # iteration time is smaller than the predefined steps.
    realmin = np.finfo(np.double).tiny 
    while e > epsilon and t < time:
        # new phi
        phi_tt = np.exp(np.matmul((special.psi(gama_t) - special.psi(np.sum(gama_t, 0))), np.ones((1, N)))) * tempBeta
        phi_tt = phi_tt / np.matmul(np.ones((k, 1)), np.reshape(np.sum(phi_tt + realmin, 0), (1, N)))
        phi_tt = phi_tt * fil
        
        # new gamma
        gama_tt = alpha + np.reshape(np.sum(phi_tt, 1), (k, 1))  # FM: reshape has been used for np.sum to make it vertical
        
        # error of the iteration
        e1 = np.sum(np.sum(np.absolute(phi_tt - phi_t), 0)) / np.sum(np.sum(phi_t, 0))
        e2 = sum(np.absolute(gama_tt - gama_t), 0) / sum(gama_t, 0)
        e = max(e1, e2)
        
        # update the variational parameters
        phi_t = phi_tt
        gama_t = gama_tt
        # disp(['t=',intVecstr(t),', e1,e2,e:',num2str(e1),',',num2str(e2),',',num2str(e)])
        t = t + 1
    
    
    return phi_t, gama_t
