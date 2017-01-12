import numpy as np
import scipy as sp
import math
import scipy.io as io
import scipy.special as special


def calculateAccuracy(true_labels, Ensemble_labels):
    #
    # Calculate  micro-precision given clustering results and true labels.
    #
    #   k = number of ensemble clusters
    #   M = number of data points
    #
    # Input:
    #   true_labels:        1*M, true class labels for the data points
    #   Ensemble_labels:    1*M, labels obtained from BCE
    #   
    # Output:
    #   micro_precision:    micro-precision
    #--------------------------------------------------------------------
    
    k = np.size(np.unique(true_labels))
    M = np.size(true_labels)
    intVec = np.vectorize(int)  # FM: vectorized version of int function to be applied on an array
    accurence = np.zeros((k, k))
    for j in range(k):
         for jj in range(k):
            accurence[j, jj] = np.shape(np.nonzero(intVec((intVec(Ensemble_labels == (jj + 1)) * (j + 1)) == true_labels)))[1]
    [rowm, coln] = np.shape(accurence)
    amatrix = accurence
    sumMax = 0
    while rowm >= 1:
        xx = np.amax(np.amax(amatrix, 0), 0)
        [x, y] = np.nonzero(intVec(amatrix == xx)) 
        sumMax = sumMax + xx                      
        iyy = 0
        temp = np.zeros((rowm, rowm - 1))
        for iy in range(rowm):
            if iy == y[0]:
                continue  
            else:                        
                temp[:, iyy] = amatrix[:, iy]
                iyy = iyy + 1
        temp2 = np.zeros((rowm - 1, rowm - 1))
        ixx = 0
        for ix in range(rowm):
            if ix == x[0]:
                continue
            else:                       
                temp2[ixx, :] = temp[ix, :]
                ixx = ixx + 1
        rowm = rowm - 1
        amatrix = np.zeros((rowm, rowm))
        amatrix = temp2
    
    micro_precision = sumMax / M
    return micro_precision
