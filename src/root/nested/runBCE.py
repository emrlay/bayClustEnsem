# Author: Farzan Memarian
# Affiliation: PhD student at University of Texas at Austin
# 
# This python code consists of 5 modules, runBCE, bceEstep, bceMstep, calculateAccuracy and learnBCE. 
# It is developed based on Python 3 formatting.  
# It is based on the MATLAB code provided by authors of the following paper:
# H. Wang, H. Shan, A. Banerjee. Bayesian Cluster Ensembles. Statistical Analysis and Data Mining, 2011

import numpy as np
import scipy as sp
import math
import scipy.io as io
import scipy.special as special

from learnBCE import *
from calculateAccuracy import *



def main():
    # load the data and initial model parameters
    #
    #   k = number of ensemble clusters
    #   N = number of base clusterings
    #   M = number of data points
    # 
    # base_labels:              M*N, base clustering results to be processed
    # Palpha:                   k*1, initial value for the model parameter of Dirichlet distribution
    # Pbeta:                    cell with N elements for N base clusterings, each element is a
    #                           k*q matrix, i.e., initial value for k parameters of a q-dimensional discrete distribution
    # number_baseclusterers:    1*N, number of clusters in each of N base clustering results
    
    intVec = np.vectorize(int)  # FM: vectorized version of int function to be applied on an array
    mat = io.loadmat('Iris.mat')  # FM: mat is a dictionary the way Iris.mat is loaded in python. 
    # FM: Note that in order for the code to run properly the following information 
    # must be provided with the specific format mentioned:
    # base_labels:              an array of size M*N
    # true_labels:              an array of shape 1*M
    # number_baseclusterers:    an array of shape 1*N
    # Palpha:                   an array of shape k*1
    # Pbeta:                    an array of arrays, which consists of an outer array
    #                           of size 1*N, each of these arrays is of size k*q
    base_labels = mat["base_labels"]
    true_labels = mat["true_labels"]
    number_baseclusterers = mat["number_baseclusterers"]
    Palpha = mat["Palpha"]
    Pbeta = mat["Pbeta"]
    
    
    # PramaLap:                 parameter for laplace smoothing
    PramaLap = 0.000001

    # If use random initialization
    # Palpha = rand(size(Palpha));
    # for i = 1:length(Pbeta)
    #     temp = (size(Pbeta{i}));
    #     [k, q] = size(temp);
    #     temp = temp ./ (sum(temp, 2) * ones(1, q));
    #     Pbeta{i} = temp;
    # end


    
    # learn BCE 
    phiAll, gammaAll, resultAlpha, resultBeta = learnBCE(base_labels, Palpha, Pbeta, PramaLap, \
                                                         number_baseclusterers)

    # calculate accuracy
    k = np.size(np.unique(true_labels))
    M = np.size(true_labels)

    # Obtain the cluster assignments from BCE
    Ensemble_labels = np.zeros((1, M))
    wtheta = np.zeros((k, M))  # FM: initializing wtheta for python version.
    for index in range(M):
        wtheta[:, index] = gammaAll[:, index]
        bb = np.nonzero(intVec(wtheta[:, index] == max(wtheta[:, index])))
        Ensemble_labels[0, index] = bb[0] + 1

    # Calculate the accuracy based on true labels and BCE results
    accu = calculateAccuracy(true_labels, Ensemble_labels)
    print ('The micro-precision of BCE is ', accu)
if __name__ == "__main__":
    main()
