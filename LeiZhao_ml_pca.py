# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 20:51:13 2017
Spyder(Python 3.6)
@author: LeiZhao"""

import sys
sys.modules[__name__].__dict__.clear()    #clear workspace


import os
import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
## Question 1

path="C:\\Users\LeiZhao\cygwin64\home\LeiZhao"
os.chdir(path)
print (os.getcwd());

### read input data ###
file = pd.read_csv('dataset_1.csv', thousands=',')
fileMatrix = np.matrix(file)
fileMatrix_xy = fileMatrix[:,0:1]
fileMatrix_yz = fileMatrix[:,1:2]
colNum = fileMatrix.shape[1]
rowNum = fileMatrix.shape[0]

### sub_question_1: variance of each variable
np.var(fileMatrix[:,0])    # for x
np.var(fileMatrix[:,1])    # for y
np.var(fileMatrix[:,2])    # for z

### sub_questin_2: covariance between x and y, and between y and z
covariance_xy = np.cov(fileMatrix_xy,rowvar=False)
covariance_yz = np.cov(fileMatrix_yz,rowvar=False)

### sub_question_3: do PCA for all the data with own module

meanVector = fileMatrix.mean(axis=0)    ### mean of each column
meanMatrix = np.repeat(meanVector,rowNum,axis=0)    ### expand mean vector to a matrix
meanCenteredMatrix = fileMatrix - meanMatrix    ### get mean centered matrix
covariance = np.cov(meanCenteredMatrix,rowvar=False)    ### get covariance matrix
values,vectors = LA.eigh(covariance)    ### get eigenvalues & eigenvectors
print("eigenvalues:\n " + str(values))
print("eigenvectors:\n " + str(vectors))
prinComp = meanCenteredMatrix.dot(vectors)    ### get principle components matrix

#print(values[0]/np.sum(values))
#print(values[1]/np.sum(values))
#print(values[2]/np.sum(values))

### how much variances can be explained by each component
result = []
for n in range(0,colNum):
    a="%.2f%%" % (values[n]/np.sum(values)*100)
    result.append(a)
    realCol = n+1
    print("principle component " + str(realCol) + " can explain " + str(a) + " variances")

#print("The first component can explain " + a + " variances")
#print("The second component can explain " + b + " variances")
#print("The third component can explain " + c + " variances")

"""
Output did not sort the compnents automatically.
Need to get the column numbers of the prinComp matrix that have the biggest two eigenvalues
"""
indices = np.argpartition(values, -2)[-2:]
print ("So, compnent " + str(indices[1]+1) + " and compnent " + str(indices[0]+1) + " should be used for PCA")

plt.scatter(np.array(prinComp[:,indices[1]]),np.array(prinComp[:,indices[0]]),c='m')
plt.title("PCA of dataset_1")
plt.xlabel("PC1")
plt.ylabel("PC2")

## Question 3(2)
a = np.array([[0,-1],[2,3]],dtype=int)
values_q3,vectors_q3 = LA.eig(a)
values_q3
vectors_q3
