# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 16:57:43 2019

@author: T
"""

#importing libraries for KNN 
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor 
from copy import deepcopy


#normalizing function
def normalize(X): # max-min normalizing
    N, m = X.shape
    Y = np.zeros([N, m])
    
    for i in range(m):
        mX = min(X[:,i])
        Y[:,i] = (X[:,i] - mX) / (max(X[:,i]) - mX)
    
    return Y

#spliting dataset 
def split_dataset(data, r):
	N = len(data)	
	X = []
	Y = []
	
	if r >= 1: 
		print ("Parameter r needs to be smaller than 1!")
		return
	elif r <= 0:
		print ("Parameter r needs to be larger than 0!")
		return

#
	n = int(round(N*r)) 
	nt = N - n 
	ind = -np.ones(n,int) 
	R = np.random.randint(N) 
	
	for i in range(n):
		while R in ind: R = np.random.randint(N) 
		ind[i] = R

	ind_ = list(set(range(N)).difference(ind)) 
	X = data[ind_,:-1] 
	XX = data[ind,:-1] 
	Y = data[ind_,-1] 
	YY = data[ind,-1] 
	return X, XX, Y, YY

#importing data set and selecting features 
r = 0.2 
dataset = pd.read_csv(r'C:\Users\T\Desktop\Python\Data science Uwash/TariqAyub-Dataset.csv')
all_inputs = normalize(dataset[:,:4]) # inputs (features)
normalized_data = deepcopy(dataset)
normalized_data[:,:4] = all_inputs
X, XX, Y, YY = split_dataset(normalized_data, r)

# kNN regression
print ("\n\nK Nearest Neighbors Regression\n")
k = 5 # number of neighbors to be used
distance_metric = 'euclidean'
#applying the KNN algorithm 
regr = KNeighborsRegressor(n_neighbors=k, metric=distance_metric)
#training the dataset
regr.fit(X, Y)
print ("predictions for test set:")
#predictive the test dataset
print (regr.predict(XX))
#printing the actual values of the test set
print ('actual target values:')
print (YY)
