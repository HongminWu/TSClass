import numpy as np
from numpy import linalg as la
from copy import deepcopy


#read a time series 
#input file name (read)
#output list
def readTimeSeries(file1):
    accx = []
    with open(file1) as f:
        lines=f.readlines()
    for line in lines:
        accx.append(line.split())
    for l in range(len(accx)):
        for j in range(len(accx[l])):
            accx[l][j]=float(accx[l][j])
    return accx

#create multivariate time series
#saves list to a npy file
#input list of files (read), file name (write)
#output multivariate time series, list of univariate time series
def multivariateTimeSeries(files,file2):
    timeSeries = []
    for f in files:
    	timeSeries.append(readTimeSeries(f))
    multTS = deepcopy(timeSeries[0])
    for k in range(len(timeSeries[0])):
        for l in range(len(timeSeries[0][k])):
            multTS[k][l]=[timeSeries[0][k][l], timeSeries[1][k][l], timeSeries[2][k][l], timeSeries[3][k][l], timeSeries[4][k][l], timeSeries[5][k][l]]
    np.save(file2,multTS)
    return multTS, timeSeries

#create labels or subjects
#saves list to a npy file
#input file name (read), file name (write)
#output list of labels or 
def readLabels(file1,file2):
    activity = []
    with open(file1) as f:
        liner=f.readlines()
    for val in liner:
        activity.append(int(val)-1)
    np.save(file2,activity)
    return activity

readLabels("/Users/skyler/Desktop/Cornell First Year/CS6780 Project/Code/UCI HAR Dataset/test/y_test.txt", "UCI_HAR_test_labels.npy")


