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


#create features
#saves list to npy file
#input file name (read), file name (write)
#output list of features
def readFeatures(file1,file2):
    x=[]
    with open(file1) as f:
        lines=f.readlines()
    
    for line in lines:
        x.append(line.split())
        
    for l in range(len(x)):
        for j in range(len(x[l])):
            x[l][j]=float(x[l][j])
    np.save(file2,x)
    return x

def combineNPY(files,save):
	bigarray = np.load(files[0])	
	for f in range(1,len(files)):
		temp = np.load(files[f])
		bigarray = np.concatenate((bigarray,temp))
	np.save(save,bigarray)
	return bigarray	

#files = ["distances_hclust_average_new_dist_train1.npy", "distances_hclust_average_new_dist_train2.npy", "distances_hclust_average_new_dist_train3.npy", "distances_hclust_average_new_dist_train4.npy", "distances_hclust_average_new_dist_train5.npy"]
#files = ["distances_hclust_dba_average_new_dist_train1.npy", "distances_hclust_dba_average_new_dist_train2.npy", "distances_hclust_dba_average_new_dist_train3.npy", "distances_hclust_dba_average_new_dist_train4.npy", "distances_hclust_dba_average_new_dist_train5.npy"]
#files = ["PUCK_distances_hclust_dba_average_new_dist_test1.npy", "PUCK_distances_hclust_dba_average_new_dist_test2.npy", "PUCK_distances_hclust_dba_average_new_dist_test3.npy", "PUCK_distances_hclust_dba_average_new_dist_test4.npy", "PUCK_distances_hclust_dba_average_new_dist_test5.npy"]


files = []

template = "dist/syn2_distance_hclust_dba_average_origdist_train"

save = template+"_all.npy"

num = 10

for f in range(num):
	files.append(template+str(f)+".npy")


combineNPY(files,save)

#files = ["dist/distance_hclust0.25_dba_average_origdist_train1.npy", "dist/distance_hclust0.25_dba_average_origdist_train2.npy", "dist/distance_hclust0.25_dba_average_origdist_train3.npy", "dist/distance_hclust0.25_dba_average_origdist_train4.npy", "dist/distance_hclust0.25_dba_average_origdist_train5.npy"]

#combineNPY(files)
#readFeatures("/Users/skyler/Desktop/Cornell First Year/CS6780 Project/Code/UCI HAR Dataset/test/X_test.txt", "X_test_UCI_features.npy")
#readLabels("/Users/skyler/Desktop/Cornell First Year/CS6780 Project/Code/UCI HAR Dataset/test/y_test.txt", "UCI_HAR_test_labels.npy")


