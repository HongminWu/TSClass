import numpy as np
from numpy import linalg as la
import TimeSeriesMethods as ts
import random as r
clusters = np.load("cluster.npy")
cluster_labels = np.load("ycluster.npy")


def buildTemplates():
	cluster_random = []
	cluster_templates_aligned = []
	for l in clusters:
		cluster_random.append(l[r.randint(0,len(l)-1)])
	for j in range(len(clusters)):
		for k in range(len(clusters[j])):
			clusters[j][k] = ts.align(cluster_random[j], clusters[j][k])
	for l in clusters:
		cluster_templates_aligned.append(ts.average(l))
	return cluster_templates_aligned

def classifySVM(file1,file2,num):
	dist_features_test = np.load("distances_hclust_average_new_dist.npy")
	xTrain = np.load(file1)
	yTrain=np.load(file2)
	yTest = np.load("yTestC.npy")
	cluster_temps = np.load("cluster_templates_hclust_average_new_dist.npy")
	dist_features_train = []
	for k in xTrain:
		dist = []
		for temp in cluster_temps:
			dist.append(ts.DTWsubseq(k,temp)[1])
		dist_features_train.append(dist)
	np.save("distances_hclust_average_new_dist.npy")
        #Xtest = pca.transform(feature_test)
	y_pred =  svm.LinearSVC().fit(dist_features_train,label_train).predict(distance_features_test)
	cm = confusion_matrix(label_test,y_pred)
	s=svm.LinearSVC().fit(dist_features_train, yTrain).score(dist_features_test, yTest)
        return cm, s
	
#file 1 is time series, file 2 is labels
#saves confusion matrix
#outputs confusion amtrix, accuracy
def getDistances(file1,file2,num):
	testTS = np.load(file1)
	test_labels = np.load(file2)
	cluster_templates = buildTemplates()
	np.save("cluster_templates_hclust_average_new_dist.npy",cluster_templates)
	#cluster_templates = np.load("dba_templates.npy")
	#build predited labels
	'''
	for k in range(len(test_labels)):
	    if test_labels[k]>2:
		test_labels[k]=3
	for w in range(len(cluster_labels)):
	    if cluster_labels[w]>2:
		cluster_labels[w]=3
	'''
        distancefeatures = []
	predicted_labels = []
	i=0
	for sample in testTS:
		dist = []
		print "sample", i
		for template in cluster_templates:
			dist.append(ts.DTWsubseq(sample,template)[1])
		distancefeatures.append(dist)
		#predicted_labels.append(cluster_labels[np.argmin(dist)])
		i+=1
	np.save("distances_hclust_average_new_dist.npy",distancefeatures)
	np.save("predicted_labels_hclust_average_new_dist_less_labels.npy", predicted_labels)
	#predicted_labels = np.load("predicted_labels_hclust_average_lessweird.npy")
	'''
	#accuracy and confusion matrix
	accuracy = 0
	confusion_matrix = np.zeros(shape=(6,6))	
	for p in range(len(predicted_labels)):
		confusion_matrix[predicted_labels[p]][test_labels[p]]+=1
		print predicted_labels[p], test_labels[p]
		if predicted_labels[p] == (test_labels[p]):
			accuracy+=1

	np.save("confusion_matirx_hclust_new_dist_less_labels"+str(num)+".npy",confusion_matrix)
	return confusion_matrix, float(accuracy)/len(test_labels)
	'''
print getDistances("xTestC.npy","yTestC.npy",0)
#print classify("PUCK_xTestC_5.npy", "PUCK_yTestC_5.npy",5)
#np.save("PUCK_xTestC_1.npy",np.load("PUCK_xTestC.npy")[0:int(len(np.load("PUCK_xTestC.npy"))/5)])
#np.save("PUCK_xTestC_2.npy",np.load("PUCK_xTestC.npy")[int(len(np.load("PUCK_xTestC.npy"))/5):int(len(np.load("PUCK_xTestC.npy"))*2/5)])
#np.save("PUCK_xTestC_3.npy",np.load("PUCK_xTestC.npy")[int(len(np.load("PUCK_xTestC.npy"))*2/5):int(len(np.load("PUCK_xTestC.npy"))*3/5)])
#np.save("PUCK_xTestC_4.npy",np.load("PUCK_xTestC.npy")[int(len(np.load("PUCK_xTestC.npy"))*3/5):int(len(np.load("PUCK_xTestC.npy"))*4/5)])
#np.save("PUCK_xTestC_5.npy",np.load("PUCK_xTestC.npy")[int(len(np.load("PUCK_xTestC.npy"))*4/5):int(len(np.load("PUCK_xTestC.npy")))])	
