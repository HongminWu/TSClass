import numpy as np
from numpy import linalg as la
import TimeSeriesMethods as ts
import random as r
import sklearn.svm as svm
from sklearn.metrics import confusion_matrix

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

	np.save("cluster_templates_hclust_average_new_dist.npy",cluster_templates_aligned)
	return cluster_templates_aligned

def classifySVM():
	dist_features_train = np.load("distances_hclust_average_new_dist_train_all.npy")
	dist_features_test = np.load("distances_hclust_average_new_dist_test_all.npy")
	yTest = np.load("yTestC.npy")
	yTrain = np.load("yTrainC.npy")
	test_labels = yTest
	train_labels = yTrain
	'''
	for k in range(len(test_labels)):
            if test_labels[k]>2:
                test_labels[k]=3
        for w in range(len(cluster_labels)):
            if cluster_labels[w]>2:
                cluster_labels[w]=3

	for k in range(len(train_labels)):
            if train_labels[k]>2:
                train_labels[k]=3
        for w in range(len(cluster_labels)):
            if cluster_labels[w]>2:
                cluster_labels[w]=3
	'''

	y_pred =  svm.LinearSVC().fit(dist_features_train,yTrain).predict(dist_features_test)
	cm = confusion_matrix(yTest,y_pred)
	s=svm.LinearSVC().fit(dist_features_train, yTrain).score(dist_features_test, yTest)
        return cm, s
	
def classify():
	predicted_labels=[]
	cluster_templates = np.load("cluster_templates_hclust_average_new_dist.npy",)

	dist_features_train = np.load("distances_hclust_average_new_dist_train.npy")
	dist_features_test = np.load("distances_hclust_average_new_dist_test.npy")
	yTest = np.load("yTestC.npy")
	yTrain = np.load("yTrainC.npy")
	for k in dist_features_test:
		predicted_labels.append(np.argmin(k))
	accuracy = 0
	confusion_matrix = np.zeros(shape=(6,6))	
	for p in range(len(predicted_labels)):
		confusion_matrix[predicted_labels[p]][test_labels[p]]+=1
		print predicted_labels[p], test_labels[p]
		if predicted_labels[p] == (test_labels[p]):
			accuracy+=1

	np.save("confusion_matirx_hclust_new_dist_less_labels"+str(num)+".npy",confusion_matrix)
	return confusion_matrix, float(accuracy)/len(test_labels)

	


def getDistances(test,test_labels,train,train_labels,num):
	testTS = np.load(test)
	test_labels = np.load(test_labels)
	#cluster_templates =  buildTemplates()
	cluster_templates = np.load("cluster_templates_hclust_average_new_dist.npy",)
	#cluster_templates = np.load("dba_templates.npy")
	#build predited labels'
    	distancefeatures = []
	predicted_labels = []
	i=0
	'''
	for sample in testTS:
		dist = []
		print "sample", i
		for template in cluster_templates:
			dist.append(ts.DTWsubseq(sample,template)[1])
		distancefeatures.append(dist)
		#predicted_labels.append(cluster_labels[np.argmin(dist)])
		i+=1
	np.save("distances_hclust_average_new_dist_test"+str(num)+".npy",distancefeatures)
	'''
	dist_features_train = []
	xTrain = np.load(train)
	yTrain=np.load(train_labels)
	for k in xTrain:
		dist = []
		for temp in cluster_templates:
			dist.append(ts.DTWsubseq(k,temp)[1])
		dist_features_train.append(dist)
	np.save("distances_hclust_average_new_dist_train"+str(num)+".npy",dist_features_train)

	'''
	for k in range(len(test_labels)):
	    if test_labels[k]>2:
		test_labels[k]=3
	for w in range(len(cluster_labels)):
	    if cluster_labels[w]>2:
		cluster_labels[w]=3
	'''

	return dist_features_train,distancefeatures

	#np.save("predicted_labels_hclust_average_new_dist_less_labels.npy", predicted_labels)
	#predicted_labels = np.load("predicted_labels_hclust_average_lessweird.npy")

print classifySVM()
#print getDistances("xTestC_5.npy","yTestC_5.npy","xTrainC_5.npy", "yTrainC_5.npy",5)
#print classify("PUCK_xTestC_5.npy", "PUCK_yTestC_5.npy",5)
#np.save("yTrainC_1.npy",np.load("yTrainC.npy")[0:int(len(np.load("yTrainC.npy"))/5)])
#np.save("yTrainC_2.npy",np.load("yTrainC.npy")[int(len(np.load("yTrainC.npy"))/5):int(len(np.load("yTrainC.npy"))*2/5)])
#np.save("yTrainC_3.npy",np.load("yTrainC.npy")[int(len(np.load("yTrainC.npy"))*2/5):int(len(np.load("yTrainC.npy"))*3/5)])
#np.save("yTrainC_4.npy",np.load("yTrainC.npy")[int(len(np.load("yTrainC.npy"))*3/5):int(len(np.load("yTrainC.npy"))*4/5)])
#np.save("yTrainC_5.npy",np.load("yTrainC.npy")[int(len(np.load("yTrainC.npy"))*4/5):int(len(np.load("yTrainC.npy")))])	
