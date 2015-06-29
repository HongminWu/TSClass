import numpy as np
from numpy import linalg as la
import TimeSeriesMethods as ts
import random as r
import sklearn.svm as svm
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

dist_clusts = np.load("dist_cluster.npy")
clusters = np.load("cluster.npy")
cluster_labels = np.load("ycluster.npy")

#print len(dist_clusts[0]), len(dist_clusts[0][0]), len(dist_clusts[0][0][0])
#print len(clusters[0])

def findClosest(cluster):
	dists = []
	for k in dist_clusts[cluster]:
		#print sum(k)
		dists.append(sum(k))
	return np.argmin(dists)

def buildTemplates():
	cluster_random = []
	cluster_templates_aligned = []
	for l in range(len(clusters)):
		cluster_random.append(clusters[l][findClosest(l)])
	for j in range(len(clusters)):
		for k in range(len(clusters[j])):
			clusters[j][k] = ts.align(cluster_random[j], clusters[j][k])
	for l in clusters:
		cluster_templates_aligned.append(ts.average(l))

	np.save("cluster_mindist_templates_hclust_average_new_dist.npy",cluster_templates_aligned)
	return cluster_templates_aligned

def classifySVM():
	dist_features_train = np.load("distances_hclust_mindist_average_new_dist_train_all.npy")
	dist_features_test = np.load("distances_hclust_mindist_average_new_dist_test_all.npy")
	yTest = np.load("yTestC.npy")
	yTrain = np.load("yTrainC.npy")
	test_labels = yTest
	train_labels = yTrain
	maxs = None
	maxcm = None
	Xtrain = dist_features_train
	Xtest = dist_features_test
	q = []
	for j in range(len(dist_features_train[0])):
		pca = PCA(n_components=j)    
		Xtrain = pca.fit(dist_features_train).transform(dist_features_train)

		Xtest = pca.transform(dist_features_test)
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
		s = 0
		cm=0
		ss = svm.LinearSVC().fit(Xtrain,yTrain)
		y_pred = ss.predict(Xtest)
		cm = confusion_matrix(yTest,y_pred)
		q.append(ss.score(Xtest,yTest))
		print j, ss.score(Xtest,yTest)
	return np.argmax(q). max(q)
	'''
	for k in range(100):
		ss = svm.LinearSVC().fit(Xtrain,yTrain)
		y_pred =  ss.predict(Xtest)
		cm += confusion_matrix(yTest,y_pred)
		s+=ss.score(Xtest, yTest)
	q.append((s,cm))
	'''
		
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
		print cluster[k]
		print cluster[k]
		if predicted_labels[p] == (test_labels[p]):
			accuracy+=1

	np.save("confusion_matirx_hclust_new_dist_less_labels"+str(num)+".npy",confusion_matrix)
	return confusion_matrix, float(accuracy)/len(test_labels)

	


def getDistances(test,test_labels,train,train_labels,num):
	testTS = np.load(test)
	test_labels = np.load(test_labels)
	
	#cluster_templates =  buildTemplates()
	#cluster_templates = np.load("cluster_templates_hclust_average_new_dist.npy",)
	cluster_templates = np.load("cluster_mindist_templates_hclust_average_new_dist.npy")
	#build predited labels'
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
	np.save("distances_hclust_mindist_average_new_dist_test"+str(num)+".npy",distancefeatures)
	
	dist_features_train = []
	xTrain = np.load(train)
	yTrain=np.load(train_labels)
	for k in xTrain:
		dist = []
		for temp in cluster_templates:
			dist.append(ts.DTWsubseq(k,temp)[1])
		dist_features_train.append(dist)
	np.save("distances_hclust_mindist_average_new_dist_train"+str(num)+".npy",dist_features_train)

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

#buildTemplates() 

print classifySVM()
#print getDistances("xTestC_5.npy","yTestC_5.npy","xTrainC_5.npy", "yTrainC_5.npy",5)
#print classify("PUCK_xTestC_5.npy", "PUCK_yTestC_5.npy",5)


#np.save("PUCK_xTrainC_1.npy",np.load("PUCK_xTrainC.npy")[0:int(len(np.load("PUCK_xTrainC.npy"))/5)])
#np.save("PUCK_xTrainC_2.npy",np.load("PUCK_xTrainC.npy")[int(len(np.load("PUCK_xTrainC.npy"))/5):int(len(np.load("PUCK_xTrainC.npy"))*2/5)])
#np.save("PUCK_xTrainC_3.npy",np.load("PUCK_xTrainC.npy")[int(len(np.load("PUCK_xTrainC.npy"))*2/5):int(len(np.load("PUCK_xTrainC.npy"))*3/5)])
#np.save("PUCK_xTrainC_4.npy",np.load("PUCK_xTrainC.npy")[int(len(np.load("PUCK_xTrainC.npy"))*3/5):int(len(np.load("PUCK_xTrainC.npy"))*4/5)])
#np.save("PUCK_xTrainC_5.npy",np.load("PUCK_xTrainC.npy")[int(len(np.load("PUCK_xTrainC.npy"))*4/5):int(len(np.load("PUCK_xTrainC.npy")))])	
