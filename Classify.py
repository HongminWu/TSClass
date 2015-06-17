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


#file 1 is time series, file 2 is labels
#saves confusion matrix
#outputs confusion amtrix, accuracy
def classify(file1,file2):
	testTS = np.load(file1)
	test_labels = np.load(file2)
	cluster_templates = buildTemplates()

	#build predited labels
	predicted_labels = []
	i=0
	for sample in testTS:
		dist = []
		print "sample", i
		for template in cluster_templates:
			dist.append(ts.DTWsubseq(sample,template)[1])
		predicted_labels.append(cluster_labels[np.argmin(dist)])
		i+=1
	np.save("predicted_labels_hclust_average_lessweird.npy", predicted_labels)
	#predicted_labels = np.load("predicted_labels_hclust_average_lessweird.npy")

	#accuracy and confusion matrix
	accuracy = 0
	confusion_matrix = np.zeros(shape=(6,6))	
	for p in range(len(predicted_labels)):
		confusion_matrix[predicted_labels[p]][test_labels[p]]+=1
		print predicted_labels[p], test_labels[p]
		if predicted_labels[p] == (test_labels[p]):
			accuracy+=1

	np.save("confusion_matirx_hclust.npy",confusion_matrix)
	return confusion_matrix, float(accuracy)/len(test_labels)

print classify("xTestC.npy", "yTestC.npy")



	
