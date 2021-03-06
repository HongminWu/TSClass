import numpy as np
from numpy import linalg as la
import TimeSeriesMethods as ts
import random as r
import sklearn.svm as svm
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import argparse

############################# CHANGE THESE###############################

### FILE 3: MNo modifier 0.25

dist_clusts = np.load("Syn5/syn5_dist_cluster0.25.npy")
clusters = np.load("Syn5/syn5_cluster0.25.npy")
cluster_labels = np.load("Syn5/syn5_ycluster0.25.npy")

#########################################################################


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

############################# CHANGE THESE###############################
### I guess this line is redundant

	# np.save("template/syn5_template_original.npy",cluster_templates_aligned)
#########################################################################

	return cluster_templates_aligned

def classifySVM():
############################# CHANGE THESE###############################

	dist_features_train = np.load("dist/distances_hclust_mindist_average_newdist_train_all.npy")
	dist_features_test = np.load("dist/distances_hclust_mindist_average_newdist_test_all.npy")
	print dist_features_train[0]

	yTest = np.load("yTestC.npy")
	yTrain = np.load("yTrainC.npy")

#########################################################################

	test_labels = yTest
	train_labels = yTrain
	maxs = None
	maxcm = None
	Xtrain = dist_features_train
	Xtest = dist_features_test
	q = []
	p = []
	mislabel = []
	'''
	for j in range(len(Xtrain[0])):
		pca = PCA(n_components=j)    
		Xtrain = pca.fit(dist_features_train).transform(dist_features_train)

		Xtest = pca.transform(dist_features_test)
	'''
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
	ss = svm.LinearSVC(random_state=420).fit(Xtrain,yTrain)
	y_pred = ss.predict(Xtest)
	xtest = np.load("xTestSyn2.npy")	
	'''
	for j,label in enumerate(y_pred):
	    if label != yTrain[j]:
		mislabel.append((xtest[j],label,yTrain[j]))
	np.save("Mislabeled_hclust_mindist_average_newdist.npy",mislabel)	
	'''
	cm = confusion_matrix(yTest,y_pred)
	s = ss.score(Xtest,yTest)
	q.append(ss.score(Xtest,yTest))
	p.append(cm)
	#print j, ss.score(Xtest,yTest)
	return np.argmax(q), max(q),p[np.argmax(q)] 


	'''
	for k in range(100,len(Xtrain[0])):
		pca=PCA(n_components=k)
		Xtrain = pca.fit(dist_features_train).transform(dist_features_train)

        	Xtest = pca.transform(dist_features_test)
                ss = svm.LinearSVC(random_state=420).fit(Xtrain,yTrain)
		y_pred =  ss.predict(Xtest)
		cm = confusion_matrix(yTest,y_pred)
		s=ss.score(Xtest, yTest)
		q.append(s)
		p.append(cm)
		print k,s
	return max(q),p[np.argmax(q)]
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

############################# CHANGE THESE###############################

	
	cluster_templates =  buildTemplates()
	#cluster_templates = np.load("syn_cluster_templates_hclust_average_new_dist.npy",)
	#cluster_templates = np.load("syn_cluster10_mindist_templates_hclust_average_new_dist.npy")
	#build predited labels'
	#cluster_templates = np.load("template/syn2_dba_templates_original0.25.npy")
	#cluster_templates = np.load("template/cluster0.25_mindist_templates_hclust_average_subseq_newdist.npy")

	# cluster_templates = np.load("Syn5/syn5_dba_templates_original.npy")
#########################################################################

	
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

############################# CHANGE THESE###############################

	np.save("dist/syn5_distance_0.25_test"+str(num)+".npy",distancefeatures)

#########################################################################
		
	dist_features_train = []
	xTrain = np.load(train)
	yTrain=np.load(train_labels)
	for k in xTrain:
		dist = []
		for temp in cluster_templates:
			dist.append(ts.DTWsubseq(k,temp)[1])
		dist_features_train.append(dist)

############################# CHANGE THESE###############################

	np.save("dist/syn5_distance_0.25_train"+str(num)+".npy",dist_features_train)
#########################################################################
	'''
	for k in range(len(test_labels)):
	    if test_labels[k]>2:
		test_labels[k]=3
	for w in range(len(cluster_labels)):
	    if cluster_labels[w]>2:
		cluster_labels[w]=3
	'''

	#return dist_features_train,distancefeatures

	#np.save("predicted_labels_hclust_average_new_dist_less_labels.npy", predicted_labels)
	#predicted_labels = np.load("predicted_labels_hclust_average_lessweird.npy")



parser = argparse.ArgumentParser("blahblahblah")
parser.add_argument("num", help="file to run")

args = parser.parse_args()
#buildTemplates() 

############################# CHANGE THIS FOR FILE NUMBER###############################

num=args.num

#########################################################################################


############################### USE THIS IF CLASSIFYING##################################

#print classifySVM()

#########################################################################################

############################### USE THIS TO CALCULATE THE DISTANCES #####################

# print getDistances("Syn5/xTestSyn5.npy", "Syn5/yTestSyn5.npy", "Syn5/xTrainSyn5.npy","Syn5/yTrainSyn5.npy",0)
print getDistances("Syn5/xTestSyn5_"+str(num)+".npy","Syn5/yTestSyn5_"+str(num)+".npy","Syn5/xTrainSyn5_"+str(num)+".npy", "Syn5/yTrainSyn5_"+str(num)+".npy",num)

################################### USE THIS TO SPLIT DATA IF YOU NEED TO################
'''
f = "Syn5/xTestSyn5"
n = 10

for j in range(n):
	np.save(f+"_"+str(j)+".npy",np.load(f+".npy")[int(len(np.load(f+".npy"))*j/n):int(len(np.load(f+".npy"))*(j+1)/n)])
'''

#np.save(f+"_1.npy",np.load(f)[0:int(len(np.load(f))/5)])
#np.save("xTrainSyn2_2.npy",np.load("xTrainSyn2.npy")[int(len(np.load("PUCK_xTrainC.npy"))/5):int(len(np.load("PUCK_xTrainC.npy"))*2/5)])
#np.save("xTrainSyn2_3.npy",np.load("xTrainSyn2.npy")[int(len(np.load("PUCK_xTrainC.npy"))*2/5):int(len(np.load("PUCK_xTrainC.npy"))*3/5)])
#np.save("xTrainSyn2_4.npy",np.load("xTrainSyn2.npy")[int(len(np.load("PUCK_xTrainC.npy"))*3/5):int(len(np.load("PUCK_xTrainC.npy"))*4/5)])
#np.save("xTrainSyn2_5.npy",np.load("xTrainSyn2.npy")[int(len(np.load("PUCK_xTrainC.npy"))*4/5):int(len(np.load("PUCK_xTrainC.npy")))])	
'''
def templateAssessment(distanceFile, templateLabelFile, yLabelFile):
	distance = np.load(distanceFile)
	templateLabels = np.load(templateLabelFile)
	yLabel = np.load(yLabelFile)

	truePos = [0 for range(len(distance))]
	falsePos = [0 for range(len(distance))]

	for i in range(len(distance)):
		cat = np.argmin(distance[i])
		if yLabel[i] == templateLabels[cat]:
			truePos[cat] = truePas[cat]+1
		else:
			falsePos[cat] = falsePos[cat]+1

	ratio = [truePos[i]*1.0/(truePos[i]+falsePos[i]) for i in range(len(templateLabels))]
	print ratio
	return truePos, falsePos, ratio

'''
