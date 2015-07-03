import numpy as np
import numpy.linalg as la
import sklearn.svm as svm
from sklearn.decomposition import PCA
import sklearn.tree as tree
import FeatureGeneration as fg
import Reader as r
from sklearn.metrics import confusion_matrix

#feature_train = fg.featureExtraction("xTrainSyn.npy")
#feature_test = fg.featureExtraction("xTestSyn.npy")
#np.save("Xtrain_features_syn.npy", feature_train)
#np.save("Xtest_features_syn.npy", feature_test)
feature_train = np.load("Xtrain_features_syn.npy")
feature_test = np.load("Xtest_features_syn.npy")

label_test = np.load("yTestSyn.npy")
label_train = np.load("yTrainSyn.npy")
'''
for k in range(len(label_train)):
	if label_train[k]>2:
		label_train[k]=3
for l in range(len(label_test)):
	if label_test[l]>2:
		label_test[l]=3
'''

Xtrain=feature_train
Xtest=feature_test
#j = range(1,len(feature_train[0]))
k = 0
#pca = PCA(n_components=j)    
#Xtrain = pca.fit(feature_train).transform(feature_train)

#Xtest = pca.transform(feature_test)
y_pred =  svm.LinearSVC(random_state=420).fit(Xtrain,label_train).predict(Xtest)
cm = confusion_matrix(label_test,y_pred)
print cm
print svm.LinearSVC(random_state=420).fit(Xtrain, label_train).score(Xtest, label_test)









