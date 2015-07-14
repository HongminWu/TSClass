import numpy as np
import numpy.linalg as la
import sklearn.svm as svm
from sklearn.decomposition import PCA
import sklearn.tree as tree
import FeatureGeneration as fg
import Reader as r
from sklearn.metrics import confusion_matrix

#feature_train = fg.featureExtraction("xTrainSyn2.npy")
#feature_test = fg.featureExtraction("xTestSyn2.npy")
#np.save("Xtrain_features_Syn2.npy", feature_train)
#np.save("Xtest_features_Syn2.npy", feature_test)
feature_train = np.load("Xtrain_features.npy")
feature_test = np.load("Xtest_features.npy")
print len(feature_train[0])
label_test = np.load("yTestC.npy")
label_train = np.load("yTrainC.npy")
'''
for k in range(len(label_train)):
	if label_train[k]>2:
		label_train[k]=3
for l in range(len(label_test)):
	if label_test[l]>2:
		label_test[l]=3
'''
p = []
q = []
Xtrain=feature_train
Xtest=feature_test
#j = range(1,len(feature_train[0]))
k = 0
#j=350
#for j in range(115, len(feature_train[0])):
#for j in range(301,len(Xtrain[0])):
pca = PCA(n_components=252)    
Xtrain = pca.fit(feature_train).transform(feature_train)
ss = svm.LinearSVC(random_state=420)
Xtest = pca.transform(feature_test)
y_pred =  ss.fit(Xtrain,label_train).predict(Xtest)
cm = confusion_matrix(label_test,y_pred)
q.append(cm)
#print j, ss.fit(Xtrain, label_train).score(Xtest, label_test)
p.append(ss.fit(Xtrain, label_train).score(Xtest, label_test))
print np.argmax(p),max(p), q[np.argmax(p)]








