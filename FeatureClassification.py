import numpy as np
import numpy.linalg as la
import sklearn.svm as svm
from sklearn.decomposition import PCA
import sklearn.tree as tree
import FeatureGeneration as fg
import Reader as r
from sklearn.metrics import confusion_matrix

feature_train = fg.featureExtraction("PUCK_xTrainC.npy", "X_train_UCI_features.npy")
feature_test = fg.featureExtraction("PUCK_xTestC.npy", "X_test_UCI_features.npy")
np.save("Xtrain_features_puck.npy", feature_train)
np.save("Xtest_features_puck.npy", feature_test)
feature_train = np.load("Xtrain_features_puck.npy")
feature_test = np.load("Xtest_features_puck.npy")

label_test = np.load("PUCK_yTestC.npy")
label_train = np.load("PUCK_yTrainC.npy")
'''
for k in range(len(label_train)):
	if label_train[k]>2:
		label_train[k]=3
for l in range(len(label_test)):
	if label_test[l]>2:
		label_test[l]=3
'''

j = 350
k = 0
pca = PCA(n_components=j)    
Xtrain = pca.fit(feature_train).transform(feature_train)

Xtest = pca.transform(feature_test)
y_pred =  svm.LinearSVC().fit(Xtrain,label_train).predict(Xtest)
cm = confusion_matrix(label_test,y_pred)
print cm
print svm.LinearSVC().fit(Xtrain, label_train).score(Xtest, label_test)









