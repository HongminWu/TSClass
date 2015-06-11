import numpy as np
import numpy.linalg as la
import sklearn.svm as svm
from sklearn.decomposition import PCA
import sklearn.tree as tree
import FeatureGeneration as fg
import Reader as r

feature_train = fg.featureExtraction("UCI_HAR_all_multivariateTS.npy", "X_train_UCI_features.npy")
feature_test = fg.featureExtraction("UCI_HAR_all_test_multTS.npy", "X_test_UCI_features.npy")
np.save("Xtrain_features.npy", feature_train)
np.save("Xtest_features.npy", feature_test)
feature_train = np.load("Xtrain_features.npy")
feature_test = np.load("Xtest_features.npy")

label_test = np.load("yTest.npy")
label_train = np.load("yTrain.npy")

s = []
for k in range(10):
	#pca = PCA(n_components=256)    
	#Xtrain = pca.fit(feature_train).transform(feature_train)

	#Xtest = pca.transform(feature_test)
	s.append(svm.LinearSVC().fit(feature_train, label_train).score(feature_test, label_test))
print np.mean(s)





