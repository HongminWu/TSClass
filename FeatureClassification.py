import numpy as np
import numpy.linalg as la
import sklearn.svm as svm
from sklearn.decomposition import PCA
import sklearn.tree as tree
import FeatureGeneration as fg
import Reader as r

#feature_test = fg.featureExtraction("UCI_HAR_all_test_multTS.npy")
#feature_train = fg.featureExtraction("UCI_HAR_all_multivariateTS.npy")

label_test = np.load("UCI_HAR_test_labels.npy")
label_train = np.load("UCI_HAR_train_labels.npy")
feature_train = np.load("Xtrain.npy")
feature_test = np.load("Xtest.npy")
k = 150
Xtrain = np.array(feature_train)
pca = PCA(n_components=k)    
Xtrain1 = pca.fit(Xtrain).transform(Xtrain)

Xtest = np.array(feature_test)
Xtest1 = pca.transform(Xtest)
print svm.LinearSVC().fit(feature_train, label_train).score(feature_test, label_test)


#print svm.LinearSVC().fit(Xtrain1, label_train).score(Xtest1, label_test)




