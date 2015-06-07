import numpy as np
import numpy.linalg as la
import sklearn.svm as svm
import sklearn.tree as tree
import FeatureGeneration as fg
import Reader as r

feature_train = fg.featureExtraction("UCI_HAR_all_multivariateTS.npy")
feature_test = fg.featureExtraction("UCI_HAR_all_test_multTS.npy")

label_test = np.load("UCI_HAR_test_labels.npy")
label_train = np.load("UCI_HAR_train_labels.npy")

# print len(feature_train), len(feature_train[0])
# print feature_train[0]

print svm.LinearSVC().fit(feature_train, label_train).score(feature_test, label_test)




