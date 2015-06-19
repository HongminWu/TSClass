import numpy as np
import numpy.linalg as la
import sklearn.svm as svm
from sklearn.decomposition import PCA
import sklearn.tree as tree
import FeatureGeneration as fg
import Reader as r

#feature_train = fg.featureExtraction("UCI_HAR_all_multivariateTS.npy", "X_train_UCI_features.npy")
#feature_test = fg.featureExtraction("UCI_HAR_all_test_multTS.npy", "X_test_UCI_features.npy")
#np.save("Xtrain_features.npy", feature_train)
#np.save("Xtest_features.npy", feature_test)
feature_train = np.load("Xtrain_features.npy")
feature_test = np.load("Xtest_features.npy")

label_test = np.load("UCI_HAR_test_labels.npy")
label_train = np.load("UCI_HAR_train_labels.npy")
j = 350
k = 0
s = []
while j<400:
	w = []
	k=0
	while k<10:
		pca = PCA(n_components=j)    
		Xtrain = pca.fit(feature_train).transform(feature_train)

		Xtest = pca.transform(feature_test)
		w.append(svm.LinearSVC().fit(Xtrain, label_train).score(Xtest, label_test))
		k+=1
	s.append((max(w),j))
	j+=2
print s









