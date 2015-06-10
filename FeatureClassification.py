import numpy as np
import numpy.linalg as la
import sklearn.svm as svm
from sklearn.decomposition import PCA
import sklearn.tree as tree
import FeatureGeneration as fg
import Reader as r

# feature_train = fg.featureExtraction("xTrainC.npy")
# feature_test = fg.featureExtraction("xTestC.npy")
# np.save("XtrainC_features.npy", feature_train)
# np.save("XtestC_features.npy", feature_test)
s = []
for k in range(100):
	feature_train = np.load("XtrainC_features.npy")
	feature_test = np.load("XtestC_features.npy")

	label_test = np.load("yTestC.npy")
	label_train = np.load("yTrainC.npy")

	pca = PCA(n_components=256)    
	Xtrain = pca.fit(feature_train).transform(feature_train)

	Xtest = pca.transform(feature_test)

	s.append(svm.LinearSVC().fit(feature_train, label_train).score(feature_test, label_test))
print np.mean(s)





