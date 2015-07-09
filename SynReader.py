import numpy as np

data = np.load("Synthetic2/Mislabeled_hclust0.25_mindist_average_origdist.npy")

X = []
Y = []

for ts, pred, actl in data:
	# print ts, pred, actl
	print ts
	print ts[0]

	X.append(ts)
	Y.append(actl)

for i in range(6):
	print i, ": ", sum([1 for j in range(len(Y)) if Y[j]==i])

np.save("Synthetic2/xSyn2", X)
np.save("Synthetic2/ySyn2", Y)