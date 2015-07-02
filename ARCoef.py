from sklearn import linear_model as lm
import numpy as np

def ARCoef(x, bandwidth = 8):
	autoReg = lm.LinearRegression()
	X = []
	Y = []
	for i in range(bandwidth, len(x)):
		Y.append(x[i])
		X.append([x[j] for j in range(i-bandwidth, i)])
	autoReg.fit(X, Y)
	return list(autoReg.coef_)



