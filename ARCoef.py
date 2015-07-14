from sklearn import linear_model
import numpy as np

def ARCoef(x, bandwidth = 8):
	autoReg = linear_model.LinearRegression()
	X = []
	Y = []
	for i in range(bandwidth, len(x)):
		Y.append(x[i])
		temp = list([1])
		for j in range(i-bandwidth, i):
			temp.append(x[j])
		X.append(temp)
	autoReg.fit(X, Y)
	return list(autoReg.coef_)

# X = [1,1,2,3,5,8,13,21]
# print ARCoef(X, 2)
