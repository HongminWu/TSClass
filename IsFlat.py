import numpy as np
import Plot as plot

def Normalize(ts):
	tsTemp = np.array(ts)
	tsTemp = (tsTemp-np.mean(tsTemp))/(np.var(tsTemp)**0.5)
	return tsTemp

def IsFlat(ts, param = 1, method = 0):
	if method==0:
		for ACT in range(len(ts[0])):
			temp = max(ts[:, ACT])-min(ts[:, ACT])
			if (temp-param[ACT+ACT])*50>(param[ACT+ACT+1]-temp):
				return False
		else:
			return True
	elif method==1:
		for ACT in range(len(ts[0])):
			if sum(abs(Normalize([ts[i][ACT] for i in range(len(ts))]))>param)>0:
				return False
		else:
			return True
	elif method==2:
		for ACT in range(len(ts[0])):
			if max(abs([ts[i][ACT] for i in range(len(ts))])>param):
				return False
		else:
			return True		

def AreFlat(ts, param = 1, method = 0):
	print param
	return np.array([IsFlat(ts[i], param, method) for i in range(len(ts))])

def SampleLong(ts, nSample = 1000, T = 128):
	tsTemp = Normalize(ts)
	tsLength = length(tsTemp)
	samples = []
	k = 0
	COUNT = 0
	while (k<nSample) and (COUNT<nSample*10):
		#give up the first and last 10 points
		start = np.random.rindint(tsLength-20-T)
		subSeq = tsTemp[start:(start+T)]
		if not isFlat(subSeq):
			k += 1
			samples.append(subSeq)

	return np.array(samples)

def GetPrior(X):
	ans = []
	for ACT in range(len(X[0][0])):
		temp = []
		for i in range(len(X)):
			temp.append(max(X[i][:, ACT]) - min(X[i][:, ACT]))
		ans.append(min(temp))
		ans.append(max(temp))
	return ans

#### Working with the first dataset
print "Hellow World!"
xTrain = np.load("MotionData/Xtrain.npy")
yTrain = np.load("MotionData/Ytrain.npy")
areFlat = AreFlat(xTrain, GetPrior(xTrain))
np.save("MotionData/xTrainF.npy", [xTrain[i] for i in range(len(areFlat)) if areFlat[i]])
np.save("MotionData/yTrainF.npy", [yTrain[i] for i in range(len(areFlat)) if areFlat[i]])
np.save("MotionData/xTrainC.npy", [xTrain[i] for i in range(len(areFlat)) if not areFlat[i]])
np.save("MotionData/yTrainC.npy", [yTrain[i] for i in range(len(areFlat)) if not areFlat[i]])
# print yTrain[areFlat]
print "Train Finished", sum(areFlat)
np.save("MotionData/xTrainL.npy", areFlat)
print "Samples from: ", [sum([areFlat[i] for i in range(len(yTrain)) if yTrain[i]==ACT]) for ACT in range(6)]

xTest = np.load("MotionData/Xtest.npy")
yTest = np.load("MotionData/Ytest.npy")
areFlat = AreFlat(xTest, GetPrior(xTrain))
np.save("MotionData/xTestF.npy", [xTest[i] for i in range(len(areFlat)) if areFlat[i]])
np.save("MotionData/yTestF.npy", [yTest[i] for i in range(len(areFlat)) if areFlat[i]])
np.save("MotionData/xTestC.npy", [xTest[i] for i in range(len(areFlat)) if not areFlat[i]])
np.save("MotionData/yTestC.npy", [yTest[i] for i in range(len(areFlat)) if not areFlat[i]])
np.save("MotionData/xTestL.npy", areFlat)
print "Test Finished", sum(areFlat)

# plot.plot_raw("MotionData/xTrainF.npy", "MotionData/yTrainF.npy", "tt", 10)
plot.plot_raw("MotionData/xTrain.npy", "MotionData/yTrain.npy", "OR", 20)
plot.plot_raw("MotionData/xTrainF.npy", "MotionData/yTrainF.npy", "FL", 20)
plot.plot_raw("MotionData/xTrainC.npy", "MotionData/yTrainC.npy", "SP", 20)

