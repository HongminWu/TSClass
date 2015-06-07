import numpy as np

def Normalize(ts):
	tsTemp = np.array(ts)
	for ACT in range(len(tsTemp[0])):
		tsTemp[:,ACT] = (tsTemp[:,ACT]-np.mean(tsTemp[:,ACT]))/(np.var(tsTemp[:,ACT])**0.5)
	return tsTemp

def IsFlat(ts, threshold = 1, method = 1):
	if method==1:
		for ACT in range(len(ts[0])):
			if sum(abs(Normalize([ts[i][ACT] for i in range(len(ts))])))>threshold)>0:
				return False
		else:
			return True
	elif method==2:
		for ACT in range(len(ts[0])):
			if max(abs([ts[i][ACT] for i in range(len(ts))])>threshold):
				return False
		else:
			return True		

def AreFlat(ts, threshold = 1, method = 1):
	return [IsFlat(ts[i], threshold, method) for i in range(len(ts))]

def sampleLong(ts, nSample = 1000, T = 128):
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

