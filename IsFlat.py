import numpy as np

def IsFlat(ts, threshold = 1, method = 1):
	if method==1:
		for ACT in range(len(ts[0])):
			seq = [ts[i][ACT] for i in range(len(ts))]
			if sum(abs((seq-np.mean(seq))/(np.var(seq)**0.5))>threshold)>0:
				return False
		else:
			return True

			

