import numpy as np
from numpy import linalg as la
import scipy as sp
import Reader as r

#6 features
def getMean(x):
	means = []
	for k in x:
		means.append(np.mean(k))
	return means

#6 features
def getSD(x):
	sds = []
	for k in x:
		sds.append(np.std(k))
	return sds

def getFFT(x):
	freqx = []
	for k in x:
		freqx.append(np.fft.fft(k))
	for f in range(len(freqx)):
		for k in range(len(freqx[f])):
			freqx[f][k]=np.real_if_close(np.abs(freqx[f][k]),tol=1000)
	return freqx

#6 features
def energy(x):
	energies = []
	freqs = getFFT(x)
	for f in freqs:
		temp = 0
		for i in range(len(f)):
			temp+=(f[i]**2)/float(len(f))
		energies.append(temp)
	return energies

#15 features
def correlation(x):
	correlations = []
	for i in range(len(x)):
		q = [float(w) for w in x[i]]
		for j in range(i,len(x)):
			r = [float(m) for m in x[j]]
			correlations.append(float(np.correlate(q,r)))
	return correlations

#calculates autocorrelation
def autocorr(x):
	autocorrs = []
	for k in x:
		temp = []
		q = [float(w) for w in k]
		corr = np.correlate(q, q, mode='full')
		corr = corr[corr.size/2:]
		autocorrs.append(corr.item(0))
		for w in range(1,corr.size):
			if w % 5 ==0:
				autocorrs.append(corr.item(w))
	return autocorrs

#calculate absolute value of difference to mean
def averageAbs(x):
	means = getMean(x)
	avgabs = []
	for k in range(len(x)):
		newts = []
		for i in range(len(x[k])):
			newts.append(np.abs(x[k][i]-means[k]))
		avgabs.append(newts)
	return getMean(avgabs)

#calculates net acceleration sum of sqrt of accelerations
def averageRes(x):
	avgresacc = []
	avgresgyro = []
	for i in range(len(x[0])):
		avgresacc.append(np.sqrt(x[0][i]**2+x[1][i]**2+x[2][i]**2))
		avgresgyro.append(np.sqrt(x[3][i]**2+x[4][i]**2+x[5][i]**2))
	return [np.mean(avgresacc), np.mean(avgresgyro)]


def numZeros(x):
	zerocrosses = [0]*6
	for k in range(len(x)):
		for i in range(1,len(x[k])-1):
			#if (np.abs(x[k][i])<10**(-7) and np.abs(x[k][i-1])<0 and np.abs(x[k][i+1])>0) or (np.abs(x[k][i])<10**(-7) and np.abs(x[k][i-1])>0 and np.abs(x[k][i+1])<0):
			if np.abs(x[k][i]<10**(-15)):
				zerocrosses[k]+=1
	return zerocrosses

def getKurtosis(x):
	kurtosis = []
	for k in x:
		kurtosis.append(sp.stats.kurtosis(k))
	return kurtosis

def getSkew(x):
	skew = []
	for k in x:
		skew.append(sp.stats.skew(k))
	return skew	

#calculates root mean sq 
def rms(x):
	rmss = []
	for k in range(len(x)):
		for i in range(len(x[k])):
			x[k][i] = x[k][i]**2
	for k in x:
		rmss.append(np.sqrt(np.mean(k)))
	return rmss



def getFeatures(x):
	featureVec = []
	featureFunctions = [getMean, getSD, energy, correlation, autocorr, averageAbs, averageRes, numZeros, getKurtosis, getSkew, rms]
	#featureFunctions = [averageRes]
	for f in range(len(featureFunctions)):
		w = featureFunctions[f](x)
		for i in w:
			featureVec.append(i)
	return featureVec

def transformTS(x):
	return map(list, zip(*x))

def firstDiffTS(x):
	for k in range(len(x)):
		for i in range(len(x[k])):
			if i==0:
				x[k][i]=x[k][i+1]-x[k][i]
			elif i==len(x[k])-1:
				x[k][i]=x[k][i]-x[k][i-1]
			else:
				x[k][i] = (x[k][i+1] - x[k][i-1])/2.0
	return x



def featureExtraction(file1):
	featuresTS = []
	multTSAll= np.load(file1)
	m = multTSAll[0]
	for TS in range(len(m)):
		features=np.concatenate((np.array(getFeatures(transformTS(multTSAll[TS]))), np.array(getFeatures(firstDiffTS(transformTS(multTSAll[TS])))),np.array(getFeatures(getFFT(transformTS(multTSAll[TS]))))))
		featuresTS.append(features)	
	return featuresTS 



