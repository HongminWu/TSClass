mport numpy as np
from numpy import linalg as la


#smoothing class
class Smoother:
        #initialize with quartic kernel
        def __init__(self, kernel = "Quartic"):
                if kernel=="Quartic":
                        self.kernel = lambda x: 15.0/16*(1-x**2)**2*(-1<x and x<1)


        #smooth function smooths Y with bandwidth h
        def smooth(self, Y, h, nSample = 0):
                nRange = len(Y[0])
                if nSample==0: 
                        nSample = len(Y[0])

                x = [nRange * 1.0 / nSample * i for i in range(nSample)]
                weight = []
                sumWeight = []
                for i in range(nSample):
                        tempWeight = []
                        for j in range(nRange):
                                tempWeight.append(self.kernel((x[i]-j)/h))
                        weight.append(tempWeight)
                        sumWeight.append(sum(tempWeight))

                ans = []
                for y in Y:
                        tempY = []
                        for i in range(nSample):
                                tempY.append(sum([y[j]*weight[i][j] for j in range(nRange)])/sumWeight[i])
                        ans.append(tempY)
                return np.array(ans)
