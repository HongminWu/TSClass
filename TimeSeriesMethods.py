import numpy as np
from numpy import linalg as la


#Calculates Dynamic time warping distance and path
#input time series s (list), time series t (list), bandwidth for matrix of calculations, norm type
#output distance, path  
def DTWDistance(s,t,bandwidth = 16, pnorm = 1):
    DTW = np.zeros((len(s), len(t)))
    path = [[(0,0) for i in range(len(t))] for j in range(len(s))]
    optpath = []
    abandon = min(len(s), len(t)) - bandwidth
    pnorm = pnorm
    optpath = []
    DTW.fill(np.inf)
    s = np.array(s)
    t = np.array(t)
    n = len(s)
    m = len(t)
    DTW[0][0] = la.norm(s[0]-t[0], pnorm)

    for j in range(1, m-abandon):
        DTW[0][j] = DTW[0][j-1] + la.norm(s[0]-t[j], pnorm)
        path[0][j] = (0, j-1)

    for i in range(1, n):
        for j in range(max(0, abandon-n+i), min(m, m-abandon+i)):
            DTW[i][j] = la.norm(s[i]-t[j], pnorm)
            if DTW[i][j-1] <= DTW[i-1][j] and DTW[i][j-1] <= DTW[i-1][j-1]:
                DTW[i][j] += DTW[i][j-1]
                path[i][j] = (i, j-1)
            elif DTW[i-1][j] <= DTW[i-1][j-1]:
                DTW[i][j] += DTW[i-1][j]
                path[i][j] = (i-1, j)
            else:
                DTW[i][j] += DTW[i-1][j-1]
                path[i][j] = (i-1, j-1)

        #print len(path[1])
        #print path[126][127]
    n -= 1
    m -= 1
    while n>0 or m >0:
        optpath.append((n,m))
        n = path[n][m][0]
        m = path[n][m][1]
    optpath.append((0,0))
    return DTW[len(DTW)-1][len(DTW[0])-1], optpath[::-1]

# A variation of subsequence DTW on time-series s,t (misaligned, periodic).
# s query, t reference
# For trunc (~ period), truncate s at the start and t at the end by {1,...trunc} units,
# and returns the best distance and the corresponding truncation.
def DTWsubseq(s, t, trunc = 25, bandwidth = 5, pnorm = 1):
    best_trunc = 0
    best_dist = np.inf
    for i in range(trunc):
        dtw, path = DTWDistance(s[i:], t[:128-i], bandwidth=bandwidth, pnorm=pnorm)
        if (dtw*128/(128-i))<best_dist:
            best_dist = (dtw*128/(128-i)) # good way to account for length diff?
            best_trunc = i
    return best_trunc, best_dist

#Calculates Dynamic time warping distance and path
#input time series s (list), time series t (list)
#output time series t aligned to s
def align(s, t, bandwidth=16, pnorm=1):
    r = deepcopy(t)
    DTW, path = DTWDistance(s, r, bandwidth=bandwidth, pnorm=pnorm)
    q = deepcopy(r)
    for j in path:
        q[j[0]]=r[j[1]]
    return q

#Calculate the average of a list of aligned time series
#input list of time series
#output average of the time series or 0 if the list is empty
def average(timeSeries):
    print "Entered newAverage"
    Y = np.array(timeSeries)
    print len(Y)
    if len(Y)>0:
        return sum(Y)/len(Y)
    else:
        return 0

