import numpy as np
from TimeSeriesMethods import DTWDistance

'''

A variation of subsequence DTW on time-series s,t (misaligned, periodic).
s query, t reference
For trunc (~ period/2), truncate s at the start and t at the end by {1,...trunc} units,
and returns the best distance and the corresponding truncation.

ref: http://www.jstatsoft.org/v31/i07/paper

'''

def DTWsubseq(s, t, trunc = 10, bandwidth = 10, pnorm = 1):
    best_trunc = 0
    best_dist = np.inf
    for i in range(trunc):
        dtw, path = DTWDistance(s[i:], t[:128-i], bandwidth=bandwidth, pnorm=pnorm)
        if (dtw*128/(128-i))<best_dist:
            best_dist = (dtw*128/(128-i)) # good way to account for length diff?
            best_trunc = i
    return best_trunc, best_dist


if __name__ == "__main__":

    Xtrain = np.load('../6780Project/Xtrain.npy')
    Ytrain = np.load('../6780Project/Ytrain.npy')

    # activity 0
    Xtrain0 = Xtrain[Ytrain==0]
    n = Xtrain0.shape[0]

    dist_mat = np.zeros((n,n))
    trunc_mat = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            if j>i:
                print 'row'+str(i)+' column'+str(j)
                best_trunc, best_dist = DTWsubseq(Xtrain0[j],Xtrain0[i])
                trunc_mat[i,j] = best_trunc
                dist_mat[i,j] = best_dist

    np.save('dist_mat.npy', dist_mat)
    np.save('trunc_mat.npy', trunc_mat)







