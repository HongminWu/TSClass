import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from TimeSeriesMethods import DTWDistance
import Plot


'''

A variation of subsequence DTW on time-series s,t (misaligned, periodic).
s query, t reference
For trunc (~ period), truncate s at the start and t at the end by {1,...trunc} units,
and returns the best distance and the corresponding truncation.

ref: http://www.jstatsoft.org/v31/i07/paper

'''

def DTWsubseq(s, t, trunc = 25, bandwidth = 5, pnorm = 1):
    best_trunc = 0
    best_dist = np.inf
    for i in range(trunc):
        dtw, path = DTWDistance(s[i:], t[:128-i], bandwidth=bandwidth, pnorm=pnorm)
        if (dtw*128/(128-i))<best_dist:
            best_dist = (dtw*128/(128-i)) # good way to account for length diff?
            best_trunc = i
    return best_trunc, best_dist


if __name__ == "__main__":

    Xtrain = np.load('xTrainC.npy')
    Ytrain = np.load('yTrainC.npy')

    dist_mats = []
    trunc_mats = []

    for act in range(6):
        Xtrain_act = Xtrain[Ytrain==act]
        Ytrain_act = Ytrain[Ytrain==act]
        n = Xtrain_act.shape[0]

        dist_mat = np.zeros((n,n))
        trunc_mat = np.zeros((n,n))

        for i in range(n):
            for j in range(n):
                if j>i:
                    print 'act'+str(act)+'row'+str(i)+' column'+str(j)
                    best_trunc, best_dist = DTWsubseq(Xtrain_act[j],Xtrain_act[i])
                    trunc_mat[i,j] = best_trunc
                    dist_mat[i,j] = best_dist

        for i in range(n):
            for j in range(n):
                if j<i:
                    dist_mat[i,j] = dist_mat[j,i]

    dist_mats.append(dist_mat)
    trunc_mats.append(trunc_mat)

    np.save('dist_mats.npy', dist_mats)
    np.save('trunc_mats.npy', trunc_mats)

'''
    # Hierarchical clustering

    Xtrain = np.load('xTrainC.npy')
    Ytrain = np.load('yTrainC.npy')

    dist_mat = np.load('dist_mat.npy')
    trunc_mat = np.load('trunc_mat.npy')

    cluster = [] # cluster for all activities
    ycluster = [] # cluster label
    for act in range(6):
        Xtrain_act = Xtrain[Ytrain==act]
        data_link = linkage(dist_mat[act], method='complete')
        ind = fcluster(data_link, 0.5*data_link.max(), criterion='distance')
        num_cluster = max(ind) # number of clusters for activity
        for c in range(1,num_cluster+1):
            cluster.append(Xtrain_act[ind==c])
            ycluster.append(act)

    num_c = ycluster.__len__()
    for c in range(num_c):
        Plot.plot_raw(cluster[c], np.array(ycluster[c]*cluster[c].shape[0]),
                      'plot/act'+str(ycluster[c])+'_hcluster'+str(c), 5, act=6, file=False)

'''