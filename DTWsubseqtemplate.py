import numpy as np
from TimeSeriesMethods import DTWDistance, align, average
import Plot

'''

Make use of DTWsubseq to find the best horizontal shifts, then align and take average to find activity templates.
Due to truncating, the output average is of shorter length than the original series.

'''

def template(cluster_mat, trunc_mat, bandwidth = 5, pnorm = 1):
    n = len(cluster_mat)
    align_cluster = np.array([cluster_mat[0]]*n)
    for i in range(n):
        trunc = trunc_mat[0,i]
        align_cluster[i][:128-trunc] = align(cluster_mat[0][:128-trunc], cluster_mat[i][trunc:], bandwidth=bandwidth, pnorm=pnorm) #align ts2 to ts1
    avg = average(align_cluster)
    return align_cluster, avg

if __name__ == "__main__":

    import os
    import dba
    '''
    os.chdir('/home/wenyu/Dropbox/TSCLASS')

    cluster = np.load('Distances/cluster.npy')
    ycluster = np.load('Distances/ycluster.npy')
    trunc_cluster = np.load('Distances/trunc_cluster.npy')

    n_cluster = len(cluster)
    cluster10 = []
    ycluster10 = []
    trunc_cluster10 = []

    for c in range(n_cluster): # retain clusters with at least 10 samples
        if len(cluster[c])>=10:
            print 'cluster'+str(c)
            cluster10.append(cluster[c])
            ycluster10.append(ycluster[c])
            trunc_cluster10.append(trunc_cluster[c])

    np.save('Distances/cluster10.npy', cluster10)
    np.save('Distances/ycluster10.npy', ycluster10)
    np.save('Distances/trunc_cluster10.npy', trunc_cluster10)
    '''
    # dba templates

    cluster10 = np.load('cluster10.npy')
    ycluster10 = np.load('ycluster10.npy')
    trunc_cluster10 = np.load('trunc_cluster10.npy')

    n_cluster10 = len(cluster10)
    dba_avgs = []
    for c in range(n_cluster10):
        dbamodel = dba.DBA(max_iter=30, verbose=True, tol=1e-4)
        dba_avg = dbamodel.compute_average(cluster10[c], nstarts=5)
        dba_avgs.append(dba_avg)

    np.save('dba_templates.npy', dba_avgs)
    #Plot.plot_template_many(dba_avgs, cluster10, ycluster10, 'plot/dba_templates', file=False)

    # horizontal alignment using trunc_cluster10



'''
    align_clusters = []
    avgs = []

    for c in range(n_cluster):
        if len(cluster[c])>=10:
        print 'cluster'+str(c)
            cluster10.append(cluster[c])
            ycluster10.append(ycluster[c])
            align_cluster, avg = template(cluster[c], trunc_cluster[c])
            align_clusters.append(align_cluster)
            avgs.append(avg)

    np.save('Templates/align_clusters.npy', align_clusters)
    np.save('Templates/templates', avgs)

    Plot.plot_template_many(avgs, align_clusters, ycluster, 'plot/template', file=False)

'''