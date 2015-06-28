import numpy as np
from TimeSeriesMethods import DTWDistance, align, average
import Plot

'''

Make use of DTWsubseq to find the best horizontal shifts, then align and take average to find activity templates.
Due to truncating, the output average is of shorter length than the original series.

'''

if __name__ == "__main__":

    import dba

    # dba templates

    cluster10 = np.load('syn_cluster10.npy')
    ycluster10 = np.load('syn_ycluster10.npy')
    trunc_cluster10 = np.load('syn_trunc_cluster10.npy')

    # horizontal shifting

    cluster10_trunc = []
    l = len(cluster10)
    for c in range(l):
        cluster = []
        n = len(cluster10[c])
        for i in range(n):
            trunc = trunc_cluster10[c][0,i]
            cluster.append(cluster10[c][i][trunc:(103+trunc)]) # length 128-25, minimum length
        cluster = np.array(cluster)
        cluster10_trunc.append(cluster)
    cluster10_trunc = np.array(cluster10_trunc)

    n_cluster10 = len(cluster10_trunc)
    dba_avgs = []
    for c in range(n_cluster10):
        dbamodel = dba.DBA(max_iter=30, verbose=True, tol=1e-4)
        dba_avg = dbamodel.compute_average(cluster10_trunc[c], nstarts=5)
        dba_avgs.append(dba_avg)

    np.save('syn_dba_templates_trunc.npy', dba_avgs)
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