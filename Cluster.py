import numpy as np
import TimeSeriesMethods as ts

#DBSCAN Clustering

def regionQuery(X,Xlabel,P,p,eps):
    print "regionQuery label", p
    region = []
    regionlabels = []
    for k in range(len(X)):
	if ts.DTWDistance(X[k], P)[0]<eps:
	    region.append(X[k])
	    regionlabels.append(Xlabel[k])
    return region,regionlabels


def expandCluster(P,p,sphere_points,sphere_points_labels,eps,MinPts, inCluster, visited):
    C = []
    C.append(P)
    inCluster[p] = 1
    visited[p] = 1
    print "expand cluster", p
    for pp in range(len(sphere_points)):
	if visited[sphere_points_labels[pp]]==0:
	    visited[sphere_points_labels[pp]] = 1
	    sphere_points_p, sphere_points_p_labels = regionQuery(sphere_points, sphere_points_labels,sphere_points[pp],sphere_points_labels[pp],eps)
	    if len(sphere_points_p) >= MinPts:
		    sphere_points = sphere_points+sphere_points_p
		    sphere_points_labels = sphere_points_labels + sphere_points_p_labels
    if inCluster[sphere_points_labels[pp]] ==0:
		C.append(sphere_points[pp])
		inCluster[pp]=1
    return C, visited, inCluster 

def DBSCAN(D,eps,MinPts, inCluster, visited):
    print len(D), len(D[0])
    Dlabels = [k for k in range(len(D))]
    mainCluster = []
    for p in range(len(D)):
	print "in DBSCAN point number", p
	if visited[Dlabels[p]]==0:
	    visited[Dlabels[p]] = 1
	    sphere_points,sphere_points_labels = regionQuery(D,Dlabels,D[p],Dlabels[p],eps)
	    if len(sphere_points)> MinPts:
			tempCluster, visited, inCluster = expandCluster(D[p], Dlabels[p],sphere_points, sphere_points_labels, eps, MinPts, inCluster, visited)
			mainCluster.append(tempCluster)
    return mainCluster


dist0 = np.load("dist_mats_act0.npy")
print len(dist0), len(dist0[0])


