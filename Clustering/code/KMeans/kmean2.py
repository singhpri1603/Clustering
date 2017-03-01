import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import adjusted_rand_score
import math
import sys


#to generate random centroids
def randomCentroid():
	rands = np.random.choice(data.shape[0], k, replace=False) #random k indices from data
	#rands = np.sort(rands)
	centroids = np.copy(data[rands, :])	#copying data from those indices
	return centroids

def fixedCentroid():
	indexs = np.ones(shape = k, dtype=np.int)
	for i in range(k):
		indexs[i] = int(sys.argv[4+i])
	centroids = np.copy(data[indexs, :])
	return centroids

#to update centroids
def updateCentroid():
	for i in range(k):
		clusteri = np.argwhere(clusters == i)		#indices belonging to cluster i
		centroids[i] = np.mean(data[clusteri, :], axis = 0)	#mean of data belonging to cluster i

#clustering
def clusterk():
	for i in range(k):						
		dist = (data-centroids[i])**2
		result[i] = np.sqrt(dist.sum(axis=1))	#euclidean distance to each centroid i
	
	clusters = np.argmin(result, axis=0)	#index of centroid with min distance
	return clusters

#visualization
def visualize():
	pca = PCA(n_components = 2)
	pca.fit(data)
	b = pca.transform(data)
	pca.fit(centroids)
	c = pca.transform(centroids)
	for l in range(k):
		clusterl = np.argwhere(clusters == l)
		newplot = b[clusterl[:,0], :]
		plt.scatter(newplot[:,0],newplot[:,1], color=colors[l])
	plt.scatter(c[:,0],c[:,1], marker='+', color="black", s=100)
	plt.show()

#second visualization
def visualize2():
	plt.figure(1)
	for l in range(k):
		clusterl = np.argwhere(clusters == l)
		newplot = data[clusterl[:,0],:]
		plt.subplot(k,1,l)
		plt.plot(newplot, color=colors[l])
	plt.show()

#standardization
def standardize():
	m = np.mean(data, axis = 0)
	#print m
	s = np.sum(np.absolute(data-m), axis=0)
	#print s
	return (data-m)/s

####################### starts here ##########################

#initialize data/parameters

file = sys.argv[2]

data = np.genfromtxt(file+'.txt',dtype='f',delimiter='\t')


#data = standardize()
k = int(sys.argv[1])		#command line argument of k
l = int(sys.argv[3])
print 'Number of clusters: ', k
colors = cm.rainbow(np.linspace(0,1,k))
#centroids = randomCentroid()
centroids = fixedCentroid()
#oldCentroids = np.zeros(shape = centroids.shape)
result = np.zeros(shape = (centroids.shape[0], data.shape[0]))
oldClusters = np.zeros(shape = data.shape[0])
clusters = np.zeros(shape = data.shape[0])


#kmeans routine
iteri = 0
for var in range(l):
	clusters = clusterk()
	updateCentroid()
	if np.array_equal(oldClusters, clusters):
		break
	oldClusters = np.copy(clusters)
	iteri+=1
	#uncomment visualize() in next line to see plots at every iteration
	#visualize()

print clusters
print 'iterations: '+ str(iteri)
visualize()
visualize2()

ground = np.genfromtxt(file+'_gt.txt',dtype=np.int,delimiter='\t')
#print ground
#ars = adjusted_rand_score(ground, clusters)
#print ars

##################### jacc and rand##############
y_pred = clusters
y_true = ground
def get_matrix(y_pred):
    length_of_mat=len(y_pred)
    mat=np.zeros([length_of_mat,length_of_mat],dtype='int')
    #print truth_mat
    for i in np.unique(y_pred):
        #print i
        occurances=np.argwhere(y_pred==i)
     #   print occurances
        for k in range(0,len(occurances)):
            for m in range(k,len(occurances)):
                #print str(k)+" "+str(m)
                #print str(occurances[k][0])+" : "+str(occurances[m][0])
                mat[occurances[k][0]][occurances[m][0]]=1
                mat[occurances[m][0]][occurances[k][0]]=1
    return mat


### predicted and true labels
pred_mat=get_matrix(y_pred)
truth_mat=get_matrix(y_true)


## matrix count for m00 m11 m01
m00=0
m01=0
m11=0
print pred_mat.shape[0]
for i in range(0,pred_mat.shape[0]):
    for j in range(0,pred_mat.shape[0]):
        if(pred_mat[i][j]==truth_mat[i][j] and pred_mat[i][j]==1):
            m11=m11+1
        if(pred_mat[i][j]==truth_mat[i][j] and pred_mat[i][j]==0):
            m00=m00+1
        if(pred_mat[i][j]!=truth_mat[i][j]):
            m01=m01+1
            
jacard=1.0*m11/(m11+m01)
rand=1.0*(m11+m00)/(m11+m01+m00)

print jacard
print rand




