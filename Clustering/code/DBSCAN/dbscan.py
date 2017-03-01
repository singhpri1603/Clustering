import pandas as pd
from scipy.spatial import distance
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from collections import deque


def dbscan(data, eps, minPts, unvisited, noise, cluster, col):
    print "#"
    c=0
    for index, row in data.iterrows():
        if index in unvisited:
            #print index
            P=row[2:col]
            #print P
            unvisited.discard(index)
            NeighborPts, N = regionQuery(P, data, eps, col)
            #print NeighborPts
            #print "here"
            if len(NeighborPts)< minPts:
                noise.add(index)
                cluster[index]=-1
                
            else:
                c=c+1
                cluster = expandCluster(P, NeighborPts, N, c, eps, minPts, index, unvisited, data, col)
            #break
    return noise, cluster, c
            


def expandCluster(P, neighbors, N, c, eps, minPts, index, unvisited, data, col):
    print "%"
    print c
    cluster[index]=c
    while(len(neighbors)>0):
        #print "start of loop"
        #print len(neighbors)
        PCurr=neighbors.popleft()
        if PCurr in unvisited:
            unvisited.discard(PCurr)
            P=data.ix[PCurr,2:col]
            neighborsCurr, NCurr=regionQuery(P, data, eps, col)
            if len(neighborsCurr)>=minPts:
                while(len(neighborsCurr)>0):
                    temp=neighborsCurr.popleft()
                    if temp not in N:
                        neighbors.append(temp)
                        #print "appending"
                        #print len(neighbors)
                N.union(NCurr)
                
                #neighbors.union(neighborsCurr)
            if PCurr not in cluster:
                cluster[PCurr]=c
    return cluster


def regionQuery(P, data, eps, col):
    #print "$"
    neighbors= deque()
    N=set()
    for index, row in data.iterrows():
        B=row[2:col]
        dis=distance.euclidean(P,B)
        if dis<=eps:
            neighbors.append(index)
            N.add(index)

    #print len(neighbors)
    return neighbors, N

    #print neighbors
        

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


data = pd.read_csv('iyer.txt', sep = "\t", header = None)
#print data
#print data.shape
rows=data.shape[0]
col=data.shape[1]
#print col

eps=1.25
minPts=2

unvisited=set()

for index, row in data.iterrows():
    unvisited.add(index)

noise=set()
cluster=np.zeros(shape=len(data))

#P=data.ix[0,2:18]
#print P
#regionQuery(P, data, eps)

noise, cluster, c = dbscan(data, eps, minPts, unvisited, noise, cluster,col)

print "outliers"
print len(noise)

#print cluster

ground=[0] * rows
for i in range(rows):
    ground[i]=data.get_value(i,1,False)

pred=[0] * rows
for j in range(rows):
    #print ""
    pred[j]=cluster[j]

### predicted and true labels
pred_mat=get_matrix(pred)
truth_mat=get_matrix(ground)

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


data = data.ix[:,2:col]

pca = PCA(n_components=2)
pca.fit(data)
a = pca.transform(data)
#colors = cm.rainbow(np.linspace(0,1,c+1))
colors=["snow","darkred","deeppink","lime","blue","darkgreen","teal","orange","olive","r","orchid","royalblue","aqua","yellow","silver","tan","lightcoral"]

plt.figure(1)
for l in range(1,c+1):
    clusterl = np.argwhere(cluster == l)
    #print clusterl
    #print "done"
    newplot = a[clusterl[:,0], :]
    plt.scatter(newplot[:,0],newplot[:,1], color=colors[l])
    #plt.plot(newplot, color=colors[l])

clusterl = np.argwhere(cluster == -1)
newplot = a[clusterl[:,0], :]
plt.scatter(newplot[:,0],newplot[:,1], color="Black")
    
plt.show()



