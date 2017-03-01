import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import adjusted_rand_score
import sys

k = int(sys.argv[1])
file1 = sys.argv[2]
colors = cm.rainbow(np.linspace(0,1,k))
data = np.genfromtxt(file1+'.txt',dtype='f',delimiter='\t')
clusters = np.genfromtxt('clusters.txt', dtype=np.int, delimiter='\n')
print clusters

def visualize():
	pca = PCA(n_components = 2)
	pca.fit(data)
	b = pca.transform(data)
	#pca.fit(centroids)
	#c = pca.transform(centroids)
	for l in range(k):
		clusterl = np.argwhere(clusters == l)
		newplot = b[clusterl[:,0], :]
		plt.scatter(newplot[:,0],newplot[:,1], color=colors[l])
	#plt.scatter(c[:,0],c[:,1], marker='+', color="black", s=100)
	plt.show()

visualize()
gt = np.genfromtxt(file1+'_gt.txt',dtype=np.int,delimiter='\t')


#################### jacc and rand #################################
y_pred = clusters
y_true = gt
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
