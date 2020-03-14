import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

#============ initial settings and initial data load =========
# x = np.loadtxt('/home/mgharasu/A.txt')
def km_main(x,k):
    num_data = len(x)
    colors= ['blud','green', 'cyan', 'magnet', 'black', 'yellow','white', ]

    #================= plot data and centroids ====================
    # data = f.read().split(' ')
    # plt.style.use('ggplot')
    # plt.scatter(x[:,0],x[:,1], c=(0,0,1), s=7)
    # plt.title('original data')
    SSE = []
    for _ in range(2,10):
        
        # choosing three central points = k
        centroid = x[np.random.choice(num_data,k, replace = 0)]#[50,120]#

        colors=[np.random.rand(3) for i in range(k)]
        old_centroid = np.zeros(np.shape(centroid))
        num_same_centroids = 0
        distance_matrix = np.zeros((k, num_data))
        class_assigned = None
        while True:
            if  (old_centroid == centroid).all():
                if num_same_centroids>2:
                    sse = 0.0
                    for i in range(k):
                        tmp = x[np.where(class_assigned==i)]
                        # plt.scatter(tmp[:,0], tmp[:,1], color = colors[i], label = 'cluster'+str(i) , s=7)
                        for j in range(len(tmp)):
                            sse = sse + np.sqrt(np.sum(np.power(centroid[i]-tmp[j],2)))

                    SSE.append(sse)
                    # plt.scatter(centroid[:,0], centroid[:,1], color = ['red'], marker = 'D')
                    # plt.title('output for k='+str(k)+',  SSE = '+str(sse))
                    # plt.show()               
                    break
                else:
                    num_same_centroids += 1

            
            old_centroid = centroid[:]
            # Euclidean distance for each point with each centroid        
            for i in range(k):
                for j in range(num_data):
                    distance_matrix[i,j] = np.sqrt(np.sum(np.power(centroid[i]-x[j],2)))            

            class_assigned = np.argmin(distance_matrix, axis=0)

            new_centroid = np.zeros((k,2))

            for i in range(k):
                c = (np.mean(x[np.where(class_assigned==i),:], axis=1))[0]
                if not np.isnan(c):
                    # centroid[i] =old_centroid[i]
                    centroid[i] = c[0]



    # plt.show()
    # x=range(2,11)
    # plt.title('SSE agains k')
    # plt.plot(x, SSE)
    # plt.show()
    # plt.pause(0.02)
    return centroid


