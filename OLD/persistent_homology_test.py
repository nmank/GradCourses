import sys
sys.path.append('./pythonScripts')
sys.path.append('./pythonScripts/multipripser-master')
import calcom
import numpy as np
import graph_tools_construction as gt
import ripser_interface as ri
import matplotlib.pyplot as plt


#noisy circle
# n = 101
# th = np.linspace(0, 2*np.pi, n)
# x = np.cos(th) + 0.1*np.random.randn(n)
# y = np.sin(th) + 0.1*np.random.randn(n)
# cloud = np.vstack( [x,y] ).T


#circle
n = 101
th = np.linspace(0, 2*np.pi, n)
x = np.cos(th) 
y = np.sin(th) 
cloud = np.vstack( [x,y] ).T

#noise
# n = 101
# x = 0.1*np.random.randn(n)
# y = 0.1*np.random.randn(n)
# cloud = np.vstack( [x,y] ).T


plt.scatter(cloud[:,0],cloud[:,1])
plt.show()


X = cloud.T


A,nds = gt.adjacency_matrix(X,'correlation')

D = gt.sim2dist(A)




result = ri.run_ripser_sim(D)
fig,ax = ri.plot_PH_summary(result)
