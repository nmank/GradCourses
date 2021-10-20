import calcom
import modules
import numpy as np
import sys
sys.path.append('../pythonScripts')
import graph_tools_construction as gt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

epsilon = .997
distance = 'correlation'
h_k_p = 2
t = 60

ccd = calcom.io.CCDataSet('../data/ccd_gse73072_geneid.h5')

#q0 = {'time_id': np.arange(-50,0)}
q1 = {'time_id': t, 'shedding':True}

#idx0 = ccd.find(q0)
idx1 = ccd.find(q1)
#num_shd = len(idx1)
#idx = np.hstack([ idx0, idx1 ])
#labels = np.hstack([ np.repeat('control', len(idx0)), np.repeat('shedder', len(idx1)) ])
idx = idx1
modules.pathway_info.append_pathways_to_ccd(ccd)

data = ccd.generate_data_matrix(idx=idx, features = "REACTOME_INTERFERON_ALPHA_BETA_SIGNALING")
Data = np.log(data)


A, node_list = gt.adjacency_matrix(Data,distance,epsilon, h_k_param =h_k_p)
print(A)
n,m = data.shape

n1,n2 = gt.laplace_partition(A,fiedler = False,)
print(n1)

A1 = np.zeros((len(n1[:,0]),len(n1[:,0])))
for i in range(len(n1[:,0])):
	for j in range(len(n1[:,0])):
		A1[i,j] = A[n1[i,0],n1[j,0]] 

A2 = np.zeros((len(n2[:,0]),len(n2[:,0])))
for i in range(len(n2[:,0])):
	for j in range(len(n2[:,0])):
		A2[i,j] = A[n2[i,0],n2[j,0]] 

gt.displaygraph(A,20*np.ones(n),labels = False,layout = 'spring', plt_name = 'whole_t60_shed.png')

gt.displaygraph(A1,20*np.ones(len(n1[:,0])),labels = False,layout = 'spring', plt_name = 'half1_t60_shed.png')

gt.displaygraph(A2,20*np.ones(len(n2[:,0])),labels = False,layout = 'spring', plt_name = 'half2_t60_shed.png')




ccd = calcom.io.CCDataSet('../data/ccd_gse73072_geneid.h5')

q0 = {'time_id': np.arange(-50,0)}
q1 = {'time_id': t, 'shedding':True}

idx0 = ccd.find(q0)
idx1 = ccd.find(q1)
num_shd = len(idx1)
idx0 = idx0[0:num_shd]
idx = np.hstack([ idx0, idx1 ])
#labels = np.hstack([ np.repeat('control', len(idx0)), np.repeat('shedder', len(idx1)) ])
#idx = idx1
modules.pathway_info.append_pathways_to_ccd(ccd)

data = ccd.generate_data_matrix(idx=idx, features = "REACTOME_INTERFERON_ALPHA_BETA_SIGNALING")
Data = np.log(data)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(Data[:num_shd,n1[0,0]],Data[:num_shd,n1[1,0]],Data[:num_shd,n1[2,0]])
ax.scatter(Data[num_shd:,n1[0,0]],Data[num_shd:,n1[1,0]],Data[num_shd:,n1[2,0]])
fig.savefig('3d_n1.png') 
#ax.show()

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(Data[:num_shd,n2[0,0]],Data[:num_shd,n2[1,0]],Data[:num_shd,n2[2,0]])
ax.scatter(Data[num_shd:,n2[0,0]],Data[num_shd:,n2[1,0]],Data[num_shd:,n2[2,0]])
fig.savefig('3d_n2.png') 
#ax.show()



A, node_list = gt.adjacency_matrix(Data,distance,epsilon = .998, h_k_param =h_k_p)
print(A)
gt.displaygraph(A,20*np.ones(n),labels = False,layout = 'spring', plt_name = 'whole_t60_shed1.png')

A, node_list = gt.adjacency_matrix(Data,distance,epsilon = .999, h_k_param =h_k_p)
print(A)
gt.displaygraph(A,20*np.ones(n),labels = False,layout = 'spring', plt_name = 'whole_t60_shed2.png')


