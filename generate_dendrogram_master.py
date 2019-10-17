import sys
sys.path.append('./pythonScripts')
sys.path.append('/data3/darpa/calcom')
import numpy as np 
import graph_tools_construction1 as gt 
import calcom
import modules
import impute as imp
from sklearn import datasets, linear_model
from sklearn.preprocessing import normalize
from numpy import genfromtxt
import matplotlib.pyplot as plt
import scipy.spatial as sp
import csv


ccd = calcom.io.CCDataSet('./data/ccd_gse73072_geneid.h5')

def make_dendrogram(t, epsilon, h_k_p, distance, ccd, filename, title):

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


	A, node_list = gt.adjacency_matrix(Data,distance,epsilon,h_k_param =h_k_p)
	l,k = A.shape
	for i in range(l):
		print(A[i,:])

	if gt.connected_components(A) != 1:
		print('network disconnected')
		return

	n,m = Data.shape;

	clst_adj = [];
	nodes = np.zeros((1,m));
	nodes[0,:] = np.arange(0,m);
	clst_node = [];
	all_clusters_node = [];

	gt.cluster_laplace(A, clst_adj, nodes, 1, clst_node, all_clusters_node)

	gt.plot_dendrogram(all_clusters_node,A, data,'cut_edges', filename, title)

t= 60
epsilon = 0
h_k_p = 2
# distance = 'mutual_information'
# title = 'Shedder Data at 60 Hours After Infection'
# filename = 'den_t60_mi_norm.png'

# make_dendrogram(t, epsilon, h_k_p, distance, ccd, filename, title)

distance = 'correlation'
title = 'Shedder Data at 60 Hours After Infection'
filename = 'den_t60_c.png'

make_dendrogram(t, epsilon, h_k_p, distance, ccd, filename, title)

distance = 'heatkernel'
title = 'Shedder Data at 60 Hours After Infection'
filename = 'den_t60_hk.png'

make_dendrogram(t, epsilon, h_k_p, distance, ccd, filename, title)

distance = 'parcor'
title = 'Shedder Data at 60 Hours After Infection'
filename = 'den_t60_p.png'

make_dendrogram(t, epsilon, h_k_p, distance, ccd, filename, title)

