import calcom
import modules
import numpy as np
import sys
sys.path.append('../pythonScripts')
import graph_tools_construction as gt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

epsilon = .3
distance = 'parcor'
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

gt.displaygraph(A,80*np.ones(n),labels = False,layout = 'spring', plt_name = 'network_t60_pcor_thresh9.png')
