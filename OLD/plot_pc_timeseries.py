import sys
sys.path.append('./pythonScripts')
import calcom
import numpy as np
import graph_tools_construction as gt
from scipy.special import comb
from numpy import genfromtxt
import matplotlib.pyplot as plt

distance = 'parcor'
labels = True

#load pw404
ccd = calcom.io.CCDataSet('./data/ccd_gse73072_geneid.h5')
labels = ccd.generate_labels('shedding', make_dict = False)
dt = genfromtxt('./data/pathway404.csv',delimiter = ',')
data_raw = dt[1:,1:]
times = {}
for i in range(720):
	temp = ccd.find('time_id', i-38)
	if temp != []:
		times[i-38] = temp

Data = []
# X1 = gt.load_data('pw404', -38, False)
# X2 = gt.load_data('pw404', -30, False)
# X3 = gt.load_data('pw404', -23, False)
# X4 = gt.load_data('pw404', -21, False)
# X5 = gt.load_data('pw404', -11, False)
# Data.append(np.block([[X1], [X2], [X3],[X4], [X5]]))
for i in list(times.keys()):
	if i >= 0:
		Data.append(gt.load_data('pw404', i, True));

#only add tall matrices
#note: only necessary if using partial correaltions
tall_dtimes = []
for i in range(len(Data)): 
	ii,jj = Data[i].shape 
	if ii>jj: 
		tall_dtimes.append(i)

A = []
nodes = []
times_shown = []
#if using correlation use all the data
#for i in range(len(Data)):
#if using partial correlation only use tall data matrices
for i in tall_dtimes:
	A1, n1 = gt.adjacency_matrix(Data[i],'parcor')
	nodes.append(n1)
	A.append(A1)
	#times_shown.append(str(list(times.keys())[i+5]))
	times_shown.append(str(list(times.keys())[i]))

n,m = A[0].shape
T = len(A)
num_edges = int(comb(m,2))


#find the good nodes
#these are the nodes form the 'fat' data matrices
#they cause partial correlations to be zero
# we can check this before calculating the adjacency matrices
# goodgood = []
# for i in range(len(nodes)): 
# 	if np.all(nodes[i] >= .000001): 
# 		print(i) 
# 		goodgood.append(i)

# for i in goodgood:
# 	plt.scatter(range(n),nodes[i])
# plt.show()

taco = np.zeros((m,len(goodgood)))
for j in range(m):
	k = 0
	for i in tall_dtimes:
		taco[j][k] = nodes[i][0]
		k = k+1
plt.plot(taco)
plt.show()

#This might not work
time_series_edges = {}
for i in range(m):
		for j in range(m):
			if i > j:
				tmp = np.zeros(T)
				for t in range(T):
					tmp[t] = A[t][i,j]
				time_series_edges[(i,j)] = tmp

plt.plot(time_series_edges[(4,1)])
plt.show()

# for ii in range(50):
# 	plt.scatter(range(57),time_series_edges[ii])

# plt.show()


