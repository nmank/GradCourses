import sys
sys.path.append('./pythonScripts')
import numpy as np 
import graph_tools_construction as gt 
import calcom
import impute as imp
from sklearn import datasets, linear_model
from sklearn.preprocessing import normalize
from numpy import genfromtxt
import matplotlib.pyplot as plt

ratio = .8
time_weight = 1
distance = 'parcor'



#with pathway404
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

for i in list(times.keys()):
	if i >= 0:
		#to just get shedders
		#X = data_raw[np.intersect1d(np.where(labels ==True), times[i])]
		X = data_raw[times[i]]
		Data.append(np.log(X));



#only add tall matrices
#note: only necessary if using partial correaltions
tall_dtimes = []
for i in range(len(Data)): 
	ii,jj = Data[i].shape 
	if ii>jj: 
		tall_dtimes.append(i)

A = []
node_list = []
times_shown = []
#if using correlation use all the data
#for i in range(len(Data)):
#if using partial correlation only use tall data matrices
for i in tall_dtimes:
	A1, n1 = gt.adjacency_matrix(Data[i],distance)
	node_list.append(n1)
	A.append(A1)
	#if using correlation use all the data
	#times_shown.append(str(list(times.keys())[i+5]))
	times_shown.append(str(list(times.keys())[i+5]))

node_array= np.zeros((len(node_list),node_list[0].size))

for i in range(len(node_list)):
	node_array[i,:] = node_list[i]

fig,ax = plt.subplots()
im = ax.imshow(node_array, cmap='hot', interpolation='nearest')
plt.colorbar(im,orientation='horizontal')
ax.set_title('Partial Correlation Projection Residuals Over Time')
plt.show()


