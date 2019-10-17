import sys
sys.path.append('../pythonScripts')
sys.path.append('/data3/darpa/calcom')
import numpy as np 
import graph_tools_construction as gt 
import calcom
import modules
import impute as imp
from sklearn import datasets, linear_model
from sklearn.preprocessing import normalize
from numpy import genfromtxt
import matplotlib.pyplot as plt
import scipy.spatial as sp
import csv

def testit(y, Data, labels):
	data = Data[:,y]

	n,d = data.shape

	labels_b = []
	for l in labels:
		if l == 'shedder':
			labels_b.append(True)
		else:
			labels_b.append(False)

	train_idx = 2*np.arange(n//2)
	test_idx = list(set(range(n))- set(train_idx))
	labels_b_tr = []
	for i in train_idx:
		labels_b_tr.append(labels[i])
	labels_b_tst = []
	for i in test_idx:
		labels_b_tst.append(labels[i])
	ssvm = calcom.classifiers.SSVMClassifier()
	ssvm.fit(data[train_idx],labels_b_tr)
	predicted_labels = ssvm.predict(data[test_idx])
	confusion_matrix = calcom.metrics.ConfusionMatrix()
	confusion_matrix.evaluate(labels_b_tst, predicted_labels)
	return confusion_matrix.results['acc']

def make_bins(bin_size):
	times = [0, 2, 4, 5, 8, 10, 12, 16, 18, 20, 21, 22, 24, 26, 29, 30, 34, 36, 42, 45, 46, 48, 50, 53, 58, 60, 66, 69, 70, 72, 74, 77, 82, 84, 90, 93, 94, 96, 98, 101, 106, 108, 114, 118, 120, 122, 125, 130, 132, 136, 138, 142, 146]
	bins = []
	i=0
	t1 = 0
	t2 = 0
	while i < len(times):  
		tmp = []  
		while t2 - t1 <= bin_size:   
			tmp.append(t2) 
			i += 1 
			if i < len(times): 
				t2 = times[i] 
			else: 
				break  
		t1 = t2 
		bins.append(tmp) 
	return bins



def test_param(width, times, distance, h_k_p = 2):
	iiiii=0
	best_thresh = {}
	best_cluster = {}
	best_acc = {}
	for t in times:
		#impoprt data
		ccd = calcom.io.CCDataSet('../data/ccd_gse73072_geneid.h5')
		q0 = {'time_id': np.arange(-50,0)}
		q1 = {'time_id': t, 'shedding':True}
		idx0 = ccd.find(q0)
		idx1 = ccd.find(q1)
		idx1 = idx1[:idx0.size]
		idx0 = idx0[:idx1.size]#same number of shedder and controls
		idx = np.hstack([ idx0, idx1 ])
		labels = np.hstack([ np.repeat('control', len(idx0)), np.repeat('shedder', len(idx1)) ])
		modules.pathway_info.append_pathways_to_ccd(ccd)
		data = ccd.generate_data_matrix(idx=idx, features = "REACTOME_INTERFERON_ALPHA_BETA_SIGNALING")
		Data = np.log(data)
		#turn on or off if you wnat networks of shedders or of all
		data_shed =  ccd.generate_data_matrix(idx=idx1, features = "REACTOME_INTERFERON_ALPHA_BETA_SIGNALING")
		data_shed1 = np.log(data_shed)
		#initialize best variables		
		nds_best_tmp = list(range(56))
		eps_best_tmp = 0
		acc_best_tmp = testit(nds_best_tmp,Data,labels)

		#generate adjacency matrix
		if distance == 'random':
			N,M = Data.shape
			A = gt.random_graph(M)
			node_list = []
		else:
			# A, node_list = gt.adjacency_matrix(data_shed[:,nds_best_tmp],distance,0,h_k_param =h_k_p)
			A, node_list = gt.adjacency_matrix(data_shed1[:,nds_best_tmp],distance,0,h_k_param =h_k_p)
		A_big = A

		#init while loop stopping criteria and iteration counter
		continue_cutting = True
		cut_itr = 0
		while continue_cutting == True:

			#init epsilon iteration variables
			acc_tmp = []
			eps_tmp = []
			nds_tmp = []
			nds_tmp.append(nds_best_tmp)
			acc_tmp.append(0)
			eps_tmp.append(eps_best_tmp)

			#loop over epsilon possibilities
			for eps in range(int(1/width)):
				epsilon = eps*width	

				#thresholding
				maxA = np.max(A)
				#fixed, not in thesis
				minA = np.min(A + np.eye(A.shape[0]))


				#scale the threshold
				threshold = (np.max(A)-np.min(A))*epsilon + np.min(A)
				ii,jj =A.shape
				for i in range(ii):
					for j in range(ii):
						if i > j and A[i,j] < threshold:
							A[i,j] = 0
							A[j,i] = 0

				#if the network is disconnected or has one node stop looping through epsilons
				if A.size == 1 and cut_itr != 0:
					print('----------------------')
					print('SINGLE NODE')
					print(nds_tmp[eps])
					print('----------------------')
					break
				elif gt.connected_components(A) != 1 and cut_itr != 0:
					print('----------------------')
					print('DISCONNECTED GRAPH')
					print(threshold)
					print('----------------------')
					break
				else:
					res = []
					#cut the network
					#normcut
					n0, n1 = gt.laplace_partition(A, True)

					#debugging prints
					# print(n0)
					# print(n1)

					#append the SSVM results for each bank
					res.append(testit(list(n0[:,0]),Data,labels))
					res.append(testit(list(n1[:,0]),Data,labels))

					#choose the best bank and save the results
					if res[0] >= res[1]:
						nds_tmp.append([nds_tmp[0][i] for i in list(n0[:,0])])
						eps_tmp.append(threshold)
						acc_tmp.append(res[0])
					else:
						nds_tmp.append([nds_tmp[0][i] for i in list(n1[:,0])])
						eps_tmp.append(threshold)
						acc_tmp.append(res[1])

					iiiii+=1

			#take the best threshold
			acc_max_arg = np.argmax(acc_tmp)

			#stop cutting if current bank doesn't have better SSVM accuracy than larger network
			if acc_best_tmp > acc_tmp[acc_max_arg] and cut_itr != 0:
				continue_cutting = False
			#otherwise append save the improved accuracy, threshold, and the nodes
			else:
				acc_best_tmp =  acc_tmp[acc_max_arg]
				nds_best_tmp =  nds_tmp[acc_max_arg]
				eps_best_tmp =  eps_tmp[acc_max_arg]
				
				#define a new network that contains the nodes in that cut
				A= np.zeros((len(nds_best_tmp),len(nds_best_tmp)))
				for i in range(len(nds_best_tmp)):
					for j in range(len(nds_best_tmp)):
						A[i,j] = A_big[nds_best_tmp[i],nds_best_tmp[j]]

				#debugging prints
				print(nds_best_tmp)
				#print(cut_itr)

			cut_itr += 1
			


		best_thresh[repr(t)] = eps_best_tmp
		best_cluster[repr(t)] = nds_best_tmp
		best_acc[repr(t)] = acc_best_tmp
		print('----------------------')
		print(distance)
		print('time')
		print(t)
		print('best cluster')
		print(nds_best_tmp)
		print('best accuracy')
		print(acc_best_tmp)
		print('----------------------')	

	print(iiiii)
	return best_thresh, best_cluster, best_acc


def write_csv(filename, mydict):
	with open(filename, 'w') as csv_file:
	    writer = csv.writer(csv_file)
	    for key, value in mydict.items():
	       writer.writerow([key, value])


epsilon = 0
h_k_p = 2
min_clst_sz= 2
low_clst_sz = 5
high_clst_sz = 15
width = .1


#normalization is turned off

#change to a single number if doing a specific time
#[-38, -30, -23, -21, -11, 0, 2, 4, 5, 8, 10, 12, 16, 18, 20, 21, 22, 24, 26, 29, 30, 34, 36, 42, 45, 46, 48, 50, 53, 58, 60, 66, 69, 70, 72, 74, 77, 82, 84, 90, 93, 94, 96, 98, 101, 106, 108, 114, 118, 120, 122, 125, 130, 132, 136, 138, 142, 146, 162, 166, 170, 680]
times = make_bins(6)

distance = 'heatkernel'
threshold_h, cluster_h, acc_h = test_param(width, times, distance,h_k_p)
write_csv('hk_thresh_clst.csv', threshold_h)
write_csv('hk_cluster_clst.csv', cluster_h)
write_csv('hk_acc_clst.csv', acc_h)

# distance = 'mutual_information'
# threshold_m, cluster_m, acc_m= test_param(width, times, distance,h_k_p)
# write_csv('mi_thresh_clst.csv', threshold_m)
# write_csv('mi_cluster_clst.csv', cluster_m)
# write_csv('mi_acc_clst.csv', acc_m)

distance = 'correlation'
threshold_c, cluster_c, acc_c= test_param(width, times, distance,h_k_p)
write_csv('c_thresh_clst.csv', threshold_c)
write_csv('c_cluster_clst.csv', cluster_c)
write_csv('c_acc_clst.csv', acc_c)

distance = 'parcor'
threshold_p, cluster_p, acc_p= test_param(width, times, distance,h_k_p)
write_csv('p_thresh_clst.csv', threshold_p)
write_csv('p_cluster_clst.csv', cluster_p)
write_csv('p_acc_clst.csv', acc_p)

distance = 'random'
threshold_r, cluster_r, acc_r= test_param(width, times, distance,h_k_p)
write_csv('r_thresh_clst.csv', threshold_r)
write_csv('r_cluster_clst.csv', cluster_r)
write_csv('r_acc_clst.csv', acc_r)





