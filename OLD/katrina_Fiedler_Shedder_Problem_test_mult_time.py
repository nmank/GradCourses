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

def gen_res(ccd,times,sim_msr, hk_param, width):
	itr = 0
	q0 = {'time_id': np.arange(-50,0)}
	q1 = {'time_id': times, 'shedding':True}
 
	idx0 = ccd.find(q0)
	idx1 = ccd.find(q1)
	idx0 = idx0[:idx1.size]
	idx1 = idx1[:idx0.size]
	idx = np.hstack([ idx0, idx1 ])
	print(idx)
	labels = np.hstack([ np.repeat('control', len(idx0)), np.repeat('shedder', len(idx1)) ])
	lbl_target = []
	for l in labels:
		if l == 'shedder':
			lbl_target.append(1)
		else:
			lbl_target.append(0)
	modules.pathway_info.append_pathways_to_ccd(ccd)

	data = ccd.generate_data_matrix(idx=idx, features = "REACTOME_INTERFERON_ALPHA_BETA_SIGNALING")
	Data = np.log(data)
	N,M = Data.shape

	# for ii in range(threshhold_itr):
	# 	#only for heat kernel
	# 	for jj in range(hk_param_itr):

	#random or other split
	if sim_msr == 'random':
		nds = []
		partitions = {}
		A = gt.random_graph(N)
	else:
		nds = []
		partitions = {}
		#generate adjacency matrix
		A, nds= gt.adjacency_matrix(Data.T,sim_msr,threshhold,hk_param)
		
	Amin = np.min(A + np.eye(A.shape[0]))
	Amax = np.max(A)

		#normalize A stretch entries between 0 and 1
		# nn,mm = A.shape
		# minA = np.min(A + 10*np.eye(nn))
		# maxA = np.max(A)
		# for i in range(nn):
		# 	for j in range(nn):
		# 		if A[i,j] != 0:
		# 			A[i,j] = (A[i,j]-minA)/(maxA - minA)
		# 			if A[i,j] < threshhold:
		# 				A[i,j] = 0
	acc = []
	bsr = []
	for mm in range(width):
		if mm > 0:
			for i in range(N):
				for j in range(N):
					if A[i,j] < (mm/width)*(Amax - Amin) + Amin:
						A[i,j] = 0
			#check if the graph is connected
			if gt.connected_components(A) > 1:
				break
			else:
				partitions= gt.laplace_partition(A, False)
				subject1 = partitions[0]
				subject2 = partitions[1]
				lbl_approx1 = np.zeros(len(partitions[0])+len(partitions[1]))
				lbl_approx2 = np.zeros(len(partitions[0])+len(partitions[1]))
				for i in subject1:
					lbl_approx1[int(i)] = 1
				for i in subject2:
					lbl_approx2[int(i)] = 1
				confusion_matrix1 = calcom.metrics.ConfusionMatrix()
				confusion_matrix2 = calcom.metrics.ConfusionMatrix()
				confusion_matrix1.evaluate(lbl_target, lbl_approx1)
				confusion_matrix2.evaluate(lbl_target, lbl_approx2)
				acc1 = confusion_matrix1.results['acc']
				acc2 = confusion_matrix2.results['acc']
				bsr1 = confusion_matrix1.results['bsr']
				bsr2 = confusion_matrix2.results['bsr']

			if acc1 >= acc2:
				acc.append(acc1)
				bsr.append(bsr1)
			else:
				acc.append(acc2)
				bsr.append(bsr2)
	acc0 = np.mean(acc)
	bsr0 = np.mean(bsr)

	return acc0, bsr0

def write_csv(filename, mydict):
	with open(filename, 'w') as csv_file:
	    writer = csv.writer(csv_file)
	    for key, value in mydict.items():
	       writer.writerow([key, value])


threshhold = 0
hk_param = 2
bin_size = 2
width = 10

ccd = calcom.io.CCDataSet('../data/ccd_gse73072_geneid.h5')

best = {}
best_bin = 0
best_val = 0
# for bin_size in range(20):
# 	if bin_size > 11:
bin_size = 5
times = make_bins(bin_size)
hk_acc = {}
hk_bsr = {}
r_acc = {}
r_bsr = {}
c_acc = {}
c_bsr = {}
p_acc = {}
p_bsr = {}
for t in times:
	sim_msr = 'heatkernel'
	hk_acc[repr(t)], hk_bsr[repr(t)] = gen_res(ccd,t,sim_msr, hk_param, width)

	# sim_msr = 'mutual_information'
	# mi_acc[repr(t)], mi_bsr[repr(t)] = gen_res(ccd,t,sim_msr, hk_param, width)

	sim_msr = 'correlation'
	c_acc[repr(t)], c_bsr[repr(t)] = gen_res(ccd,t,sim_msr, hk_param, width)

	sim_msr = 'parcor'
	p_acc[repr(t)], p_bsr[repr(t)] = gen_res(ccd,t,sim_msr, hk_param, width)

	sim_msr = 'random'
	r_acc[repr(t)], r_bsr[repr(t)] = gen_res(ccd,t,sim_msr, hk_param, width)

#tmp = np.mean([np.mean(list(hk_bsr.values())),np.mean(list(mi_bsr.values())),np.mean(list(c_bsr.values())),np.mean(list(p_bsr.values()))])
#tmp = np.mean([np.mean(list(hk_bsr.values())),np.mean(list(mi_bsr.values())),np.mean(list(c_bsr.values()))])
# if tmp > best_val:
# 	best_bin = bin_size
# 	best_val = tmp
print('writing to csv files')
print('bin size is' + str(bin_size))
print('------------------------------')
write_csv('hk_acc'+str(bin_size)+'.csv',hk_acc)
write_csv('hk_bsr'+str(bin_size)+'.csv',hk_bsr)
# write_csv('mi_acc'+str(bin_size)+'.csv',mi_acc)
# write_csv('mi_bsr'+str(bin_size)+'.csv',mi_bsr)
write_csv('c_acc'+str(bin_size)+'.csv',c_acc)
write_csv('c_bsr'+str(bin_size)+'.csv',c_bsr)
write_csv('p_acc'+str(bin_size)+'.csv',p_acc)
write_csv('p_bsr'+str(bin_size)+'.csv',p_bsr)
write_csv('r_acc'+str(bin_size)+'.csv',r_acc)
write_csv('r_bsr'+str(bin_size)+'.csv',r_bsr)



# bin_size = 0
# times = make_bins(bin_size)
# hk_acc = {}
# hk_bsr = {}
# #mi_acc = {}
# #mi_bsr = {}
# c_acc = {}
# c_bsr = {}
# r_acc = {}
# r_bsr = {}
# for t in times:
# 	sim_msr = 'heatkernel'
# 	hk_acc[repr(t)], hk_bsr[repr(t)] = gen_res(ccd,t,sim_msr, hk_param, width)

# 	# sim_msr = 'mutual_information'
# 	# mi_acc[repr(t)], mi_bsr[repr(t)] = gen_res(ccd,t,sim_msr, hk_param, width)

# 	sim_msr = 'correlation'
# 	c_acc[repr(t)], c_bsr[repr(t)] = gen_res(ccd,t,sim_msr, hk_param, width)


# 	sim_msr = 'random'
# 	r_acc[repr(t)], r_bsr[repr(t)] = gen_res(ccd,t,sim_msr, hk_param, width)

# #tmp = np.mean([np.mean(list(hk_bsr.values())),np.mean(list(mi_bsr.values())),np.mean(list(c_bsr.values())),np.mean(list(p_bsr.values()))])
# #tmp = np.mean([np.mean(list(hk_bsr.values())),np.mean(list(mi_bsr.values())),np.mean(list(c_bsr.values()))])
# # if tmp > best_val:
# # 	best_bin = bin_size
# # 	best_val = tmp
# print('writing to csv files')
# print('bin size is' + str(bin_size))
# print('------------------------------')
# write_csv('hk_acc'+str(bin_size)+'.csv',hk_acc)
# write_csv('hk_bsr'+str(bin_size)+'.csv',hk_bsr)
# # write_csv('mi_acc'+str(bin_size)+'.csv',mi_acc)
# # write_csv('mi_bsr'+str(bin_size)+'.csv',mi_bsr)
# write_csv('c_acc'+str(bin_size)+'.csv',c_acc)
# write_csv('c_bsr'+str(bin_size)+'.csv',c_bsr)
# write_csv('r_acc'+str(bin_size)+'.csv',r_acc)
# write_csv('r_bsr'+str(bin_size)+'.csv',r_bsr)



# print(best_time)
# print(best_val)
		



