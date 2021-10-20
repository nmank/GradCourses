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


def get_sa(time_weight,distance,epsilon,times, ccd,h_k_p = 2):
	#with pathway404
	for t in times:
		
		q0 = {'time_id': np.arange(-50,0)}
		q1 = {'time_id': t, 'shedding':True}

		idx = ccd.find(q1)
		# idx0 = ccd.find(q0)
		# idx1 = ccd.find(q1)
		# idx0 = idx0[:idx1.size]
		# idx = np.hstack([ idx0, idx1 ])
		# labels = np.hstack([ np.repeat('control', len(idx0)), np.repeat('shedder', len(idx1)) ])
		modules.pathway_info.append_pathways_to_ccd(ccd)

		data = ccd.generate_data_matrix(idx=idx, features = "REACTOME_INTERFERON_ALPHA_BETA_SIGNALING")
		Data = np.log(data)

		A = []
		node_list = []
		times_shown = []
		# #if using partial correlation only use tall data matrices
		if distance == 'random':
			A1 = gt.random_graph(56)
			n1 = []
		else:
			A1, n1 = gt.adjacency_matrix(Data,distance,epsilon)
		node_list.append(n1)
		A.append(A1)
			

	n,m = A[0].shape

	max_time = np.max(times)
	N = m*len(A)
	sA = np.zeros((N,N))
	for i in range(len(A)):
		for j in range(len(A)):
			if i == j:
				sA[i*m:(i+1)*m, i*m:(i+1)*m] = A[i]
			if i == j+1:
				time_weight1 = np.mean(A)
				sA[i*m:(i+1)*m,j*m:(j+1)*m] = time_weight1*np.eye(m)
	return sA, len(A), m

def get_scores(sA, centrality, l, m):
	N,M = sA.shape


	#largest eigenvector centrality
	if centrality == 'large_evec':
		W,V = np.linalg.eig(sA)
		largevec = V[:,[W.argmax()]]


		scores1 = np.zeros(m)
		for i in range(m):
			for j in range(l):
				scores1[i] += np.abs(largevec[i+m*j][0])
		return scores1


	#page rank
	if centrality == 'page_rank':
		M = np.zeros((N,N))
		for i in range(N): 
			M[:,i] = sA[:,i]/np.sum(sA[:,i])

		#taken from da wikipedia
		eps = 0.001
		d = 0.85

		v = np.random.rand(N, 1)
		v = v / np.linalg.norm(v, 1)
		last_v = np.ones((N, 1), dtype=np.float32) * 100
		M_hat = (d * M) + (((1 - d) / N) * np.ones((N, N), dtype=np.float32))
		    
		while np.linalg.norm(v - last_v, 2) > eps:
			last_v = v
			v = np.matmul(M_hat, v)

		scores1 = np.zeros(m)
		for i in range(m):
			for j in range(l):
				scores1[i] += v[i+m*j]
		return scores1



def ssvm_test(scores,k, ccd,m):
	print(scores)
	y = []
	y.append(scores.argmax())
	argtmp = 0
	for i in range(k-1):
		tmp = np.min(scores) 
		for j in range(m): 
			if scores[j] > tmp and j not in y: 
				tmp  = scores[j]
				argtmp = j
		y.append(argtmp)
	print(y)


	q0 = {'time_id': np.arange(-50,0)}
	q1 = {'time_id': times, 'shedding':True}

	idx0 = ccd.find(q0)
	idx1 = ccd.find(q1)
	idx1 = idx1[:idx0.size]
	idx0 = idx0[:idx1.size]
	num_shd = len(idx1)
	idx = np.hstack([ idx0, idx1 ])
	labels = np.hstack([ np.repeat('control', len(idx0)), np.repeat('shedder', len(idx1)) ])
	modules.pathway_info.append_pathways_to_ccd(ccd)

	data = ccd.generate_data_matrix(idx=idx, features = "REACTOME_INTERFERON_ALPHA_BETA_SIGNALING")
	Data = data[:,y]

	n,d = Data.shape



	labels_b = []
	for l in labels:
		if l == 'shedder':
			labels_b.append(True)
		else:
			labels_b.append(False)
	train_idx = 2*np.arange(n//2)
	test_idx = list(set(range(n)) - set(train_idx))
	labels_b_tr = []
	for i in train_idx:
		labels_b_tr.append(labels[i])
	labels_b_tst = []
	for i in test_idx:
		labels_b_tst.append(labels[i])
	ssvm = calcom.classifiers.SSVMClassifier()
	ssvm.fit(Data[train_idx],labels_b_tr)
	predicted_labels = ssvm.predict(Data[test_idx])
	confusion_matrix = calcom.metrics.ConfusionMatrix()

	confusion_matrix.evaluate(labels_b_tst, predicted_labels)

	return confusion_matrix.results['acc'], confusion_matrix.results['bsr'], y

def write_csv(filename, mydict):
	with open(filename, 'w') as csv_file:
	    writer = csv.writer(csv_file)
	    for key, value in mydict.items():
	       writer.writerow([key, value])

def write_csv_list(filename, my_list):
	with open(filename, 'w') as csv_file:
	    writer = csv.writer(csv_file)
	    for l in my_list:
	       writer.writerow(l)

def testit(distance, ccd):

	time_weight = .5
	heatkernel_param = 2
	epsilon = 0
	acc_p = {}
	bsr_p = {}
	y_p = {}
	acc_l = {}
	bsr_l = {}
	y_l = {}
	supA, n_times, n_gns = get_sa(time_weight,distance,epsilon,times,ccd,h_k_p = heatkernel_param)
	scrsP = get_scores(supA, 'page_rank',n_times,n_gns)
	scrsL = get_scores(supA, 'large_evec',n_times,n_gns)
	for k in range(25):
		if k > 1:
			accP,bsrP,yP = ssvm_test(scrsP,k,ccd,n_gns)
			acc_p[repr(k)] = accP
			bsr_p[repr(k)] = bsrP
			y_p[repr(k)] = yP
			print(str(k) + distance)
			print('Accuracy:' + str(accP))
			print('------------------------')		
			accL,bsrL,yL = ssvm_test(scrsL,k,ccd,n_gns)
			acc_l[repr(k)] = accL
			bsr_l[repr(k)] = bsrL
			y_l[repr(k)] = yL
			print(str(k) + distance)
			print('large_evec')
			print('Accuracy:' + str(accL))
			print('------------------------')


	write_csv(distance+'acc_p.csv', acc_p)
	write_csv(distance+'bsr_p.csv', bsr_p)
	write_csv(distance+'y_p.csv', y_p)
	write_csv(distance+'acc_l.csv', acc_l)
	write_csv(distance+'bsr_l.csv', bsr_l)
	write_csv(distance+'y_l.csv', y_l)

	amin_p = min(acc_p, key=acc_p.get)
	amin_l = min(acc_l, key=acc_l.get)

	if acc_p[amin_p] <= acc_l[amin_l]:
		best_res1 = [distance, 'page rank', amin_p, acc_p[amin_p], bsr_p[amin_p], y_p]
	else:
		best_res1 = [distance, 'largest eigenvector', amin_l, acc_l[amin_l], bsr_l[amin_l], y_l]

	return best_res1

#ACCURACY WITH ALL THE DATA IS: 0.7857142857142857
#ACCURACY WITH 5.,  6.,  9., 13., 14., 22.
#(time_weight,distance,epsilon,times,centrality,k,h_k_p = 2)

ccd = calcom.io.CCDataSet('../data/ccd_gse73072_geneid.h5')

#find the top k genes

times = [48, 50, 53, 58, 60, 66, 69, 70, 72, 74, 77, 82, 84, 90, 93, 94, 96, 98, 101, 106, 108, 114, 118, 120, 122, 125, 130, 132, 136, 138, 142, 146, 162, 166, 170, 680]


best_res = []
best_res.append(['distance', 'centrality measure', 'number of genes', 'accuracy', 'bsr', 'genes'])
best_res.append(testit('heatkernel', ccd))
best_res.append(testit('correlation', ccd))
best_res.append(testit('parcor', ccd))
#best_res.append(testit('mutual_information', ccd))
best_res.append(testit('random', ccd))

write_csv_list('SA_winners.csv',best_res)
# results_acc = []
# results_bsr = []
# results_gene_set = []
# for i in range(10):
# 	acc,bsr,gene_set = gsp_test(time_weight,distance,epsilon*i,times,centrality,k)
# 	results_acc.append(acc)
# 	results_bsr.append(bsr)
# 	results_gene_set.append(gene_set)




