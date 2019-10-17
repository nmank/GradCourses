import sys
sys.path.append('./pythonScripts')
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


def write_csv(filename, mydict):
	with open(filename, 'w') as csv_file:
	    writer = csv.writer(csv_file)
	    for key, value in mydict.items():
	       writer.writerow([key, value])



# def testit(times, ccd,ids):
def testit(times, ccd):
	accuracy = {}
	bsr = {}
	for t in times:
		q0 = {'time_id': np.arange(-50,0)}
		q1 = {'time_id': t, 'shedding':True}

		idx0 = ccd.find(q0)
		idx1 = ccd.find(q1)
		idx0 = idx0[:idx1.size]
		idx1 = idx1[:idx0.size]
		idx = np.hstack([ idx0, idx1 ])
		labels = np.hstack([ np.repeat('control', len(idx0)), np.repeat('shedder', len(idx1)) ])
		modules.pathway_info.append_pathways_to_ccd(ccd)

		data = ccd.generate_data_matrix(idx=idx, features = "REACTOME_INTERFERON_ALPHA_BETA_SIGNALING")
		Data = np.log(data)

		#supA
		# ids = [12,14,39,32,28,5,45,52,15,55,41,40,50,17]
		# Data = Data[:,ids]
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

		accuracy[repr(t)] = confusion_matrix.results['acc']
		bsr[repr(t)] = confusion_matrix.results['bsr']

	return accuracy, bsr



def make_bins(bin_size):
	times = [0, 2, 4, 5, 8, 10, 12, 16, 18, 20, 21, 22, 24, 26, 29, 30, 34, 36, 42, 45, 46, 48, 50, 53, 58, 60, 66, 69, 70, 72, 74, 77, 82, 84, 90, 93, 94, 96, 98, 101, 106, 108, 114, 118, 120, 122, 125, 130, 132, 136, 138, 142, 146]
	#times = [0, 2, 4, 5, 8, 10, 12, 16, 18, 20, 21, 22, 24, 26, 29, 30, 34, 36, 42, 45, 46, 48, 50, 53, 58, 60, 66, 69, 70, 72, 74, 77, 82, 84, 90, 93, 94, 96, 98, 101, 106, 108, 114, 118, 120, 122, 125, 130, 132, 136, 138, 142, 146, 162, 166, 170, 680]

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

ccd = calcom.io.CCDataSet('../data/ccd_gse73072_geneid.h5')

# for bin_size in range(20):
# 	if bin_size>0:
# 		bins = make_bins(bin_size)
# 		accuracy, bsr = testit(bins,ccd)
		# write_csv('ssvm_bsr_'+str(bin_size)+'.csv', accuracy)
		# write_csv('ssvm_acc_'+str(bin_size)+'.csv', bsr)

bin_size = 0
bins = make_bins(bin_size)
accuracy, bsr = testit(bins,ccd)
write_csv('ssvm_bsr_'+str(bin_size)+'.csv', accuracy)
write_csv('ssvm_acc_'+str(bin_size)+'.csv', bsr)

bin_size = 5
bins = make_bins(bin_size)
accuracy, bsr = testit(bins,ccd)
write_csv('ssvm_bsr_'+str(bin_size)+'.csv', accuracy)
write_csv('ssvm_acc_'+str(bin_size)+'.csv', bsr)

bin_size = 6
bins = make_bins(bin_size)
accuracy, bsr = testit(bins,ccd)
write_csv('ssvm_bsr_'+str(bin_size)+'.csv', accuracy)
write_csv('ssvm_acc_'+str(bin_size)+'.csv', bsr)



# bin_size = 6
# bins = make_bins(bin_size)

#cor
#old_thresh
# ids = [12, 19, 20, 21, 22, 23, 24, 25, 26, 54, 18, 27,  6,  5,  3]
#new_thresh
# ids = [22, 24, 25, 26, 54, 28, 14,  1, 49,  0,  8,  5,  9,  6,  3]
# accuracy, bsr = testit(bins,ccd,ids)
# write_csv('ssvm_bsr_'+str(bin_size)+'c.csv', accuracy)
# write_csv('ssvm_acc_'+str(bin_size)+'c.csv', bsr)
# #pcor
# #old_thresh
# #ids = [26, 34, 30, 41, 14,  9,  8,  6,  1,  2,  0,  5,  3,  7,  4]
# #new_thresh
# ids = [25, 18, 41, 17,  6, 20,  8,  3,  2,  9,  5,  7,  4,  1,  0]
# accuracy, bsr = testit(bins,ccd,ids)
# write_csv('ssvm_bsr_'+str(bin_size)+'p.csv', accuracy)
# write_csv('ssvm_acc_'+str(bin_size)+'p.csv', bsr)
# #hk
# #old_thresh
# # ids = [ 6,  7,  8, 32,  9, 11, 12, 14, 15, 16, 17, 28, 30, 10, 55]
# #new_thresh
# ids = [ 8, 30, 44, 46, 35, 36,  1,  2,  3,  7, 5,  6,  9,  4,  0]
# accuracy, bsr = testit(bins,ccd,ids)
# write_csv('ssvm_bsr_'+str(bin_size)+'h.csv', accuracy)
# write_csv('ssvm_acc_'+str(bin_size)+'h.csv', bsr)
# #rand
# #old_thresh
# #ids = [14, 50, 35, 39, 31,  7,  9,  8,  6,  4,  5,  3,  2,  1,  0]
# #new_thresh
# ids = [18, 20, 29, 11, 43, 5, 9,  8,  6,  3,  2,  7,  4,  1,  0]
# accuracy, bsr = testit(bins,ccd,ids)
# write_csv('ssvm_bsr_'+str(bin_size)+'r.csv', accuracy)
# write_csv('ssvm_acc_'+str(bin_size)+'r.csv', bsr)

