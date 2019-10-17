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


def write_csv(filename, mydict):
	with open(filename, 'w') as csv_file:
	    writer = csv.writer(csv_file)
	    for key, value in mydict.items():
	       writer.writerow([key, value])



#WILL HAVE TO REWORK COMMENTED CODE TO GET IT WORKING
# labels = ccd.generate_labels('shedding', make_dict = False)
# times = {}
# for i in range(720):
# 		temp = ccd.find('time_id', i-38)
# 		print(temp)
# 		if temp != []:
# 			times[i-38] = temp
# print(list(times.keys()))

#num_times = len(list(times.keys()))

# dt = genfromtxt('./data/pathway404.csv',delimiter = ',')
# data_raw = dt[1:,1:]
def testit(ccd,ids):
	times = [0, 2, 4, 5, 8, 10, 12, 16, 18, 20, 21, 22, 24, 26, 29, 30, 34, 36, 42, 45, 46, 48, 50, 53, 58, 60, 66, 69, 70, 72, 74, 77, 82, 84, 90, 93, 94, 96, 98, 101, 106, 108, 114, 118, 120, 122, 125, 130, 132, 136, 138, 142, 146, 162, 166, 170, 680]

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
		#ids = [12,14,39,32,28,5,45,52,15,55,41,40,50,17]
		Data = Data[:,ids]
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

ccd = calcom.io.CCDataSet('../data/ccd_gse73072_geneid.h5')

#hk supA
# ids = [ 0, 33, 54, 13,  8,  6, 1, 34, 27,  5,  4,  3,  9,  7,  2]
# accuracy, bsr = testit(ccd,ids)
# write_csv('ssvm_bsr_h_win_supA.csv', accuracy)
# write_csv('ssvm_acc_h_win_supA.csv', bsr)
#cor
#same size labels
ids = [41, 43, 18, 16,  4,  7,  8,  2,  9,  0,  5, 13,  1,  6,  3]
#other
#ids = [41,  2, 18, 21,  7,  4,  0,  8, 26, 29,  9,  1,  5, 6,  3]
accuracy, bsr = testit(ccd,ids)
write_csv('ssvm_bsr_c.csv', accuracy)
write_csv('ssvm_acc_c.csv', bsr)
# #hk
#same size labels
ids = [31, 45, 28, 40,  6, 32,  0,  1,  3,  8,  7,  5,  9,  4,  2]
#other
#ids = [ 0, 33, 54, 13,  8,  6, 1, 34, 27,  5,  4,  3,  9,  7,  2]
accuracy, bsr = testit(ccd,ids)
write_csv('ssvm_bsr_h.csv', accuracy)
write_csv('ssvm_acc_h.csv', bsr)
#rand
#same size labels
ids = [25, 16, 10, 52, 11,  7,  4,  6,  9,  5,  8,  2,  1,  0,  3]
#other
#ids = [15, 16, 22, 18, 32,  6,  5,  7,  8,  9, 0,  1,  4,  2,  3]
accuracy, bsr = testit(ccd,ids)
write_csv('ssvm_bsr_r.csv', accuracy)
write_csv('ssvm_acc_r.csv', bsr)

