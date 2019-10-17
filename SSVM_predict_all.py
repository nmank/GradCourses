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


def testit(data, labels):

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

def write_csv(filename, mydict):
	with open(filename, 'w') as csv_file:
	    writer = csv.writer(csv_file)
	    for key, value in mydict.items():
	       writer.writerow([key, value])

ccd = calcom.io.CCDataSet('../data/ccd_gse73072_geneid.h5')

times = [48, 50, 53, 58, 60, 66, 69, 70, 72, 74, 77, 82, 84, 90, 93, 94, 96, 98, 101, 106, 108, 114, 118, 120, 122, 125, 130, 132, 136, 138, 142, 146]
res = {}
for t in times:
	q0 = {'time_id': np.arange(-50,0)}
	idx0 = ccd.find(q0)
	q1 = {'time_id': t, 'shedding':True}
	idx1 = ccd.find(q1)
	idx0 = idx0[:idx1.size]
	idx1 = idx1[:idx0.size] #same number of shedder and controls
	idx = np.hstack([ idx0, idx1 ])
	Data = ccd.generate_data_matrix(idx=idx)
	data = np.log(Data)
	labels = np.hstack([ np.repeat('control', len(idx0)), np.repeat('shedder', len(idx1)) ])
	res[repr(t)] = testit(data,labels)

write_csv('ssvm_all0.csv', res)

times = make_bins(5)
res = {}
for t in times:
	q0 = {'time_id': np.arange(-50,0)}
	idx0 = ccd.find(q0)
	q1 = {'time_id': t, 'shedding':True}
	idx1 = ccd.find(q1)
	idx0 = idx0[:idx1.size] #same number of shedder and controls
	idx1 = idx1[:idx0.size]
	idx = np.hstack([ idx0, idx1 ])
	Data = ccd.generate_data_matrix(idx=idx)
	data = np.log(Data)
	labels = np.hstack([ np.repeat('control', len(idx0)), np.repeat('shedder', len(idx1)) ])
	res[repr(t)] = testit(data,labels)

write_csv('ssvm_all5.csv', res)


times = make_bins(6)
res = {}
for t in times:
	q0 = {'time_id': np.arange(-50,0)}
	idx0 = ccd.find(q0)
	q1 = {'time_id': t, 'shedding':True}
	idx1 = ccd.find(q1)
	idx0 = idx0[:idx1.size] #same number of shedder and controls
	idx1 = idx1[:idx0.size]
	idx = np.hstack([ idx0, idx1 ])
	Data = ccd.generate_data_matrix(idx=idx)
	data = np.log(Data)
	labels = np.hstack([ np.repeat('control', len(idx0)), np.repeat('shedder', len(idx1)) ])
	res[repr(t)] = testit(data,labels)

write_csv('ssvm_all6.csv', res)

	