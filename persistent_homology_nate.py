import calcom
import numpy as np
import impute as imp
import graph_tools as gt
from ripser import Rips   
from sklearn import datasets

#set epsilon
epsilon = 0;
ratio = .8
distance = 'parcor';
labels = True;

#import data
ccd = calcom.io.CCDataSet('../data/ccd_duke_prospective_cytokine_plasma.h5');

dta = np.zeros(78);

dta[0:39] = ccd.find('time_id', -2)
dta[39:78] = ccd.find('time_id', -24)


#get a vector for a subject at a given time
X_raw = np.zeros((78,30));
X = np.zeros((78,30));
for i in range(dta.size):
	X_raw[i,:] = np.array(ccd.data[int(dta[i])]);

#fix nans and zeros
X = np.transpose(imp.impute_llod_nans(np.transpose(X_raw),ratio));

n,m = X.shape;

#logtransform
for i in range(n):
	for j in range(m):
		if X[i,j] != 0:
			X[i,j]= np.log(X[i,j]);

#build adjacency matrix
A = gt.adjacency_matrix(X,distance, epsilon);

A = 1-A;

rips = Rips();
rips.fit_transform(A,True);
rips.plot() 

taco = 0;
for kk in [24,48,72]:
	taco = taco+1;
	n = ccd.find('time_id', kk).size;
	dta = np.zeros(n);

	dta = ccd.find('time_id', kk)

	#get a vector for a subject at a given time
	X_raw = np.zeros((n,30));
	X = np.zeros((n,30));
	for i in range(dta.size):
		X_raw[i,:] = np.array(ccd.data[int(dta[i])]);

	#fix nans and zeros
	X = np.transpose(imp.impute_llod_nans(np.transpose(X_raw),ratio));

	n,m = X.shape;

	#logtransform
	for i in range(n):
		for j in range(m):
			if X[i,j] != 0:
				X[i,j]= np.log(X[i,j]);

	##build adjacency matrix
	A = gt.adjacency_matrix(X,distance, epsilon)
	A = 1-A;

	rips = Rips();
	rips.fit_transform(A,True);
	rips.plot() 

