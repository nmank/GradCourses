# module imports
import pandas
import numpy as np
from numpy import ndarray
from scipy.sparse import csr_matrix
from scipy import *
from matplotlib import pyplot as plt
import graph_tools_construction as gt
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

class SpectralClustering(BaseEstimator):
    '''
    This class is for classification-informed spectral higherarchical clustering.
    We generate an adjacency matrix, then iteratively cut the graph into smaller graphs.
    At each cut, we chose the smaller graph which produces the highest BSR with an SVM on all the data.
    If the chosen smaller graph has a lower bsr than the bigger graph, we stop cutting and return the 
    nodes in the bigger graph.
    '''

    def __init__(self, similarity: str = None, A: ndarray = None):
        '''
        Inputs:
            similarity (string) can be either 'correlation', 'heatkernel' or 'zobs'
            A (numpy array) a precomputed adjacency matrix (If using this parameter, son't input a simiarity)
        '''
        # set params
        self.similarity_ = similarity
        self.A_ = A

    @property
    def similarity(self):
        return self.similarity_

    @property
    def A(self):
        return self.A_


    def fit(self, X: ndarray= None, y: ndarray =None):
        '''
        Generates an adjacency matrix.

        Inputs:
            X (numpy array): a row of X is a datapoint and a column of X corresponds to a node in the graph with adjacency matrix A
            y (numpy array): binary labels for the rows of X (not used)
        '''
        X = check_array(X)

        if self.similarity_ == 'zobs':
            self.A_ = gt.zobs(X, y)
        elif self.similarity_ is not None:
            self.A_ = gt.adjacency_matrix(X, self.similarity_, negative = False)

    def transform(self, X: ndarray = None, y: ndarray  = None, loso = False, fiedler = True):
        '''
        SVM higherarchical clustering.

        Inputs:
            X (numpy array): a row of X is a datapoint and a column of X corresponds to a node in the graph with adjacency matrix A
            y (numpy array): binary labels for the rows of X (not used)
            loso (boolean): True to do leave one subject out ssvm
            fiedler (boolean): True for vanilla laplacian and False for normalized laplacian

        Outputs:
            current_idx (numpy array): a numpy array of the nodes in the module with the best BSR. 
                                    These correspond to columns in X.
            best_bsr (float): the SVM bsr for the module defined by current_idx
        
        '''
        X = check_array(X)
        self.A_ = check_array(self.A_)


        keep_cutting = True

        best_bsrs = []
        best_bsr = self.test_cut_loso(X, y)
        best_bsrs.append(best_bsr)

        current_idx = np.arange(self.A_.shape[0])

        cut_num = 0
        while keep_cutting:
            current_A = self.A_[current_idx,:][:,current_idx]

            n0, n1 = gt.laplace_partition(current_A, fiedler) #false for normalized laplacian

            n0 = n0.T[0]
            n1 = n1.T[0]

            if len(n0)> 0 or len(n1)>0: #change to and for runnable code

                if len(n0) > 0:
                    current_idx0 = current_idx[n0]

                    if loso:
                        bsr0 = self.test_cut_loso(X[:,current_idx0], y)
                    else:
                        bsr0 = self.test_cut(X[:,current_idx0], y)

                else:
                    bsr0 = 0

                
                if len(n1) > 0:
                    current_idx1 = current_idx[n1]

                    if loso:
                        bsr1 = self.test_cut_loso(X[:,current_idx1], y)
                    else:
                        bsr1 = self.test_cut(X[:,current_idx1], y)
                else:
                    bsr1 = 0
                    

                if bsr0 >= best_bsr or bsr1 >= best_bsr:
                    if bsr0 > bsr1:
                        current_idx = current_idx[n0]
                        best_bsr = bsr0
                    else:
                        current_idx = current_idx[n1]
                        best_bsr = bsr1
                else:
                    keep_cutting = False
                
                # print('cut number '+str(cut_num))
                cut_num += 1
            else:
                keep_cutting = False
            
            if len(current_idx) <= 1:
                keep_cutting = False

            best_bsrs.append(best_bsr)
        
        return current_idx, best_bsrs


    def test_cut(self, data, labels):
        '''
        Train and run an SVM classifier on the data and labels.

        Inputs:
            data (numpy array): the data where a datapoint in a row
            labels (numpy array or list) the labels of the rows of data
        
        Outputs:
            bsr (float): the BSR of the SVM classifier on the data and labels
        '''
        clf = make_pipeline(LinearSVC(dual = False))

        clf.fit(data, labels)

        predictions = clf.predict(data)

        bsr = balanced_accuracy_score(predictions, labels) 

        return bsr

    def test_cut_loso(self, data, labels):
        '''
        Train and run an SVM classifier on the data and labels with leave one subject out framework.

        Inputs:
            data (numpy array): the data where a datapoint in a row
            labels (numpy array or list) the labels of the rows of data
        
        Outputs:
            bsr (float): the BSR of the SVM classifier on the data and labels
        '''
        subject_idxs = list(range(data.shape[0]))
        
        predictions = []
        for fold in subject_idxs:
            train_data_idx = np.setdiff1d(np.array(subject_idxs), np.array([fold]))
            train_data = data[train_data_idx,:]
            train_labels = [labels[t] for t in train_data_idx]

            val_data = data[[fold],:]

            clf = make_pipeline(LinearSVC(dual = False))

            clf.fit(train_data, train_labels)

            predictions.append(clf.predict(val_data))

        bsr = balanced_accuracy_score(predictions, labels)

        return bsr