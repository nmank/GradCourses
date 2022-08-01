# module imports
from locale import ABMON_1
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

from binarytree import Node

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


    def fit(self, X: ndarray= None, y: ndarray =None) -> None:
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

    def transform(self, X: ndarray = None, y: ndarray  = None, loo = False, fiedler = True) -> tuple:
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


        all_bsr = self.test_cut_loo(X, y)

        nodes = np.arange(len(self.A_))
        clst_nodes = []
        clst_bsrs = []
        clst_mean_edges = []
        root = Node(0)
        clst_tree = root
        new_root = root

        self.cluster_laplace_svm(self.A_, X, y, nodes, clst_nodes, clst_bsrs, 
                                clst_mean_edges, clst_tree, new_root, previous_bsr = all_bsr, 
                                fiedler_switch =fiedler, loo = loo)

        return clst_nodes, clst_bsrs, clst_mean_edges, clst_tree
 
    def cluster_laplace_svm(self, A: ndarray, data: ndarray, labels: list, nodes: ndarray, 
                            clst_nodes: list, clst_bsrs: list, clst_mean_edges: list, 
                            clst_tree: Node, new_root: Node, previous_bsr: float = 0, 
                            fiedler_switch: bool = True, loo: bool = False) -> None:

        #partition the data using the fiedler vector
        N1,N2 = gt.laplace_partition(A,fiedler_switch,1)

        #sizes of the clusters
        s1 = N1.size
        s2 = N2.size

        #clean N1, N2
        N1 = N1.T[0]
        N2 = N2.T[0]

        #init bsrs
        bsr1 = 0
        bsr2 = 0

        if s1 > 1:
            nodes1 = nodes[N1]
            data1 = data[:,N1]
            A1 = A[N1,:][:,N1]

            if loo:
                bsr1 = self.test_cut_loo(data1, labels)
            else:
                bsr1 = self.test_cut(data1, labels)

        if s2 > 1:
            nodes2 = nodes[N2]
            data2 = data[:,N2]
            A2 = A[N2,:][:,N2]
            if loo:
                bsr2 = self.test_cut_loo(data2, labels)
            else:
                bsr2 = self.test_cut(data2, labels)

        if (bsr1 < previous_bsr and bsr2 < previous_bsr) or (len(nodes) == 1):
            clst_nodes.append(np.array([int(node) for node in nodes]))
            clst_bsrs.append(previous_bsr)
            if len(clst_mean_edges) > 0:
                print(f'leaf {clst_mean_edges[-1]}')
        else:
            if bsr1 == bsr2:
                new_root = clst_tree
                clst_mean_edges.append(np.mean(A1[A1!=0]))
                clst_tree.left = Node(np.round(clst_mean_edges[-1],3))
                clst_tree = clst_tree.left
                self.cluster_laplace_svm(A1, 
                                    data1, 
                                    labels,
                                    nodes1, 
                                    clst_nodes,
                                    clst_bsrs,
                                    clst_mean_edges,
                                    clst_tree,
                                    new_root, 
                                    previous_bsr = bsr1, 
                                    fiedler_switch = True,
                                    loo = loo)

                clst_mean_edges.append(np.mean(A2[A2!=0]))
                clst_tree = new_root
                clst_tree.right = Node(np.round(clst_mean_edges[-1],3))
                clst_tree = clst_tree.right
                self.cluster_laplace_svm(A2, 
                                    data2, 
                                    labels, 
                                    nodes2, 
                                    clst_nodes,
                                    clst_bsrs,
                                    clst_mean_edges,
                                    clst_tree,
                                    new_root,  
                                    previous_bsr = bsr2, 
                                    fiedler_switch = True,
                                    loo = loo)
                

            elif bsr1 > bsr2:
                clst_mean_edges.append(np.mean(A1[A1!=0]))
                clst_tree.left = Node(np.round(clst_mean_edges[-1],3))
                clst_tree = clst_tree.left
                self.cluster_laplace_svm(A1, 
                                    data1, 
                                    labels, 
                                    nodes1, 
                                    clst_nodes,
                                    clst_bsrs,
                                    clst_mean_edges,
                                    clst_tree,
                                    new_root,  
                                    previous_bsr = bsr1, 
                                    fiedler_switch = True,
                                    loo = loo)
                
                

            elif bsr2 > bsr1:
                clst_mean_edges.append(np.mean(A2[A2!=0]))
                clst_tree.right = Node(np.round(clst_mean_edges[-1],3))
                clst_tree = clst_tree.right
                self.cluster_laplace_svm(A2, 
                                    data2, 
                                    labels, 
                                    nodes2, 
                                    clst_nodes,
                                    clst_bsrs,
                                    clst_mean_edges,
                                    clst_tree,
                                    new_root,  
                                    previous_bsr = bsr2, 
                                    fiedler_switch = True,
                                    loo = loo)
                
                

    def test_cut(self, data: ndarray, labels: list) -> float:
        '''
        Train and run an SVM classifier on the data and labels.

        Inputs:
            data (numpy array): the data where a datapoint in a row
            labels (numpy array or list) the labels of the rows of data
        
        Outputs:
            bsr (float): the BSR of the SVM classifier on the data and labels
        '''
        # clf = make_pipeline(LinearSVC(dual = False)) #old
        clf = make_pipeline(LinearSVC(max_iter = 20000)) #new

        clf.fit(data, labels)

        predictions = clf.predict(data)

        bsr = balanced_accuracy_score(predictions, labels) 

        return bsr

    def test_cut_loo(self, data: ndarray, labels: list) -> float:
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

                # clf = make_pipeline(LinearSVC(dual = False)) #old
                clf = make_pipeline(LinearSVC(max_iter = 20000)) #new

                clf.fit(train_data, train_labels)

                predictions.append(clf.predict(val_data))

            bsr = balanced_accuracy_score(predictions, labels)

            return bsr