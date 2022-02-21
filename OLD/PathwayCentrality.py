import pandas
import numpy as np
from numpy import ndarray
from scipy.sparse import csr_matrix
from scipy import *
from matplotlib import pyplot as plt
import graph_tools_construction as gt
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted



'''
subclass for centrality pathway transition matrix using biological networks or generating networks (eg. correlation)
'''
class PathwayCentrality(BaseEstimator):
    '''
    Creaetes pathway_transition_matrix using network centrality in the fit function
        Inputs:
            centrality_measure (string)
                                Measure of centrality for the network.
                                Options are: 'degree', 'page_rank', 'large_evec'
            network_type (string)
                                The type of network to be used.
                                Options are: 'precomputed', 'correlation', 'heatkernel'
            incidence matrix: (numpy array)
                                if netwok_type is precomputed it must have the following columns:
                                    column 0 is the source index which corresponds to column of X
                                    column 1 is the destination index which corresponds to column of X
                                    column 2 is a string 'directed' for a directed edge and 'undireced' for an undirected edge
                                    column 3 is the pathway_id
                                if network type is not precomputed is must have the following columns:
                                    column 0 corresponds to column of X
                                    column 1 is the pathway_id
            heat_kernel_param: (float)
                                the heat for the heat kernel colculation if network type is heatkernel
            
    '''
    def __init__(self, 
                centrality_measure: str = None, 
                network_type: str = None,
                featureset: ndarray = None,
                incidence_matrix: ndarray = None,
                heat_kernel_param: float = 2):

         # set params
        self.centrality_measure_ = str(centrality_measure)
        self.network_type_ = str(network_type)
        self.incidence_matrix_ = np.array(incidence_matrix)
        self.heat_kernel_param_ = float(heat_kernel_param)
        self.featureset_ = np.array(featureset)
        self.pathway_names_ = []
        self.feature_names_ = []
    
    @property
    def centrality_measure(self):
        return self.centrality_measure_

    @property
    def network_type(self):
        return self.network_type_

    @property
    def incidence_matrix(self):
        return self.incidence_matrix_

    @property
    def heat_kernel_param(self):
        return self.heat_kernel_param_

    @property
    def pathway_names(self):
        return self.pathway_names_

    @property
    def feature_names(self):
        return self.feature_names_


    @property
    def featureset(self):
        return self.featureset_


    def generate_adjacency_matrix(self, X = None, pathway_name = None):
        '''
        Generates a feature adjacency matrix.
        If network_type is precomputed then we uses the incidence matrix.
        Otherwise, use network_type to generate the adjacency matrix using X.


        Inputs:
            X (numpy array): A data matrix. (subject x features)
            pathway_name (string): The identifier for the pathway. In incidence_matrix.
        Outputs:
            A (numpy array): (features in pathway) x (features in pathway) adjacency matrix
            feature_idx (numpy array): the index in X of the features in the adjacency matrix
        '''
        
        if self.network_type_ == 'precomputed':

            #get incidence matrix for this pathway
            edge_idx = np.where(self.incidence_matrix_[:,3] == pathway_name)[0]
            pathway_incidence = self.incidence_matrix_[edge_idx,:3]
            
            #features in this pathway
            feature_idx = np.unique(pathway_incidence[:,:2])
            n_nodes = len(feature_idx)

            A = np.zeros((n_nodes, n_nodes))

            for row in pathway_incidence:
                i = np.where(feature_idx == row[0])[0][0]
                j = np.where(feature_idx == row[1])[0][0]
                A[i, j] = 1
                if row[2] == 'undirected':
                    A[j, i] = A[i, j].copy()
            feature_idx = np.array(feature_idx).astype(int)

        else:
            #feature_ids in the pathway
            idx = np.where(self.incidence_matrix_[:,1] == pathway_name)[0]
            feature_idx = self.incidence_matrix_[idx,0]

            feature_idx = np.array(feature_idx).astype(int)
            
            #data matrix for features in one pathway (subjects x features)
            pathway_data =  X[:,feature_idx]

            #generate adjacency matrix
            A = gt.adjacency_matrix(pathway_data, self.network_type_, h_k_param = self.heat_kernel_param_)

        return A, feature_idx

    
    def calc_centralities(self, X = None, pathway_name = None):
        #adjacency matrix
        A, feature_idx = self.generate_adjacency_matrix(X, pathway_name)

        #centrality scores
        scores = gt.centrality_scores(A, self.centrality_measure_)

        #normalize degree centrality by maximum degree
        if self.centrality_measure_ == 'degree':
            degrees = np.sum(A,axis = 0)
            scores = scores / np.max(degrees)
    
        return scores, feature_idx


    def fit_transform(self, X: ndarray= None):
        '''
        Generates a pathway transition matrix using network centrality.

        Inputs:
            X (numpy array): a data matrix that is (subject x features)
        '''

        X = check_array(X)

        if self.network_type_ == 'precomputed':
            self.feature_names_ = np.unique(self.incidence_matrix_[:, :2])
            self.pathway_names_ = np.unique(self.incidence_matrix_[:, 3])
        else:
            self.feature_names_ = np.unique(self.incidence_matrix_[:, :1])
            self.pathway_names_ = np.unique(self.incidence_matrix_[:,1])
            



        self.all_scores_ = []

        #define pathway names
        for pathway_name in self.pathway_names_:
    
            scores, feature_idx = self.calc_centralities(X, pathway_name)

            # one_pathway_features_idx = np.where(self.pathway_features[1:,0] == pathway_name)
            # one_pathway_features = self.pathway_features[1,one_pathway_features_idx]

            featureset_and_pathway_idx = np.nonzero(np.in1d(feature_idx,self.featureset))[0]

            self.all_scores_.append(np.sum(scores[featureset_and_pathway_idx]))

        self.all_scores_ = np.array(self.all_scores_)

        # replace na with 0
        np.nan_to_num(self.all_scores_, copy=False)

        return self

