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
import os



class GLPE(BaseEstimator):
    '''
    This is a class for Generalized Linear Pathway Expression. 
    Creates Pathway expression vectors from a pathway transition matrix and a dataset.

        pathway_transition_matrix (numpy array) pathway x features with weights for each 
                                                feature in a pathway
        X (numpy array) subject x features with the dataset.
    '''

    def __init__(self, pathway_transition_matrix: ndarray = None):
        # set params
        self.pathway_transition_matrix_ = np.array(pathway_transition_matrix)
        self.feature_names_ = None

    @property
    def pathway_transition_matrix(self):

        # check for sparsity
        sparsity = (self.pathway_transition_matrix_ == 0).sum() / self.pathway_transition_matrix_.size

        if sparsity >= .50:
            return csr_matrix(self.pathway_transition_matrix_)
        else:
            return self.pathway_transition_matrix_

    @property
    def feature_names(self):
        return self.feature_names_

    def fit(self, X=None, y=None):

        # nothing...

        return self

    def transform(self, X):
        '''
        Transforms a dataset using matrix product with pathway_transition_matrix

        Inputs: 
            X (numpy array) (subject x features) with the dataset.
        Outputs:
            X_transformed (numpy) (subject x pathways) pathway expression vectors
        '''
        #check X is fitted and X is the right type
        check_is_fitted(self)

        X = check_array(X)

        #restrict X to freature names using copy
        Y = X[:,self.feature_names.astype(int)]

        # pathway_transition_matrix is pathway x features
        # X is subject x features
        #output is subject x pathway
        
        # matrix multiplication
        X_transformed = self.pathway_transition_matrix.dot( Y.T ).T

        return X_transformed



'''
subclass for centrality pathway transition matrix using biological networks or generating networks (eg. correlation)
'''
class CLPE(GLPE):
    '''
    Creaetes pathway_transition_matrix using network centrality in the fit function
        Inputs:
            centrality_measure (string)
                                Measure of centrality for the network.
                                Options are: 'degree', 'page_rank', 'large_evec'
            network_type (string)
                                The type of network to be used.
                                Options are: 'precomputed', 'correlation', 'heatkernel'
            pathway_files: (string)
                                File path for .csv of matrices for each pathway 
                                
            heat_kernel_param: (float)
                                the heat for the heat kernel colculation if network type is heatkernel
            
    '''
    def __init__(self, 
                centrality_measure: str = None, 
                network_type: str = None,
                feature_ids: list = None,
                pathway_files: str = None,
                directed: bool = False,
                heat_kernel_param: float = 2,
                normalize_rows: bool = True):

         # set params
        self.centrality_measure_ = str(centrality_measure)
        self.network_type_ = str(network_type)
        self.feature_ids_ = list(feature_ids)
        self.pathway_files_ = str(pathway_files)
        self.directed_ = bool(directed)
        self.heat_kernel_param_ = float(heat_kernel_param)
        self.normalize_rows_ = float(normalize_rows)
        self.pathway_names_ = []
    
    @property
    def centrality_measure(self):
        return self.centrality_measure_

    @property
    def network_type(self):
        return self.network_type_
    
    @property
    def feature_ids(self):
        return self.feature_ids_

    @property
    def pathway_files(self):
        return self.pathway_files_

    @property
    def directed(self):
        return self.directed_

    @property
    def heat_kernel_param(self):
        return self.heat_kernel_param_

    @property
    def pathway_names(self):
        return self.pathway_names_

    @property
    def normalize_rows(self):
        return self.normalize_rows_


    def generate_adjacency_matrix(self, X = None, f = None, from_file = True):
        '''
        Generates a feature adjacency matrix.
        If network_type is precomputed then we uses the incidence matrix.
        Otherwise, use network_type to generate the adjacency matrix using X.


        Inputs:
            X (numpy array): A data matrix. (subject x features)
            f (string): The identifier for the pathway. In incidence_matrix.
        Outputs:
            A (numpy array): (features in pathway) x (features in pathway) adjacency matrix
            feature_idx (numpy array): the labels of the features in the adjacency matrix
        '''
        
        if from_file:
            x = pandas.read_csv(self.pathway_files_ + f, index_col = 0)

            feature_names = list(x.columns)
            feature_names = [e.partition("_")[2] for e in feature_names] #feature names after underscore

            restricted_feature_names = list(set(self.feature_ids_).intersection(set(feature_names)))
            idx = np.array([feature_names.index(i) for i in restricted_feature_names])

            if len(idx) > 0:

                if self.network_type_ == 'precomputed':
                    
                    A = np.array(x)
                    
                    A = A[:,idx][idx,:]

                else:

                    #data matrix for features in one pathway (subjects x features)
                    pathway_data =  X[:,idx]

                    #generate adjacency matrix
                    A = gt.adjacency_matrix(np.array(pathway_data), self.network_type_, h_k_param = self.heat_kernel_param_)
            else:
                A = None    

        else:
            feature_names = f

            restricted_feature_names = list(set(self.feature_ids_).intersection(set(feature_names)))
            idx = np.array([self.feature_ids_.index(i) for i in restricted_feature_names])

            if len(idx) > 0:
                pathway_data =  X[:,idx]

                #generate adjacency matrix
                A = gt.adjacency_matrix(np.array(pathway_data), self.network_type_, h_k_param = self.heat_kernel_param_)
            else:
                A = None

        return A, restricted_feature_names



    #overrides glpe method
    def fit(self, X: ndarray= None, y=None):
        '''
        Generates a pathway transition matrix using network centrality.

        Inputs:
            X (pandas DataFrame): a data matrix that is (subject x features)
        '''

        X = check_array(X)

        n_features = len(self.feature_ids_)

        self.pathway_transition_matrix_ = []

        if os.path.isfile(self.pathway_files_):

            #THIS NEEDS TO BE FIXED

            pathway_data = pandas.read_csv(self.pathway_files_, index_col = 'ReactomeID')
            pathway_data = pathway_data.fillna(0)
            
            self.pathway_names_ = []
            for pathway_name, row in pathway_data.iterrows():

                entries = row.values
                features_in_pathway = row.index[entries != 0] 

                A, feature_idx = self.generate_adjacency_matrix(X, features_in_pathway, from_file = False)

                #FIX HERE!

                row = list(scores)

                pathway_data[pathway_name] = row

                self.pathway_names_.append(pathway_name)
            
            self.pathway_transition_matrix_  = np.array(pathway_data)



        else:
            self.pathway_names_ = []
            #define pathway names
            for f in os.listdir(self.pathway_files_):

                start = f.find("R-HSA")
                end = f.find(".csv")

                pathway_name = f[start:end]
        
                #adjacency matrix
                A, feature_idx = self.generate_adjacency_matrix(X, f)

                score_row = np.zeros(n_features)

                if len(feature_idx) > 0:
                    #centrality scores
                    scores = gt.centrality_scores(A, self.centrality_measure_)

                    #normalize degree centrality by maximum degree
                    if self.centrality_measure_ == 'degree':
                        degrees = np.sum(A,axis = 0)
                        scores = scores / np.max(degrees)

                    #normalize centrality score by l1 norm
                    if self.normalize_rows:
                        scores = scores/np.sum(scores)

                    #add feature scores to row for pathway_transition matrix
                    idx = np.array([self.feature_ids_.index(i) for i in feature_idx])
                    score_row[idx] = scores

                #add to pathway_transition_matrix
                self.pathway_transition_matrix_.append(score_row)

                self.pathway_names_.append(pathway_name)
        
            self.pathway_transition_matrix_ = np.vstack(self.pathway_transition_matrix_)

        # replace na with 0
        np.nan_to_num(self.pathway_transition_matrix_, copy=False)

        return self

    def pathway_centrality_score(self, idxs = None):
        '''
        Generates a simple pathway centrality score

        Inputs:
            idxs (numpy array): the columns in X that are in the featureset
        Outputs:
            scores (numpy array): the centrality scores for each pathway
        '''
        scores = np.sum(self.pathway_transition_matrix_[:,idxs], axis = 1)
        return scores
    
    def simple_transform(self, featureset_transition_matrix_ids = None, n_null_trials = 10):
        '''
        Generates simple centrality scores for each pathway given a featureset (featureset_transition_matrix_ids)
        along with a p-value of each pathway (low p is good)

        Inputs:
            featureset_transition_matrix_ids (numpy array): the columns in X that are in the featureset
            n_null_trials (int): the number of null trials to generate the p value
        Outputs:
            scores_and_p (pandas.DataFrame): the centrality scores for each pathway along with their respective p values
        '''
        #calc centrality scores
        scores = self.pathway_centrality_score(featureset_transition_matrix_ids)

        null_scores = []
        for seed in range(n_null_trials):
            np.random.seed(seed)
            null_featureset_ids = np.random.choice(self.feature_names_, len(featureset_transition_matrix_ids), replace = False)
            null_featureset_transition_matrix_ids = np.nonzero(np.in1d(self.feature_names_,np.array(null_featureset_ids)))[0]
            null_scores.append(self.pathway_centrality_score(null_featureset_transition_matrix_ids))
            if seed %10 == 0:
                print('null trial '+str(seed)+' done')

        null_array = np.vstack(null_scores)

        bigger_genes = np.zeros(null_array.shape)
        for i in range(n_null_trials):
            idx = np.where(scores < null_array[i,:])
            bigger_genes[i, idx] = 1
            
        p_val = np.mean(bigger_genes, axis = 0)

        scores_and_p = pandas.DataFrame(columns = ['pathway','score', 'p_val'])
        scores_and_p['pathway'] = self.pathway_names_
        scores_and_p['score'] = scores
        scores_and_p['p_val'] = p_val
        

        return scores_and_p


    





    




# a function for simple centrality pathway ranking using features can use the subclass



# def make_network(pathway_name, all_edge_dataframe, undirected):
#     '''
#     Make a network from the known edges.

#     Inputs:
#         pathway_name: a string for the name of a pathway
#         all_edge_dataframe: a dataframe with all the edges

#     Outputs:
#         A: a numpy array of the adjacency matrix (directed)
#         node_eids: a list of EntrezIDs whose indices correspond to the entries of A
#     '''

#     edge_dataframe = all_edge_dataframe[all_edge_dataframe['pathway_id'] == pathway_name]

#     node_eids = np.array(list(set(edge_dataframe['src']).union(set(edge_dataframe['dest']))))

#     n_nodes = len(node_eids)

#     A = np.zeros((n_nodes, n_nodes))

#     for _, row in edge_dataframe.iterrows():
#         i = np.where(node_eids == row['src'])[0][0]
#         j = np.where(node_eids == row['dest'])[0][0]
#         A[i, j] = 1
#         if undirected or row['direction'] == 'undirected':
#             A[j, i] = A[i, j].copy()

#     return A, node_eids

# def calc_pathway_scores(centrality_measure, undirected, pid_2_eid, pathway_edges, featureset):
#     # load names of the pathways and init pathway dataframe
#     pathway_names = np.unique(np.array(pathway_edges['pathway_id']))

#     pathway_scores = pandas.DataFrame(
#         columns=['pathway_id', 'unnormalized', 'path norm', 'feature path norm', 'avg degree norm',
#                  'feature path count'])

#     lengths = []

#     scores_list = []

#     ii = 0
#     # go through every pathway name
#     for pathway_name in pathway_names:

#         # make adjacency matrix
#         A, n_eids = make_network(pathway_name, pathway_edges, undirected)

#         # node eids as strings
#         string_node_eids = [str(int(node)) for node in n_eids]

#         ###########################
#         # ToDo
#         # write a helper function that does this conversion outside of this function that gives consistent indexing
#         # import featureset with index
#         # rename index based on pid to eid dictionary
#         # comment functions

#         # get featureset eids
#         featureset_pids = list(featureset['Unnamed: 0'])

#         featureset_eids = []
#         # load eids from the probeids in the featureset
#         for p in featureset_pids:
#             if p in list(pid_2_eid['ProbeID']):
#                 featureset_eids.append(str(pid_2_eid[pid_2_eid['ProbeID'] == p]['EntrezID'].item()))

#         ############################

#         # find the featureset nodes in the pathway
#         discriminatory_nodes = list(set(featureset_eids).intersection(set(string_node_eids)))

#         # calculate pathway scores
#         scores = gt.centrality_scores(A, centrality_measure)

#         # average degree
#         degrees = np.sum(A, axis=0)
#         avg_degree = np.mean(degrees)

#         # find the indices of the nodes in the adjacency matrix that correspond to nodes in the featureset
#         idx = [string_node_eids.index(r) for r in discriminatory_nodes]

#         # calculate pathway scores
#         node_scores = scores[idx]

#         if len(node_scores) > 0:
#             pathway_score = np.sum(node_scores)

#             # pathway_score = np.sum(node_scores)

#             pathway_scores = pathway_scores.append({'pathway_id': pathway_name,
#                                                     'unnormalized': pathway_score,
#                                                     'path norm': pathway_score / len(scores),
#                                                     'feature path norm': pathway_score / len(node_scores),
#                                                     'avg degree norm': pathway_score / avg_degree,
#                                                     'feature path count': len(node_scores)},
#                                                    ignore_index=True)

#             scores_list.append(pathway_score)
#             lengths.append(len(scores))

#         if ii % 200 == 0:
#             print('pathway ' + str(ii) + ' done')

#         ii += 1

#     plt.figure()
#     plt.scatter(lengths, scores_list)
#     plt.xlabel('Pathway Size')
#     plt.ylabel('Centrality Score')

#     pathway_scores = pathway_scores.sort_values(by='unnormalized', ascending=False).dropna()
#     # change location of saved csvs to be parameter
#     if undirected:
#         pathway_scores.to_csv(
#             '/data4/mankovic/GSE73072/network_centrality/undirected/gse73072_undirected_' + centrality_measure + '_pval_and_lfc.csv',
#             index=False)
#     else:
#         pathway_scores.to_csv(
#             '/data4/mankovic/GSE73072/network_centrality/directed/gse73072_directed_' + centrality_measure + '_pval_and_lfc.csv',
#             index=False)
