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



class GLPE(BaseEstimator):

    def __init__(self, pathway_transition_matrix: ndarray = None):
        # set params
        self.pathway_transition_matrix_ = np.array(pathway_transition_matrix)

    @property
    def pathway_transition_matrix(self):

        # check for sparsity
        sparsity = (self.pathway_transition_matrix_ == 0).sum() / self.pathway_transition_matrix_.size

        if sparsity >= .50:
            return csr_matrix(self.pathway_transition_matrix_)
        else:
            return self.pathway_transition_matrix_

    def fit(self, X=None, y=None):

        # nothing...

        return self

    def transform(self, X):

        # code

        
        #check X is fitted and X is the right type
        check_is_fitted(self)

        X = check_array(X)

        # pathway_transition_matrix is pathway x features
        # X is subject x features
        #output is subject x pathway
        
        # matrix multiplication
        X_transformed = self.pathway_transition_matrix.dot( X.T ).T

        return X_transformed



'''
subclass for centrality pathway transition matrix using biological networks or generating networks (eg. correlation)
'''
class CLPE(GLPE):
    '''
        incidence matrix: (numpy array)
                            if netwok_type is precomputed it must have the following columns:
                                column 0 is the source index which corresponds to row of X
                                column 1 is the destination index which corresponds to row of X
                                column 2 is a string 'directed' for a directed edge and 'undireced' for an undirected edge
                                column 3 is the pathway_id
                            if network type is not precomputed is must have the follwoing columns:
                                column 0 corresponds to row of X
                                column 1 is the pathway_id
            
    '''
    def __init__(self, 
                centrality_measure: str = None, 
                network_type: str = None,
                pathway_names: list = None,
                incidence_matrix: ndarray = None,
                heat_kernel_param: float = None):

         # set params
        self.centrality_measure_ = str(centrality_measure)
        self.network_type_ = str(network_type)
        self.pathway_names_ = list(pathway_names)
        self.incidence_matrix_ = np.array(incidence_matrix)
        self.heat_kernel_param_ = float(heat_kernel_param)   

    def generate_adjacency_matrix(self, X = None, pathway_name = None):
        
        if self.network_type_ == 'precomputed':

            #get incidence matrix for this pathway
            edge_idx = np.where(self.incidence_matrix_[:,3] == pathway_name)[0]
            pathway_incidence = self.incidence_matrix_[edge_idx,:3]
            feature_idx = np.unique(pathway_incidence[:,:2])

            #features in this pathway
            nodes = np.unique(pathway_incidence[:,:2])
            n_nodes = len(nodes)

            A = np.zeros((n_nodes, n_nodes))

            for row in pathway_incidence:
                i = np.where(feature_idx == row[0])[0][0]
                j = np.where(feature_idx == row[1])[0][0]
                A[i, j] = 1
                if row[2] == 'undirected':
                    A[j, i] = A[i, j].copy()
        else:
            #feature_ids in the pathway
            idx = np.where(self.incidence_matrix_[:,1] == pathway_name)
            feature_idx = self.incidence_matrix_[idx,0]

            #data matrix for features in one pathway (subjects x features)
            pathway_data =  X[feature_idx,:].T

            #generate adjacency matrix
            A = gt.adjacency_matrix(pathway_data, self.network_type_, h_k_param = self.heat_kernel_param_)

        return A, feature_idx



    #overrides glpe method
    def fit(self, X: ndarray= None, y=None):
        '''
        X is an array with features as rows and columns as subjects

        in fit:
            args:  centrality measure, generation type, pre-computed incedence matrix

            functions:
                generate a network for each pathway
                calculate node centrality values

            use these to generate the pathway transition matrix
        '''

        if self.network_type_ == 'precomputed':
            feature_names = np.unique(self.incidence_matrix_[:1])
        else:
            feature_names = np.unique(self.incidence_matrix_[:2])

        n_features = len(feature_names)

        self.pathway_transition_matrix_ = []

        #define pathway names
        for pathway_name in self.pathway_names_:
    
            #adjacency matrix
            A, feature_idx = self.generate_adjacency_matrix(X, pathway_name)

            #centrality scores
            scores = gt.centrality_scores(A, self.centrality_measure_)

            #normalize centrality score by l1 norm
            scores = scores/np.sum(scores)

            #add feature scores to row for pathway_transition matrix
            score_row = np.zeros(n_features)
            score_row[feature_idx] = scores

            #add to pathway_transition_matrix
            self.pathway_transition_matrix_.append(score_row)
        
        self.pathway_transition_matrix = np.vstack(self.pathway_transition_matrix)

        return self



    





    




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
