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
        # self.feature_names_ = None

    @property
    def pathway_transition_matrix(self):

        # check for sparsity
        sparsity = (self.pathway_transition_matrix_ == 0).sum() / self.pathway_transition_matrix_.size

        if sparsity >= .50:
            return csr_matrix(self.pathway_transition_matrix_)
        else:
            return self.pathway_transition_matrix_

    # @property
    # def feature_names(self):
    #     return self.feature_names_

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
        # Y = X[:,self.feature_names.astype(int)]

        # pathway_transition_matrix is pathway x features
        # X is subject x features
        #output is subject x pathway
        
        # matrix multiplication
        X_transformed = self.pathway_transition_matrix.dot( X.T ).T

        return X_transformed



'''
subclass for centrality pathway transition matrix for simple linear pathway expression
'''
class LPE(GLPE):
    def __init__(self, 
                pathway_transition_matrix: np.array = None,
                feature_ids: list = None,
                pathway_files: str = None,
                normalize_rows: bool = True):
        # set params
        self.feature_ids_ = list(feature_ids)
        self.pathway_files_ = str(pathway_files)
        self.normalize_rows_ = bool(normalize_rows)
        self.pathway_names_ = []

        super().__init__(pathway_transition_matrix)
    
    @property
    def feature_ids(self):
        return self.feature_ids_

    @property
    def pathway_files(self):
        return self.pathway_files_

    @property
    def pathway_names(self):
        return self.pathway_names_

    @property
    def normalize_rows(self):
        return self.normalize_rows_
       
    @property
    def pathway_names(self):
        
        # check if precomputed
        if  len(self.pathway_names_) > 0:
            return self.pathway_names_

        # calculate from files
        self.pathway_names_ = []
        if os.path.isfile(self.pathway_files_):
            #if the pathway data is in one file
            #then it should be indexed by first column (pathway names)
            #rest of columns should be feature names for the pathway
            #1 if feature is in the pathway
            #0 otherwise

            pathway_data = pandas.read_csv(self.pathway_files_, index_col = 0)
            pathway_data = pathway_data.fillna(0)
        
            for pathway_name, _ in pathway_data.iterrows():

                #keep track of pathway names
                self.pathway_names_.append(pathway_name)

        elif os.path.isdir(self.pathway_files_):
            #define pathway names
            for f in os.listdir(self.pathway_files_):

                # start = f.find("R-HSA")
                # end = f.find(".csv")

                # pathway_name = f[start:end]
                pathway_name = f[7:-4]

                #keep track of pathway names
                self.pathway_names_.append(pathway_name)
        
        else:
            print('pathway_files must be a file or directory path')
        
        return self.pathway_names_

    def fit(self, X: ndarray= None, y = None):

        def restrict_feat_names(feature_names):
            #restrict pathway feature names to those in the dataset (X)
            restricted_feature_names = list(set(self.feature_ids_).intersection(set(feature_names)))
            return restricted_feature_names


        if os.path.isdir(self.pathway_files_):

            
            self.pathway_transition_matrix_ = []

            self.pathway_names_ = []

            #define pathway names
            for f in os.listdir(self.pathway_files_):

                # start = f.find("R-HSA")
                # end = f.find(".csv")

                # pathway_name = f[start:end]
                pathway_name = f[7:-4]

                #read the csv and take the feature ids of the pathway to be the part of the string after the '_'
                x = pandas.read_csv(self.pathway_files_ +'/'+ f, index_col = 0)
                x = x.fillna(0)

                feature_names = list(x.columns)
                if len(feature_names) > 0:
                    if 'entrez' in feature_names[0]:
                        feature_names = [e.partition("_")[2] for e in feature_names] #feature names after underscore

                restricted_feature_names = restrict_feat_names(feature_names)

                row = np.isin(self.feature_ids_, restricted_feature_names).astype(int)
                if self.normalize_rows_:
                    row = row/np.sum(row)

                #add to pathway_transition_matrix
                self.pathway_transition_matrix_.append(row)

                #keep track of pathway names
                self.pathway_names_.append(pathway_name)


            self.pathway_transition_matrix_ = np.array(self.pathway_transition_matrix_)

            np.nan_to_num(self.pathway_transition_matrix_, copy=False)
                
        else:
            print('fit did not run- pathway_files is not a directory')
        
        return self





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
            feature_ids (list)
                                A list of strings for gene (or other feature) ids that 
                                correspond to rows of the data matrixX
            pathway_files: (string)
                                File path for .csv of matrices for each pathway 
            directed (bool)
                                Whether or not we will use a precomputed 
                                directed or undirected feature (gene) graph
            heat_kernel_param: (float)
                                the heat for the heat kernel colculation if network type is heatkernel
            normalize rows (bool)
                                Whether or not to normalize the rows of the pathway expression matrix
            
    '''
    def __init__(self, 
                pathway_transition_matrix: np.array = None,
                centrality_measure: str = None, 
                network_type: str = None,
                feature_ids: list = None,
                pathway_files: str = None,
                directed: bool = None,
                heat_kernel_param: float = 2,
                normalize_rows: bool = True):

         # set params
        self.centrality_measure_ = str(centrality_measure)
        self.network_type_ = str(network_type)
        self.feature_ids_ = list(feature_ids)
        self.pathway_files_ = str(pathway_files)
        self.directed_ = bool(directed)
        self.heat_kernel_param_ = float(heat_kernel_param)
        self.normalize_rows_ = bool(normalize_rows)
        self.pathway_names_ = []

        super().__init__(pathway_transition_matrix)
    
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
    
    @property
    def pathway_names(self):
        
        # check if precomputed
        if  len(self.pathway_names_) > 0:
            return self.pathway_names_

        # calculate from files
        self.pathway_names_ = []
        if os.path.isfile(self.pathway_files_):
            #if the pathway data is in one file
            #then it should be indexed by first column (pathway names)
            #rest of columns should be feature names for the pathway
            #1 if feature is in the pathway
            #0 otherwise

            pathway_data = pandas.read_csv(self.pathway_files_, index_col = 0)
            pathway_data = pathway_data.fillna(0)
        
            for pathway_name, _ in pathway_data.iterrows():

                #keep track of pathway names
                self.pathway_names_.append(pathway_name)

        elif os.path.isdir(self.pathway_files_):
            #define pathway names
            for f in os.listdir(self.pathway_files_):

                # start = f.find("R-HSA")
                # end = f.find(".csv")

                # pathway_name = f[start:end]
                pathway_name = f[7:-4]

                #keep track of pathway names
                self.pathway_names_.append(pathway_name)
        
        else:
            print('pathway_files must be a file or directory path')
        
        return self.pathway_names_


    def generate_adjacency_matrix(self, X = None, f = None):
        '''
        Generates a feature adjacency matrix.
        If network_type is precomputed then we uses the incidence matrix.
        Otherwise, use network_type to generate the adjacency matrix using X.


        Inputs:
            X (numpy array): A data matrix. (subject x features)
            f (string or ): Either 1) A file path for precomputed pathway matrix with row and column labels 
                                    as feature names. 1 for directed edge and 2 for undirected edge.
                                   2) A list of feature names in the pathway.
        Outputs:
            A (numpy array): (features in pathway) x (features in pathway) adjacency matrix
            restricted_feature_names (numpy array): the labels of the features in the adjacency matrix
        '''

        def restrict_feat_names(feature_names):
            #restrict pathway feature names to those in the dataset (X)
            restricted_feature_names = list(set(self.feature_ids_).intersection(set(feature_names)))
            restricted_idx = np.array([feature_names.index(i) for i in restricted_feature_names])
            return restricted_feature_names, restricted_idx
        
        #if there's a file for each pathway
        if isinstance(f, str):

            #read the csv and take the feature ids of the pathway to be the part of the string after the '_'
            x = pandas.read_csv(self.pathway_files_ +'/'+ f, index_col = 0)
            x = x.fillna(0)

            feature_names = list(x.columns)
            if len(feature_names) > 0:
                if 'entrez' in feature_names[0]:
                    feature_names = [e.partition("_")[2] for e in feature_names] #feature names after underscore


            #restrict the features in the pathway to the features in the dataset
            restricted_feature_names, restricted_idx = restrict_feat_names(feature_names)

            #verify that there is at least one feature in the pathway and the dataset
            if len(restricted_idx) > 0:

                if self.network_type_ == 'precomputed':
                    
                    #load the adjacency matrix
                    A = np.array(x)
                    
                    #restrict adjacency matrix to restricted_feature_names
                    A = A[:,restricted_idx][restricted_idx,:]

                    #undirected edges are 2 and directed edges are 1
                    #make the undirected edges into 1s in the adjacency matrix
                    undirected_idx = np.where(A == 2)
                    A[undirected_idx] = 1
                    A[undirected_idx[1],undirected_idx[0]] = 1

                    #make sure the network is undirected if needed
                    if not self.directed_:
                        A[A != A.T] = 0

                else:

                    #data matrix for features in one pathway (subjects x features)
                    pathway_data =  X[:,restricted_idx]

                    #generate adjacency matrix
                    A = gt.adjacency_matrix(np.array(pathway_data), self.network_type_, h_k_param = self.heat_kernel_param_)
            else:
                A = None    

        else:
            #feature names are f
            feature_names = f
            
            #restrict features in pathway to those in the dataset
            restricted_feature_names, restricted_idx = restrict_feat_names(feature_names)

            if len(restricted_idx) > 0:
                #restrict the pathway data
                pathway_data =  X[:,restricted_idx]

                #generate adjacency matrix
                A = gt.adjacency_matrix(np.array(pathway_data), self.network_type_, h_k_param = self.heat_kernel_param_)
            else:
                A = None

        return A, restricted_feature_names


    def score_the_row(self, A, n_features_X, feature_idx):
        '''
        A function that calculates the score of a row.

        Inputs:
            A (numpy array) adjacency matrix
            n_features_X (int) number of features in X
            feature_idx (list) the list of ids of the features in the pathway
        Outputs:
            score_row (numpy array) the pathway score of all the features in 
                                    X based on their centrality in the pathway
        '''
        score_row = np.zeros(n_features_X)

        if len(feature_idx) > 0:
            #centrality scores
            scores = gt.centrality_scores(A, self.centrality_measure_, in_rank = True) #using True in_rank means A and False in_rank means A.T

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

        return score_row


    #overrides glpe method
    def fit(self, X: ndarray= None, y=None):
        '''
        Generates a pathway transition matrix using network centrality.

        Inputs:
            X (pandas DataFrame): a data matrix that is (subject x features)
        '''

        #verify X is a numpy array
        X = check_array(X)

        #number of features in X
        n_features = len(self.feature_ids_)

        #pathway transition matrix initialization
        self.pathway_transition_matrix_ = []


        if os.path.isfile(self.pathway_files_):
            #if the pathway data is in one file
            #then it should be indexed by first column (pathway names)
            #rest of columns should be feature names for the pathway
            #1 if feature is in the pathway
            #0 otherwise

            pathway_data = pandas.read_csv(self.pathway_files_, index_col = 0)
            pathway_data = pathway_data.fillna(0)
            
            self.pathway_names_ = []
            for pathway_name, row in pathway_data.iterrows():

                #features in the pathway
                entries = row.values
                features_in_pathway = list(row.index[entries != 0])

                #generate feature matrix
                A, feature_idx = self.generate_adjacency_matrix(X, features_in_pathway)

                #calculate the score of the row
                score_row = self.score_the_row(A, len(self.feature_ids_), feature_idx)

                #add score_row to pathway_data
                self.pathway_transition_matrix_.append(score_row)

                #keep track of pathway names
                self.pathway_names_.append(pathway_name)



        elif os.path.isdir(self.pathway_files_):
            self.pathway_names_ = []
            #define pathway names
            for f in os.listdir(self.pathway_files_):

                # start = f.find("R-HSA")
                # end = f.find(".csv")

                # pathway_name = f[start:end]
                pathway_name = f[7:-4]


                #adjacency matrix
                A, feature_idx = self.generate_adjacency_matrix(X, f)

                #calculate the score of the row
                score_row = self.score_the_row(A, n_features, feature_idx)

                #add to pathway_transition_matrix
                self.pathway_transition_matrix_.append(score_row)

                #keep track of pathway names
                self.pathway_names_.append(pathway_name)
        
        
        else:
            print('pathway_files must be a file or directory path')
        
        #make pathway transition matrix
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
    
    def simple_transform(self, featureset_names = None, n_null_trials = 10):
        '''
        Generates simple centrality scores for each pathway given a featureset (featureset_transition_matrix_ids)
        along with a p-value of each pathway (low p is good)

        Inputs:
            featureset_names (numpy array): names of the features in the featureset
            n_null_trials (int): the number of null trials to generate the p value
        Outputs:
            scores_and_p (pandas.DataFrame): the centrality scores for each pathway along with their respective p values
        '''
        #calc centrality scores
        featureset_transition_idx = [self.feature_ids_.index(i) for i in featureset_names]

        scores = self.pathway_centrality_score(featureset_transition_idx)

        null_scores = []
        for seed in range(n_null_trials):
            np.random.seed(seed)
            null_featureset_ids = np.random.choice(self.feature_ids_, len(featureset_names), replace = False)
            null_featureset_transition_idx = [self.feature_ids_.index(i) for i in null_featureset_ids]
            null_scores.append(self.pathway_centrality_score(null_featureset_transition_idx))
            # if seed %10 == 0:
            #     print('null trial '+str(seed)+' done')

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
