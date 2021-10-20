

# module imports
import pandas
import numpy as np
from numpy import ndarray
from scipy.sparse import csr_matrix
from scipy import
from matplotlib import pyplot as plt
import graph_tools_construction as gt
from sklearn.base import BaseEstimator

class GLPE(BaseEstimator):

    def __init__(self,
                 pathway_transition_matrix: ndarray = None,
                 ) -> None:

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

        # code

        return self

    def transform(self, X):

        # code

        return X_transformed


def make_network(pathway_name, all_edge_dataframe, undirected):
    '''
    Make a network from the known edges.

    Inputs:
        pathway_name: a string for the name of a pathway
        all_edge_dataframe: a dataframe with all the edges

    Outputs:
        A: a numpy array of the adjacency matrix (directed)
        node_eids: a list of EntrezIDs whose indices correspond to the entries of A
    '''

    edge_dataframe = all_edge_dataframe[all_edge_dataframe['pathway_id'] == pathway_name]

    node_eids = np.array(list(set(edge_dataframe['src']).union(set(edge_dataframe['dest']))))

    n_nodes = len(node_eids)

    A = np.zeros((n_nodes, n_nodes))

    for _, row in edge_dataframe.iterrows():
        i = np.where(node_eids == row['src'])[0][0]
        j = np.where(node_eids == row['dest'])[0][0]
        A[i, j] = 1
        if undirected or row['direction'] == 'undirected':
            A[j, i] = A[i, j].copy()

    return A, node_eids

def calc_pathway_scores(centrality_measure, undirected, pid_2_eid, pathway_edges, featureset):
    # load names of the pathways and init pathway dataframe
    pathway_names = np.unique(np.array(pathway_edges['pathway_id']))

    pathway_scores = pandas.DataFrame(
        columns=['pathway_id', 'unnormalized', 'path norm', 'feature path norm', 'avg degree norm',
                 'feature path count'])

    lengths = []

    scores_list = []

    ii = 0
    # go through every pathway name
    for pathway_name in pathway_names:

        # make adjacency matrix
        A, n_eids = make_network(pathway_name, pathway_edges, undirected)

        # node eids as strings
        string_node_eids = [str(int(node)) for node in n_eids]

        ###########################
        # ToDo
        # write a helper function that does this conversion outside of this function that gives consistent indexing
        # import featureset with index
        # rename index based on pid to eid dictionary
        # comment functions

        # get featureset eids
        featureset_pids = list(featureset['Unnamed: 0'])

        featureset_eids = []
        # load eids from the probeids in the featureset
        for p in featureset_pids:
            if p in list(pid_2_eid['ProbeID']):
                featureset_eids.append(str(pid_2_eid[pid_2_eid['ProbeID'] == p]['EntrezID'].item()))

        ############################

        # find the featureset nodes in the pathway
        discriminatory_nodes = list(set(featureset_eids).intersection(set(string_node_eids)))

        # calculate pathway scores
        scores = gt.centrality_scores(A, centrality_measure)

        # average degree
        degrees = np.sum(A, axis=0)
        avg_degree = np.mean(degrees)

        # find the indices of the nodes in the adjacency matrix that correspond to nodes in the featureset
        idx = [string_node_eids.index(r) for r in discriminatory_nodes]

        # calculate pathway scores
        node_scores = scores[idx]

        if len(node_scores) > 0:
            pathway_score = np.sum(node_scores)

            # pathway_score = np.sum(node_scores)

            pathway_scores = pathway_scores.append({'pathway_id': pathway_name,
                                                    'unnormalized': pathway_score,
                                                    'path norm': pathway_score / len(scores),
                                                    'feature path norm': pathway_score / len(node_scores),
                                                    'avg degree norm': pathway_score / avg_degree,
                                                    'feature path count': len(node_scores)},
                                                   ignore_index=True)

            scores_list.append(pathway_score)
            lengths.append(len(scores))

        if ii % 200 == 0:
            print('pathway ' + str(ii) + ' done')

        ii += 1

    plt.figure()
    plt.scatter(lengths, scores_list)
    plt.xlabel('Pathway Size')
    plt.ylabel('Centrality Score')

    pathway_scores = pathway_scores.sort_values(by='unnormalized', ascending=False).dropna()
    # change location of saved csvs to be parameter
    if undirected:
        pathway_scores.to_csv(
            '/data4/mankovic/GSE73072/network_centrality/undirected/gse73072_undirected_' + centrality_measure + '_pval_and_lfc.csv',
            index=False)
    else:
        pathway_scores.to_csv(
            '/data4/mankovic/GSE73072/network_centrality/directed/gse73072_directed_' + centrality_measure + '_pval_and_lfc.csv',
            index=False)
