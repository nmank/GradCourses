import pandas
import numpy as np
from ast import literal_eval
from matplotlib import pyplot as plt
import graph_tools_construction as gt

#load the data

# metadata = pandas.read_csv('/data4/kehoe/GSE73072/GSE73072_metadata.csv')
# vardata = pandas.read_csv('/data4/kehoe/GSE73072/GSE73072_vardata.csv')

pathway_edges = pandas.read_csv('/data3/darpa/omics_databases/ensembl2pathway/reactome_human_pathway_edges.csv').dropna()

featureset = pandas.read_csv('/data4/mankovic/GSE73072/network_centrality/featuresets/diffgenes_gse73072_pval_and_lfc.csv')

pid_2_eid = pandas.read_csv('/data4/mankovic/GSE73072/probe_2_entrez.csv')




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

    for _,row in edge_dataframe.iterrows():
        i = np.where(node_eids == row['src'])[0][0]
        j = np.where(node_eids == row['dest'])[0][0]
        A[i,j] = 1
        if undirected or row['direction'] == 'undirected':
            A[j,i] = A[i,j].copy()
    

    return A, node_eids

def calc_pathway_scores(centrality_measure, undirected, pid_2_eid, pathway_edges, featureset):
    # load names of the pathways and init pathway dataframe
    pathway_names = np.unique(np.array(pathway_edges['pathway_id']))

    pathway_scores = pandas.DataFrame(columns = ['pathway_id', 'unnormalized', 'path norm', 'feature path norm', 'avg degree norm', 'feature path count'])

    lengths = []

    scores_list = []

    ii=0
    # go through every pathway name
    for pathway_name in pathway_names:

        #make adjacency matrix
        A, n_eids = make_network(pathway_name, pathway_edges, undirected)

        #node eids as strings
        string_node_eids = [str(int(node)) for node in n_eids]


        ###########################
        #ToDo
        #write a helper function that does this conversion outside of this function that gives consistent indexing
        #import featureset with index
        #rename index based on pid to eid dictionary
        #comment functions

        #get featureset eids
        featureset_pids = list(featureset['Unnamed: 0'])

        featureset_eids = []
        #load eids from the probeids in the featureset
        for p in featureset_pids:
            if p in list(pid_2_eid['ProbeID']):
                featureset_eids.append(str(pid_2_eid[pid_2_eid['ProbeID'] == p]['EntrezID'].item()))


        ############################

        #find the featureset nodes in the pathway
        discriminatory_nodes = list(set(featureset_eids).intersection(set(string_node_eids)))

        #calculate pathway scores
        scores = gt.centrality_scores(A, centrality_measure)

        #average degree
        degrees = np.sum(A,axis = 0)
        avg_degree = np.mean(degrees)

        #find the indices of the nodes in the adjacency matrix that correspond to nodes in the featureset
        idx = [string_node_eids.index(r) for r in discriminatory_nodes]

        #calculate pathway scores
        node_scores = scores[idx]

        if len(node_scores) > 0:
            pathway_score = np.sum(node_scores)

            # pathway_score = np.sum(node_scores)

            pathway_scores = pathway_scores.append({'pathway_id': pathway_name, 
                                                    'unnormalized': pathway_score, 
                                                    'path norm': pathway_score/len(scores), 
                                                    'feature path norm': pathway_score/len(node_scores), 
                                                    'avg degree norm': pathway_score/avg_degree, 
                                                    'feature path count': len(node_scores)}, 
                                                    ignore_index = True)

            scores_list.append(pathway_score)
            lengths.append(len(scores))

        if ii % 200 == 0:
            print('pathway '+str(ii)+' done')

        ii+=1
    
    plt.figure()
    plt.scatter(lengths, scores_list)
    plt.xlabel('Pathway Size')
    plt.ylabel('Centrality Score')

    pathway_scores = pathway_scores.sort_values(by = 'unnormalized', ascending=False).dropna()
    #change location of saved csvs to be parameter
    if undirected:
        pathway_scores.to_csv('/data4/mankovic/GSE73072/network_centrality/undirected/gse73072_undirected_'+centrality_measure+'_pval_and_lfc.csv', index = False)
    else:
            pathway_scores.to_csv('/data4/mankovic/GSE73072/network_centrality/directed/gse73072_directed_'+centrality_measure+'_pval_and_lfc.csv', index = False)


print('starting degree directed')
calc_pathway_scores('degree', False, pid_2_eid, pathway_edges, featureset)

print('starting page rank directed')
calc_pathway_scores('page_rank', False, pid_2_eid, pathway_edges, featureset)

print('starting degree undirected')
calc_pathway_scores('degree', True, pid_2_eid, pathway_edges, featureset)

print('starting page rank undirected')
calc_pathway_scores('page_rank', True, pid_2_eid, pathway_edges, featureset)

print('starting evec undirected')
calc_pathway_scores('large_evec', True, pid_2_eid, pathway_edges, featureset)