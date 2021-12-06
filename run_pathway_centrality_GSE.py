import pandas
import numpy as np
from matplotlib import pyplot as plt
import graph_tools_construction as gt





def make_network(edge_dataframe, undirected, node_eids):
    '''
    Make a network from the known edges.

    Inputs:
        pathway_name: a string for the name of a pathway, corresponds to 'pathway_id' in all_edge_dataframe
        all_edge_dataframe: a dataframe with network data. The columns are:
                            'pathway_id': identifier of the pathway
                            'src': source node identifier
                            'dest': destination node identifier
                            'weight': the edge weight
                            'direction': 'undirected' for undirected edge
        undirected: a boolean, True for undirected and False for directed

    
    Outputs:
        A: a numpy array of the adjacency matrix (directed)
        node_eids: a list of EntrezIDs whose indices correspond to the entries of A
    '''

    if 'weight' in list(edge_dataframe.columns):
        weighted = True
    else:
        weighted = False

    n_nodes = len(node_eids)

    A = np.zeros((n_nodes, n_nodes))

    for _,row in edge_dataframe.iterrows():
        if np.isnan(row['other_genes']):
            i = np.where(node_eids == row['src'])[0][0]
            j = np.where(node_eids == row['dest'])[0][0]
            if weighted:
                A[i,j] = row['weight'].item()
            else:
                A[i,j] = 1
                if undirected or row['direction'] == 'undirected':
                    A[j,i] = A[i,j].copy()
        else:
            i = np.where(node_eids == row['other_genes'])[0][0]
            A[i,i] = 0
    

    return A, node_eids

def calc_pathway_scores(centrality_measure, undirected, pathway_edges, featureset_eids, outfile = 'output.csv'):
    '''
    '''
    # load names of the pathways and init pathway dataframe
    pathway_names = np.unique(np.array(pathway_edges['pathway_id']))

    pathway_scores = pandas.DataFrame(columns = ['pathway_id', 'unnormalized', 'path norm', 'feature path norm', 'max degree norm', 'feature path count', 'path count'])

    lengths = []

    scores_list = []

    ii=0
    # go through every pathway name
    for pathway_name in pathway_names:

        pathway_dataframe = pathway_edges[pathway_edges['pathway_id'] == pathway_name]

        edge_dataframe = pathway_dataframe[pathway_dataframe['other_genes'].isnull()]

        isolated_dataframe = pathway_dataframe[pathway_dataframe['other_genes'].notnull()]

        scores = np.zeros(len(pathway_dataframe))

        if len(edge_dataframe) > 0 :
            #genes with edges
            edge_node_eids = np.array(list(set(edge_dataframe['src']).union(set(edge_dataframe['dest']))))

            #make adjacency matrix
            A, n_eids = make_network(edge_dataframe, undirected, edge_node_eids)

            #node eids as strings
            string_edge_node_eids = [str(int(node)) for node in n_eids]

            #discriminatory genes
            discriminatory_edge_nodes = list(set(featureset_eids).intersection(set(string_edge_node_eids)))

            edge_scores = 1 + gt.centrality_scores(A, centrality_measure)

            #find the indices of the nodes in the adjacency matrix that correspond to nodes in the featureset
            idx = [string_edge_node_eids.index(r) for r in discriminatory_edge_nodes]

            featureset_edge_scores = edge_scores[idx]
            
            #degrees
            max_degree = np.max(np.sum(A,axis = 0))

        else:
            discriminatory_edge_nodes = []
            featureset_edge_scores = np.array([])
            max_degree = 1

            
        if len(isolated_dataframe) > 0:
            #isolated genes
            isolated_node_eids = np.array(isolated_dataframe['other_genes'])

            #node eids as strings
            string_isolated_node_eids = [str(int(node)) for node in isolated_node_eids]

            discriminatory_isolated_nodes = list(set(featureset_eids).intersection(set(string_isolated_node_eids)))

            featureset_isolated_scores = np.ones(len(discriminatory_isolated_nodes))

        else:
            discriminatory_isolated_nodes = []
            featureset_isolated_scores = np.array([])

        #find the featureset nodes in the pathway
        # discriminatory_nodes = discriminatory_edge_nodes + discriminatory_isolated_nodes

        


        if len(featureset_edge_scores) + len(featureset_isolated_scores) > 0:
            node_scores = np.hstack([featureset_edge_scores,featureset_isolated_scores])
            pathway_score = np.sum(node_scores)


            # pathway_score = np.sum(node_scores)
 
            pathway_scores = pathway_scores.append({'pathway_id': pathway_name, 
                                                    'unnormalized': pathway_score, 
                                                    'path norm': pathway_score/len(scores), 
                                                    'feature path norm': pathway_score/len(node_scores), 
                                                    'max degree norm': pathway_score/max_degree,
                                                    'feature path count': len(node_scores),
                                                    'path count' : len(scores)},
                                                    ignore_index = True)

            scores_list.append(pathway_score)
            lengths.append(len(scores))

        if ii % 200 == 0:
            print('pathway '+str(ii)+' done')

        ii+=1

    pathway_scores.sort_values(by = 'unnormalized', ascending=False).dropna()

    pathway_scores.to_csv(outfile, index = False)
    
    # plt.figure()
    # plt.scatter(lengths, scores_list)
    # plt.xlabel('Pathway Size')
    # plt.ylabel('Centrality Score')




#load the data

# metadata = pandas.read_csv('/data4/kehoe/GSE73072/GSE73072_metadata.csv')
# vardata = pandas.read_csv('/data4/kehoe/GSE73072/GSE73072_vardata.csv')

pathway_edges = pandas.read_csv('//data3/darpa/omics_databases/ensembl2pathway/reactome_edges_overlap_fixed1_isolated.csv')
pathway_edges['dest'] = pandas.to_numeric(pathway_edges['dest'], downcast='integer') 
pathway_edges['src'] = pandas.to_numeric(pathway_edges['src'], downcast='integer') 
for pref in ["MN", "NM", "NR", "NC", "U"]:
    pathway_edges = pathway_edges[~pathway_edges.other_genes.str.contains(pref).fillna(False)]

pathway_edges['other_genes'] = pandas.to_numeric(pathway_edges['other_genes'], downcast='integer') 

# featureset = pandas.read_csv('/data4/mankovic/GSE73072/network_centrality/featuresets/diffgenes_gse73072_pval_and_lfc.csv', index_col=0)

#####################

#do this only for train_best_probe_ids.csv file
featureset = pandas.read_csv('/data4/mankovic/GSE73072/network_centrality/featuresets/train_best_probe_ids.csv', index_col=0)
pid_2_eid = pandas.read_csv('/data4/mankovic/GSE73072/probe_2_entrez.csv')
featureset_pids = list(featureset.index)
featureset_eids = []
#load eids from the probeids in the featureset
for p in featureset_pids:
    if p in list(pid_2_eid['ProbeID']):
        featureset_eids.append(str(pid_2_eid[pid_2_eid['ProbeID'] == p]['EntrezID'].item()))
directories = '/data4/mankovic/GSE73072/network_centrality/simple_rankings/2-4hr/lfc/'

#####################

#ssvm features
# featureset = pandas.read_csv('/data4/mankovic/GSE73072/network_centrality/featuresets/ssvm_ranked_features.csv', index_col=0)
# #do this for top 316 ssvm features with frequency greater than 8
# featureset_eids = [str(f) for f in list(featureset.query("Frequency>8").index)]
# directories = '/data4/mankovic/GSE73072/network_centrality/simple_rankings/2-4hr/ssvm/'

#####################



print('starting degree directed')
calc_pathway_scores('degree', False, pathway_edges, featureset_eids, directories+'gse73072_directed_degree.csv')

print('starting degree undirected')
calc_pathway_scores('degree', True, pathway_edges, featureset_eids, directories+'gse73072_undirected_degree.csv')

print('starting page rank undirected')
calc_pathway_scores('page_rank', True, pathway_edges, featureset_eids, directories+'gse73072_undirected_pagerank.csv')


#####################


for trial in range(20):
    print('Null trial'+str(trial))

    #null models
    source = np.unique(pathway_edges['src'])
    source = source[~np.isnan(source)]
    dest = np.unique(pathway_edges['dest'])
    dest = dest[~np.isnan(dest)]
    isolated = np.unique(pathway_edges['other_genes'])
    isolated = isolated[~np.isnan(isolated)]

    #make an empty dataframe using all the eids from pathway_edges
    str_all_eids = [str(int(eid)) for eid in np.sort(list(set(source).union(set(dest).union(set(isolated)))))]

    
    np.random.seed(trial)
    null_featureset = np.random.choice(str_all_eids, len(set(featureset_eids).intersection(set(str_all_eids))), replace = False)
    null_featureset = [str(f) for f in null_featureset]

    print('starting degree directed')
    calc_pathway_scores('degree', False, pathway_edges, null_featureset, directories+'gse73072_directed_degree_null'+str(trial)+'.csv')

    print('starting degree undirected')
    calc_pathway_scores('degree', True, pathway_edges, null_featureset, directories+'gse73072_undirected_degree_null'+str(trial)+'.csv')

    print('starting page rank undirected')
    calc_pathway_scores('page_rank', True, pathway_edges, null_featureset, directories+'gse73072_undirected_pagerank_null'+str(trial)+'.csv')