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





def run_test(pathway_edges, centrality_measure, undirected, file_prefix):

    source = np.unique(pathway_edges['src'])
    source = source[~np.isnan(source)]
    dest = np.unique(pathway_edges['dest'])
    dest = dest[~np.isnan(dest)]
    isolated = np.unique(pathway_edges['other_genes'])
    isolated = isolated[~np.isnan(isolated)]

    #make an empty dataframe using all the eids from pathway_edges
    eids = [str(int(eid)) for eid in np.sort(list(set(source).union(set(dest).union(set(isolated)))))]

    big_pathway_centralities = pandas.DataFrame(columns = eids)

    pathway_names = np.unique(np.array(pathway_edges['pathway_id']))

    ii=0
    # go through every pathway name
    for pathway_name in pathway_names:

        row = pandas.DataFrame(columns = eids, data = [[0]*len(eids)], index=[pathway_name])

        pathway_dataframe = pathway_edges[pathway_edges['pathway_id'] == pathway_name]

        edge_dataframe = pathway_dataframe[pathway_dataframe['other_genes'].isnull()]

        isolated_dataframe = pathway_dataframe[pathway_dataframe['other_genes'].notnull()]

        scores = np.zeros(len(eids))

        if len(edge_dataframe) > 0 :
            #genes with edges
            edge_node_eids = np.array(list(set(edge_dataframe['src']).union(set(edge_dataframe['dest']))))

            #make adjacency matrix
            A, n_eids = make_network(edge_dataframe, undirected, edge_node_eids)

            #node eids as strings
            string_edge_node_eids = [str(int(node)) for node in n_eids]

            edge_scores = 1 + gt.centrality_scores(A, centrality_measure)
            
            #degrees
            max_degree = np.max(np.sum(A,axis = 0))

            row[string_edge_node_eids] = edge_scores

        else:
            edge_scores = []
            max_degree = 0

            
        if len(isolated_dataframe) > 0:
            #isolated genes
            isolated_node_eids = np.array(isolated_dataframe['other_genes'])

            #node eids as strings
            string_isolated_node_eids = [str(int(node)) for node in isolated_node_eids]

            isolated_scores = np.ones(len(string_isolated_node_eids))

            row[string_isolated_node_eids] = isolated_scores

        else:
            isolated_node_eids = []
            isolated_scores = np.array([])

        if centrality_measure == 'degree' and len(edge_scores) > 0:
            row = row/max_degree
        
        row = row/row.sum(axis = 1).item()

        if ii% 200 == 0:
            print(str(100*ii/len(pathway_names))+' percent finished')

        big_pathway_centralities = big_pathway_centralities.append(row)
        ii+=1
    if undirected:
        big_pathway_centralities.to_csv(file_prefix+'_'+centrality_measure+'_undirected.csv')
    else:
        big_pathway_centralities.to_csv(file_prefix+'_'+centrality_measure+'_directed.csv')



#load the data
pathway_edges = pandas.read_csv('/data3/darpa/omics_databases/ensembl2pathway/reactome_edges_overlap_fixed1_isolated.csv')
pathway_edges['dest'] = pandas.to_numeric(pathway_edges['dest'], downcast='integer') 
pathway_edges['src'] = pandas.to_numeric(pathway_edges['src'], downcast='integer') 
for pref in ["MN", "NM", "NR", "NC", "U"]:
    pathway_edges = pathway_edges[~pathway_edges.other_genes.str.contains(pref).fillna(False)]

pathway_edges['other_genes'] = pandas.to_numeric(pathway_edges['other_genes'], downcast='integer') 


file_prefix = '/data4/mankovic/GSE73072/network_centrality/pathway_matrix/pathway_matrix_isolated'

#degree directed
run_test(pathway_edges, 'degree', False, file_prefix)

#degree undirected
run_test(pathway_edges, 'degree', True, file_prefix)

#page rank directed
run_test(pathway_edges, 'page_rank', True, file_prefix)