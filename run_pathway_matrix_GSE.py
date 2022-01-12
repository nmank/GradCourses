import pandas
import numpy as np
from matplotlib import pyplot as plt
import graph_tools_construction as gt
import os



def run_test(centrality_measure, undirected, file_prefix):

    base_dir = '/data3/darpa/omics_databases/ensembl2pathway/pathways_edges/pathways/pw_edge_mtx/'

    all_eids = []
    for f in os.listdir(base_dir):
        x = pandas.read_csv(base_dir + f, index_col = 0)
        # if not x.isnull().all().all():
        eids = []
        for g in list(x.index):
            try:
                eids.append(int(g[7:]))
            except:
                print('')
        all_eids += eids
    
    all_eids = list(np.unique(eids))

    big_pathway_centralities = pandas.DataFrame(columns = all_eids)

    for f in os.listdir(base_dir):
        pid= f.replace("/data3/darpa/omics_databases/ensembl2pathway/pathways_edges/pathways/pw_edge_mtx/pw_mtx_", '').replace('.csv', '')
        x = pandas.read_csv(base_dir + f, index_col = 0)
        gene_cols = []
        eids = []
        for g in list(x.index):
            try:
                eids.append(int(g[7:]))
                gene_cols.append(g)
            except:
                print('')

        A = np.array(x[gene_cols])
        idx = np.where(A == 2)
        A[idx] = 1
        A[idx[1],idx[0]] = 1

        A[A != A] = 0

        if undirected:
            idx = np.where(A == 1)
            A[idx[1],idx[0]] = 1

    
        row = pandas.DataFrame(columns = eids, data = [[0]*len(eids)], index=[pid])

        dangle_idx = np.where(np.sum(A, axis = 0) == 0)[0]
        connected_idx = np.where(np.sum(A, axis = 0) != 0)[0]

        if len(connected_idx) > 0:
            c_eidx = [eids[i] for i in connected_idx]
            row[c_eidx] = 1 + gt.centrality_scores(A[connected_idx,:][:,connected_idx], centrality_measure)
        if len(dangle_idx) > 0:
            d_eidx = [eids[i] for i in dangle_idx]
            row[d_eidx] = 1

        row = row/row.sum(axis = 1).item()

        big_pathway_centralities = big_pathway_centralities.append(row)

    if undirected:
        big_pathway_centralities.to_csv(file_prefix+'_'+centrality_measure+'_undirected.csv')
    else:
        big_pathway_centralities.to_csv(file_prefix+'_'+centrality_measure+'_directed.csv')





file_prefix = '/data4/mankovic/GSE73072/network_centrality/pathway_matrix/pathway_matrix_isolated'

#degree directed
run_test('degree', False, file_prefix)

#degree undirected
run_test('degree', True, file_prefix)

#page rank directed
run_test('page_rank', True, file_prefix)