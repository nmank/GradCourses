import pandas
import numpy as np
from matplotlib import pyplot as plt
import graph_tools_construction as gt
import os

def calc_pathway_scores(centrality_measure, undirected, featureset_eids, outfile = 'output.csv', random = False, seed = 0):


    base_dir = '/data3/darpa/omics_databases/ensembl2pathway/pathways_edges/pathways/pw_edge_mtx/'

    pathway_scores = pandas.DataFrame(columns = ['pathway_id', 'unnormalized', 'path norm', 'feature path norm', 'max degree norm', 'feature path count', 'path count'])


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


        dangle_idx = np.where(np.sum(A, axis = 0) == 0)[0]
        connected_idx = np.where(np.sum(A, axis = 0) != 0)[0]

        scores = np.zeros(len(A))

        if len(connected_idx) > 0:
            scores[connected_idx] = 1 + gt.centrality_scores(A[connected_idx,:][:,connected_idx], centrality_measure)
        if len(dangle_idx) > 0:
            scores[dangle_idx] = 1

        max_degree = np.max(np.sum(A,axis = 0))
        #avoid dividing by 0
        if max_degree == 0:
            max_degree = 1

        string_eids = [str(node) for node in eids]

        discriminatory_idx = [eids.index(int(e)) for e in list(set(featureset_eids).intersection(set(string_eids)))]

        node_scores = scores[discriminatory_idx]

        pathway_score = np.sum(node_scores)

        pathway_name = pid

        pathway_scores = pathway_scores.append({'pathway_id': pathway_name, 
                                                    'unnormalized': pathway_score, 
                                                    'path norm': pathway_score/len(scores), 
                                                    'feature path norm': pathway_score/len(node_scores), 
                                                    'max degree norm': pathway_score/max_degree,
                                                    'feature path count': len(node_scores),
                                                    'path count' : len(scores)},
                                                    ignore_index = True)


    pathway_scores.sort_values(by = 'unnormalized', ascending=False).dropna()

    pathway_scores.to_csv(outfile, index = False)
    
    # plt.figure()
    # plt.scatter(lengths, scores_list)
    # plt.xlabel('Pathway Size')
    # plt.ylabel('Centrality Score')




#load the data

# metadata = pandas.read_csv('/data4/kehoe/GSE73072/GSE73072_metadata.csv')
# vardata = pandas.read_csv('/data4/kehoe/GSE73072/GSE73072_vardata.csv')

# pathway_edges = pandas.read_csv('/data3/darpa/omics_databases/ensembl2pathway/reactome_edges_overlap_fixed1_noisolated.csv')
# pathway_edges['dest'] = pandas.to_numeric(pathway_edges['dest'], downcast='integer') 
# pathway_edges['src'] = pandas.to_numeric(pathway_edges['src'], downcast='integer') 

# for pref in ["MN", "NM", "NR", "NC", "U"]:
#     pathway_edges = pathway_edges[~pathway_edges.other_genes.str.contains(pref).fillna(False)]

# pathway_edges['other_genes'] = pandas.to_numeric(pathway_edges['other_genes'], downcast='integer') 

# featureset = pandas.read_csv('/data4/mankovic/GSE73072/network_centrality/featuresets/diffgenes_gse73072_pval_and_lfc.csv', index_col=0)

#####################

#do this only for train_best_probe_ids.csv file
# featureset = pandas.read_csv('/data4/mankovic/GSE73072/network_centrality/featuresets/2-4hr/train_best_probe_ids.csv', index_col=0)
# pid_2_eid = pandas.read_csv('/data4/mankovic/GSE73072/probe_2_entrez.csv')
# featureset_pids = list(featureset.index)
# featureset_eids = []
# #load eids from the probeids in the featureset
# for p in featureset_pids:
#     if p in list(pid_2_eid['ProbeID']):
#         featureset_eids.append(str(pid_2_eid[pid_2_eid['ProbeID'] == p]['EntrezID'].item()))
# directories = '/data4/mankovic/GSE73072/network_centrality/simple_rankings/2-4hr/lfc/'

#####################

#ssvm features
featureset = pandas.read_csv('/data4/mankovic/GSE73072/network_centrality/featuresets/2-4hr/ssvm_ranked_features.csv', index_col=0)
#do this for top 316 ssvm features with frequency greater than 8
featureset_eids = [str(f) for f in list(featureset.query("Frequency>8").index)]
directories = '/data4/mankovic/GSE73072/network_centrality/simple_rankings/2-4hr/ssvm/'

#####################

# print('starting degree directed')
# calc_pathway_scores('degree', False, featureset_eids, directories+'gse73072_directed_degree.csv')

# print('starting degree undirected')
# calc_pathway_scores('degree', True, featureset_eids, directories+'gse73072_undirected_degree.csv')

# print('starting page rank undirected')
# calc_pathway_scores('page_rank', True, featureset_eids, directories+'gse73072_undirected_pagerank.csv')


# #####################


for trial in range(100,500,1):
    print('Null trial'+str(trial))

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
    str_all_eids = [str(e) for e in all_eids]
    
    np.random.seed(trial)
    null_featureset = np.random.choice(str_all_eids, len(set(featureset_eids).intersection(set(str_all_eids))), replace = False)
    null_featureset = [str(f) for f in null_featureset]

    print('starting degree directed')
    calc_pathway_scores('degree', False,  null_featureset, directories+'gse73072_directed_degree_null'+str(trial)+'.csv')

    print('starting degree undirected')
    calc_pathway_scores('degree', True, null_featureset, directories+'gse73072_undirected_degree_null'+str(trial)+'.csv')

    print('starting page rank undirected')
    calc_pathway_scores('page_rank', True,  null_featureset, directories+'gse73072_undirected_pagerank_null'+str(trial)+'.csv')

    # print('starting degree undirected')
    # calc_pathway_scores('degree', True, pathway_edges, null_featureset, directories+'gse73072_undirected_degree_ER_null'+str(trial)+'.csv', random = True, seed = trial)

    # print('starting page rank undirected')
    # calc_pathway_scores('page_rank', True, pathway_edges, null_featureset, directories+'gse73072_undirected_pagerank_ER_null'+str(trial)+'.csv', random = True, seed = trial)