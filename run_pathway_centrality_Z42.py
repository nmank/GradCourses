#https://github.com/ekehoe32/orthrus
import sys
sys.path.append('/home/katrina/a/mankovic/ZOETIS/Fall2021/Orthrus/orthrus')
import orthrus
from orthrus import core
from orthrus.core import dataset
import numpy as np
import graph_tools_construction as gt
from matplotlib import pyplot as plt
import pandas
# from orthrus.core.pipeline import *
from sklearn.preprocessing import FunctionTransformer
from orthrus.preprocessing.imputation import HalfMinimum
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from orthrus.core.helper import load_object

'''
By Nate Mankovich 

Pathway scores for Z42 dataset using features in: /data4/mankovic/De-Identified_CZ/z40_f_ranks.pickle.

A network for a pathway is generated using correlation. 

The centrality score for a gene in the pathway is 
    the sum of the weights of the edges adjacent to the gene
    divided by maximum weighted edge degree in the network

The centrality score for the pathway is the 
    average centrality score of the genes that are in 
    the pathway and Z42 featureset

We only do this for pathways with more than 5 genes.
'''


def run_test(centrality_measure, similarity, file_name, null = False, seed = 1):
    #load data
        # f_ranks = load_object('/data4/mankovic/De-Identified_CZ/z40_f_ranks.pickle')
        # feature_ids = f_ranks['frequency'] > 1
        # featureset_randIDs = list(f_ranks[feature_ids].index)

        # Z40_dataset = dataset.load_dataset('/data4/mankovic/De-Identified_CZ/DeId_TPM_C1_Z40_Z34.ds')
        # Z40_dataset.metadata = Z40_dataset.metadata.query("Project == 'Z40' & Treatment == 'High' & Timepoint == 0.0")
        # sidx = list(Z40_dataset.metadata.index)
        # Z40_dataset.data = Z40_dataset.data.loc[sidx]

        # transform = make_pipeline(HalfMinimum(missing_values=0), FunctionTransformer(np.log2))
        # Z40_dataset.normalize(transform, norm_name='HalfMinimum + Log2')
        # Z40_data = Z40_dataset.data

    #load Kartikay's gene feature set and store RandIDs for genes
    Z42_features = pandas.read_csv('/data4/mankovic/De-Identified_CZ/z42_features_after_hyperparameter_search.csv', index_col = 0)
    featureset_randIDs = [str(r) for r in list(Z42_features.index)]

    #load the entire C1 and Z40 dataset and select the C1 data and metadata
    ds = dataset.load_dataset('/data4/mankovic/De-Identified_CZ/DeId_TPM_C1_Z40_Z34_Z42.ds')
    sample_ids = (ds.metadata['Project'] == 'Z42') & (ds.metadata['Time'] != -21)
    Z42_dataset = ds.slice_dataset(sample_ids = sample_ids)

    #normalize the dataset
    transform = make_pipeline(HalfMinimum(missing_values=0), FunctionTransformer(np.log2))
    Z42_dataset.normalize(transform, norm_name='HalfMinimum + Log2')

    Z42_data = Z42_dataset.data
    all_randIDs = np.unique(list(Z42_data.columns))

    if null:
        np.random.seed(seed)
        featureset_randIDs = np.random.choice(all_randIDs, len(featureset_randIDs), replace = False)

    pathway_data = pandas.read_csv('/data4/mankovic/De-Identified_CZ/deidentified_fcpw_updated.csv')
    pathway_data = pathway_data.fillna(0)



    pathway_scores = pandas.DataFrame(columns = ['pathway_id', 'unnormalized', 'path norm', 'feature path norm', 'avg degree norm', 'max degree norm', 'feature path count', 'path count'])


    all_len = len(list(pathway_data['RandID']))
    ii= 0
    for pathway_id in list(pathway_data['RandID']):
        ii+=1

        #select one pathway
        pathway = pathway_data[pathway_data['RandID'] == pathway_id]

        #only do this if there are more than 5 genes in the pathway
        if len(np.where(pathway == 1)[1]) > 5:

            # one_pathway_data = []
            # pathway_randIDs = []
            # #go through each randID in data
            # for randID in all_randIDs:
            #     if randID in list(pathway.columns[1:]):
            #         #if the gene is in the pathway
            #         if list(pathway[randID])[0] == True:
            #             #collect randid's in the pathway
            #             pathway_randIDs.append(randID)
            #             #add it's data to the pathway data
            #             one_pathway_data.append(np.array(Z42_data[randID]))

            pathway_randIDs = pathway.columns[np.where(pathway == 1)[1]]
            pathway_randIDs = [str(int(float(p))) for p in  list(pathway_randIDs)]
            try:
                X = np.array(Z42_data[pathway_randIDs])
            except:
                new_pathway_randIDs = []
                for p in pathway_randIDs:
                    if p in all_randIDs:
                        new_pathway_randIDs.append(p)
                pathway_randIDs = new_pathway_randIDs
                X = np.array(Z42_data[pathway_randIDs])
                
            #data matrix
            # if len(one_pathway_data) > 0:
            if len(X) > 0:
                # X = np.vstack(one_pathway_data).T

                #adjacency matrix
                A = gt.adjacency_matrix(X, similarity, h_k_param=300)

                #average degree
                degrees = np.sum(A,axis = 0)

                #centrality scores
                scores = 1 + gt.centrality_scores(A,centrality_measure)

                #genes in pathway that are also in the featureset
                intersect_randIDs = list(set(pathway_randIDs).intersection(set(featureset_randIDs)))

                #calculate the weighted sum of the genes in the pathway and in the featureset
                #index of nodes in graph that correspond to genes in the pathway as in the featureset
                idx = [pathway_randIDs.index(r) for r in intersect_randIDs]

                #scores for these nodes
                node_scores = scores[idx]

                #pathway score as sum of node scores 
                pathway_score = np.sum(node_scores)

                #degrees
                degrees = np.sum(A,axis = 0)

                #add to dataframe
                pathway_scores = pathway_scores.append({'pathway_id': pathway_id, 
                                                        'unnormalized': pathway_score, 
                                                        'path norm': pathway_score/len(scores), 
                                                        'feature path norm': pathway_score/len(node_scores), 
                                                        'avg degree norm': pathway_score/np.mean(degrees), 
                                                        'max degree norm': pathway_score/np.max(degrees),
                                                        'feature path count': len(node_scores),
                                                        'path count' : len(scores)},
                                                        ignore_index = True)
            if ii % 100 == 0:
                print('percent finished: '+ str(100*ii/all_len))


    #sort pathway scores from highest to lowest
    pathway_scores = pathway_scores.sort_values(by = 'unnormalized', ascending=False)

    #save to csv
    pathway_scores.to_csv(file_name+similarity+'_'+centrality_measure+'.csv', index = False)



#choose centrality measure

save_prefix = '/home/katrina/a/mankovic/ZOETIS/Fall2021/pathway_ranking/Z42_pathway_scores_'

print('heat kernel started')
run_test(   'degree', 
            'heatkernel', 
            save_prefix)
print('degree heat kernel done')

run_test(   'page_rank', 
            'heatkernel', 
            save_prefix)
print('degree heat kernel done')

save_prefix = '/home/katrina/a/mankovic/ZOETIS/Fall2021/pathway_ranking/Z42_pathway_scores_null'

print('heat kernel started')
run_test(   'degree', 
            'heatkernel', 
            save_prefix,
            null = True)
print('degree heat kernel done')

run_test(   'page_rank', 
            'heatkernel', 
            save_prefix,
            null = True)
print('degree heat kernel done')

# run_test(   'large_evec', 
#             'heatkernel', 
#             '/home/katrina/a/mankovic/ZOETIS/Fall2021/pathway_ranking/Z40_pathway_scores_')
# print('large evec heat kernel done')

save_prefix = '/home/katrina/a/mankovic/ZOETIS/Fall2021/pathway_ranking/Z42_pathway_scores_'

print('begin correlation')
run_test(   'degree', 
            'correlation', 
            save_prefix)
print('degree correlation done')

run_test(   'page_rank', 
            'correlation', 
            save_prefix)
print('page rank correlation done')

save_prefix = '/home/katrina/a/mankovic/ZOETIS/Fall2021/pathway_ranking/Z42_pathway_scores_null'

run_test(   'degree', 
            'correlation', 
            save_prefix,
            null = True)
print('degree correlation null done')

run_test(   'page_rank', 
            'correlation', 
            save_prefix,
            null = True)
print('page rank correlation null done')

# run_test(   'large_evec', 
#             'correlation', 
#             '/home/katrina/a/mankovic/ZOETIS/Fall2021/pathway_ranking/Z40_pathway_scores_')
# print('large evec correlation done')

for seed in range(20):
    save_prefix = '/home/katrina/a/mankovic/ZOETIS/Fall2021/pathway_ranking/Z42_pathway_scores_null'+str(seed)

    run_test(   'degree', 
                'correlation', 
                save_prefix,
                null = True,
                seed = seed)
    print('degree heat kernel done')

    run_test(   'page_rank', 
                'correlation', 
                save_prefix,
                null = True,
                seed = seed)
    print('degree heat kernel done')

    run_test(   'degree', 
                'heatkernel', 
                save_prefix,
                null = True,
                seed = seed)
    print('degree heat kernel done')

    run_test(   'page_rank', 
                'heatkernel', 
                save_prefix,
                null = True,
                seed =seed)
    print('degree heat kernel done')