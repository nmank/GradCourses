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
from sklearn.preprocessing import StandardScaler
import os


##NEED TO RERUN!

'''
By Nate Mankovich 

Pathway scores for Z75 dataset using features in: /data4/mankovic/De-Identified_CZ/z40_f_ranks.pickle.

A network for a pathway is generated using correlation. 

The centrality score for a gene in the pathway is 
    the sum of the weights of the edges adjacent to the gene
    divided by maximum weighted edge degree in the network

The centrality score for the pathway is the 
    average centrality score of the genes that are in 
    the pathway and Z75s featureset

We only do this for pathways with more than 5 genes.
'''


def run_test(centrality_measure, similarity, file_name, null = False, seed = 1):

    #load_data
    ds = dataset.load_dataset(os.path.join('/data4/zoetis/Data/TPM_C1_Z34_Z40_Z42_Z75.ds'))
    sample_ids  = ds.metadata['Project'] == 'Z75'


    feature_ids = dataset.load_dataset('/data4/sharmak/zoetis/nate/z75_ssvm_feature_set_no_partitioning.pickle')
    featureset_randIDs= list(feature_ids.index)

    Z75_dataset = ds.slice_dataset(sample_ids=sample_ids)


    preprocessing_transform = make_pipeline(HalfMinimum(missing_values=0), FunctionTransformer(np.log2), StandardScaler())
    Z75_data = pandas.DataFrame(data = preprocessing_transform.fit_transform(Z75_dataset.data), columns = Z75_dataset.data.columns, index = Z75_dataset.data.index)


    all_randIDs = list(Z75_dataset.data.columns)

    if null:
        np.random.seed(seed)
        featureset_randIDs = np.random.choice(all_randIDs, len(featureset_randIDs), replace = False)

    #which genes are in which pathways  
    pathway_data = pandas.read_csv('/data4/mankovic/ZOETIS/felis_catus_pathways.csv')
    pathway_data = pathway_data.fillna(0)
    pathway_data= pathway_data.rename(columns={"Unnamed: 0": "RandID"})

    pathway_scores = pandas.DataFrame(columns = ['pathway_id', 'unnormalized', 'max degree', 'feature path count', 'path count'])


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

            pathway_randIDs = list(pathway.columns[np.where(pathway == 1)[1]])
            # pathway_randIDs = [str(int(float(p))) for p in  list(pathway_randIDs)]
            try:
                X = np.array(Z75_data[pathway_randIDs])
            except:
                new_pathway_randIDs = []
                for p in pathway_randIDs:
                    if p in all_randIDs:
                        new_pathway_randIDs.append(p)
                pathway_randIDs = new_pathway_randIDs
                X = np.array(Z75_data[pathway_randIDs])
                
            #data matrix
            # if len(one_pathway_data) > 0:
            # if len(X) > 0:
            if np.any(X):
                # X = np.vstack(one_pathway_data).T

                #adjacency matrix
                A = gt.adjacency_matrix(X, similarity, h_k_param=300, epsilon = .3)

                #average degree
                degrees = np.sum(A,axis = 0)

                #centrality scores
                scores = 1 + gt.centrality_scores(A,centrality_measure)

                #genes in pathway that are also in the featureset
                intersect_randIDs = list(set(pathway_randIDs).intersection(set(featureset_randIDs)))

                #calculate the weighted sum of the genes in the pathway and in the featureset
                #index of nodes in graph that correspond to genes in the pathway as in the featureset
                idx = [pathway_randIDs.index(r) for r in intersect_randIDs]

                if len(idx) > 0:

                    #scores for these nodes
                    node_scores = scores[idx]

                    #pathway score as sum of node scores 
                    pathway_score = np.sum(node_scores)
                    
                    #degrees
                    degrees = np.sum(A,axis = 0)

                    #add to dataframe
                    pathway_scores = pathway_scores.append({'pathway_id': pathway_id, 
                                                            'unnormalized': pathway_score, 
                                                            'max degree': np.max(degrees),
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

# save_prefix = '/data4/mankovic/ZOETIS/pathway_ranking/Z75/pathway_scores/Z75_pathway_scores_'

# print('heat kernel started')
# run_test(   'degree', 
#             'heatkernel', 
#             save_prefix)
# print('degree heat kernel done')

# run_test(   'page_rank', 
#             'heatkernel', 
#             save_prefix)
# print('degree heat kernel done')


# save_prefix = '/data4/mankovic/ZOETIS/pathway_ranking/Z75/pathway_scores/Z75_pathway_scores_'
save_prefix = './'

print('begin correlation')
run_test(   'degree', 
            'correlation', 
            save_prefix)
print('degree correlation done')

run_test(   'page_rank', 
            'correlation', 
            save_prefix)
print('page rank correlation done')


# for seed in range(500):
#     save_prefix = '/data4/mankovic/ZOETIS/pathway_ranking/Z75/pathway_scores/Z75_pathway_scores_null'+str(seed)

#     run_test(   'degree', 
#                 'correlation', 
#                 save_prefix,
#                 null = True,
#                 seed = seed)
#     print('degree heat kernel done')

#     run_test(   'page_rank', 
#                 'correlation', 
#                 save_prefix,
#                 null = True,
#                 seed = seed)
#     print('degree heat kernel done')

#     run_test(   'degree', 
#                 'heatkernel', 
#                 save_prefix,
#                 null = True,
#                 seed = seed)
#     print('degree heat kernel done')

#     run_test(   'page_rank', 
#                 'heatkernel', 
#                 save_prefix,
#                 null = True,
#                 seed =seed)
#     print('degree heat kernel done')