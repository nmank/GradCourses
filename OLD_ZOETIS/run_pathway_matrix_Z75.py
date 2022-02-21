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

'''
This script calculates a pathway matrix for (pathways x RandIDs) using different centrality measures and similarities.

By nate mankovich

'''



def run_test(centrality_measure, similarity):
    '''
    Calculates a pathway matrix.

    Inputs:
        centrality_measure: string for the type of centrality to use:. Options are:
                                'degree'
                                'page_rank'
                                'large_rank'
        similarity: string for the edge generation method. Options are:
                        'correlation'
                        'heat_kernel'
                                
'''




     #load_data
    ds = dataset.load_dataset(os.path.join('/data4/zoetis/Data/TPM_C1_Z34_Z40_Z42_Z75.ds'))
    sample_ids  = ds.metadata['Project'] == 'Z75'
    Z75_dataset = ds.slice_dataset(sample_ids=sample_ids)
    preprocessing_transform = make_pipeline(HalfMinimum(missing_values=0), FunctionTransformer(np.log2), StandardScaler())
    Z75_data = pandas.DataFrame(data = preprocessing_transform.fit_transform(Z75_dataset.data), columns = Z75_dataset.data.columns, index = Z75_dataset.data.index)


    #which genes are in which pathways
    pathway_data = pandas.read_csv('/data4/mankovic/ZOETIS/felis_catus_pathways.csv')
    pathway_data = pathway_data.fillna(0)
    pathway_data= pathway_data.rename(columns={"Unnamed: 0": "RandID"})

    #restrict pathway data to to genes that are actually there
    pathway_data = pathway_data[['RandID']+list(Z75_data.columns)]

    #new dataframe to store pathway matrix
    network_pathway_data = pandas.DataFrame(columns = list(pathway_data.columns))

    #all rand_ids for pathway_data including 'RandID' as first item
    rand_ids = list(pathway_data.columns)
  
    count = 0
    for pathway_id in list(pathway_data['RandID']):

        #select one pathway
        pathway = pathway_data[pathway_data['RandID'] == pathway_id]

        #genes in the pathway
        idx = np.where(np.array(pathway) == True)[1]
        pathway_rand_ids = [rand_ids[i] for i in idx]
        
        #data for genes in the pathway
        current_pathway_data = np.array(Z75_data[pathway_rand_ids ] )

        #adjacency matrix
        A = gt.adjacency_matrix(current_pathway_data,similarity, h_k_param=100)

        #centrality scores
        scores = gt.centrality_scores(A,centrality_measure)

        #normalize centrality score by l1 norm
        scores = scores/np.sum(scores)

        #add to dataframe
        row = pandas.DataFrame([[pathway_id]+[0]*(len(rand_ids)-1)], columns = rand_ids)
        row[pathway_rand_ids] = scores.flatten()
        network_pathway_data = network_pathway_data.append(row, ignore_index = True)

        if count % 500 == 0:
            print('.')

        count += 1



    network_pathway_data.to_csv('/data4/mankovic/ZOETIS/pathway_ranking/Z75/Z75_pathway_matrix_'+similarity+'_'+centrality_measure+'.csv', index = False)


#generate pathway matrices for degree, pagerank using correlation and heat kernel
run_test('degree', 'correlation')
print('degree done correlation done')
run_test('page_rank', 'correlation')
print('page rank correlation done')

run_test('degree', 'heatkernel')
print('degree heat kernel done')
run_test('page_rank', 'heatkernel')
print('page rank heat kernel done')