#https://github.com/ekehoe32/orthrus
import sys
sys.path.append('/home/katrina/a/mankovic/ZOETIS/Fall2021/Orthrus/orthrus')
import orthrus
from orthrus import core
from orthrus.core import dataset
import numpy as np
from NetworkDataAnalysis import graph_tools_construction as gt
from matplotlib import pyplot as plt
import pandas
# from orthrus.core.pipeline import *
from sklearn.preprocessing import FunctionTransformer
from orthrus.preprocessing.imputation import HalfMinimum
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from orthrus.core.helper import load_object

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


    #load data
    Z40_dataset = dataset.load_dataset('/data4/mankovic/De-Identified_CZ/DeId_TPM_C1_Z40_Z34.ds')
    Z40_dataset.metadata = Z40_dataset.metadata.query("Project == 'Z40' & Treatment == 'High' & Timepoint == 0.0")
    sidx = list(Z40_dataset.metadata.index)
    Z40_dataset.data = Z40_dataset.data.loc[sidx]

    #transform data (according to Kartikay)
    transform = make_pipeline(HalfMinimum(missing_values=0), FunctionTransformer(np.log2))
    Z40_dataset.normalize(transform, norm_name='HalfMinimum + Log2')
    Z40_data = Z40_dataset.data

    #which genes are in which pathways
    pathway_data = pandas.read_csv('/data4/mankovic/De-Identified_CZ/deidentified_fcpw.csv')

    #restrict pathway data to to genes that are actually there
    pathway_data = pathway_data[['RandID']+list(Z40_data.columns)]

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
        current_pathway_data = np.array(Z40_data[pathway_rand_ids ] )

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



    network_pathway_data.to_csv('/home/katrina/a/mankovic/ZOETIS/Fall2021/pathway_ranking/Z40_pathway_matrix_'+similarity+'_'+centrality_measure+'.csv', index = False)


#generate pathway matrices for degree, pagerank using correlation and heat kernel
run_test('degree', 'correlation')
print('degree done correlation done')
run_test('page_rank', 'correlation')
print('page rank correlation done')

run_test('degree', 'heatkernel')
print('degree heat kernel done')
run_test('page_rank', 'heatkernel')
print('page rank heat kernel done')