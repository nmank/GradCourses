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
By Nate Mankovich 10/8/21

Pathway scores for Z40 dataset using features in: /data4/mankovic/De-Identified_CZ/z40_f_ranks.pickle.

A network for a pathway is generated using correlation. 

The centrality score for a gene in the pathway is 
    the sum of the weights of the edges adjacent to the gene
    divided by maximum weighted edge degree in the network

The centrality score for the pathway is the 
    average centrality score of the genes that are in 
    the pathway and Z40 featureset

We only do this for pathways with more than 5 genes.
'''

#choose centrality measure
centrality_measure = 'page_rank'

similarity = 'heatkernel'


#load data
f_ranks = load_object('/data4/mankovic/De-Identified_CZ/z40_f_ranks.pickle')
feature_ids = f_ranks['frequency'] > 1
featureset_randIDs = list(f_ranks[feature_ids].index)

Z40_dataset = dataset.load_dataset('/data4/mankovic/De-Identified_CZ/DeId_TPM_C1_Z40_Z34.ds')
Z40_dataset.metadata = Z40_dataset.metadata.query("Project == 'Z40' & Treatment == 'High' & Timepoint == 0.0")
sidx = list(Z40_dataset.metadata.index)
Z40_dataset.data = Z40_dataset.data.loc[sidx]

transform = make_pipeline(HalfMinimum(missing_values=0), FunctionTransformer(np.log2))
Z40_dataset.normalize(transform, norm_name='HalfMinimum + Log2')
Z40_data = Z40_dataset.data


pathway_data = pandas.read_csv('/data4/mankovic/De-Identified_CZ/deidentified_fcpw.csv')


pathway_scores = pandas.DataFrame(columns = ['pathway id', 'unnormalized', 'path norm', 'feature path norm', 'avg degree norm', 'feature path count'])


for pathway_id in list(pathway_data['RandID']):

    #select one pathway
    pathway = pathway_data[pathway_data['RandID'] == pathway_id]

    #only do this if there are more than 5 genes in the pathway
    if len(np.where(pathway == True)[1]) > 5:

        one_pathway_data = []
        pathway_randIDs = []
        #go through each randID in C1_data
        for randID in Z40_data.columns:
            #if the gene is in the pathway
            if list(pathway[randID])[0] == True:
                #collect randid's in the pathway
                pathway_randIDs.append(randID)
                #add it's data to the pathway data
                one_pathway_data.append(np.array(Z40_data[randID]))
            
        #data matrix
        X = np.vstack(one_pathway_data).T

        #adjacency matrix
        A = gt.adjacency_matrix(X,similarity, h_k_param=100)

        #average degree
        degrees = np.sum(A,axis = 0)
        avg_degree = np.mean(degrees)

        #centrality scores
        scores = gt.centrality_scores(A,centrality_measure)

        #genes in pathway that are also in the featureset
        intersect_randIDs = list(set(pathway_randIDs).intersection(set(featureset_randIDs)))

        #calculate the weighted sum of the genes in the pathway and in the featureset
        if len(intersect_randIDs) != 0:
            #index of nodes in graph that correspond to genes in the pathway as in the featureset
            idx = [pathway_randIDs.index(r) for r in intersect_randIDs]

            #scores for these nodes
            node_scores = scores[idx]

            #pathway score as sum of node scores 
            pathway_score = np.sum(node_scores)

            #add to dataframe
            pathway_scores = pathway_scores.append({'pathway id': pathway_id, 
                                                    'unnormalized': pathway_score, 
                                                    'path norm': pathway_score/len(scores), 
                                                    'feature path norm': pathway_score/len(node_scores), 
                                                    'avg degree norm': pathway_score/avg_degree, 
                                                    'feature path count': len(node_scores)}, 
                                                    ignore_index = True)

    print(pathway_id + ' done!')

#sort pathway scores from highest to lowest
pathway_scores = pathway_scores.sort_values(by = 'unnormalized', ascending=False)

#save to csv
pathway_scores.to_csv('/home/katrina/a/mankovic/ZOETIS/Fall2021/pathway_ranking/Z40_pathway_scores_'+similarity+'_'+centrality_measure+'.csv', index = False)
