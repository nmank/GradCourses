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

import GLPE

pathway_data = pandas.read_csv('/data4/mankovic/ZOETIS/felis_catus_pathways.csv')
pathway_data = pathway_data.fillna(0)
pathway_data= pathway_data.rename(columns={"Unnamed: 0": "ReactomeID"})


ds = dataset.load_dataset(os.path.join('/data4/zoetis/Data/TPM_C1_Z34_Z40_Z42_Z75.ds'))
sample_ids  = ds.metadata['Project'] == 'Z75'
Z75_dataset = ds.slice_dataset(sample_ids=sample_ids)
preprocessing_transform = make_pipeline(HalfMinimum(missing_values=0), FunctionTransformer(np.log2), StandardScaler())
Z75_data = pandas.DataFrame(data = preprocessing_transform.fit_transform(Z75_dataset.data), columns = Z75_dataset.data.columns, index = Z75_dataset.data.index)

pathway_data = pathway_data[["ReactomeID"]+list(Z75_data.columns)]

new_column_dict = dict(zip(pathway_data.columns, ["ReactomeID"] + list(range(len(Z75_data.columns)))))
de_identified_pathway_data = pathway_data.rename(columns = new_column_dict)

better_pathway_data=pandas.DataFrame(columns = ['feature_id', 'pathway_id'])
gene_names = de_identified_pathway_data.columns
for row in np.array(de_identified_pathway_data):
    idx = np.where(row == 1)
    # print(row)
    for g in gene_names[idx]:
        better_pathway_data = better_pathway_data.append({'feature_id': int(g), 'pathway_id':row[0]}, ignore_index = True)

featureset = dataset.load_dataset('/data4/sharmak/zoetis/nate/z75_ssvm_feature_set_no_partitioning.pickle')
featureset_ids= list(featureset.index)
featureset_transition_matrix_ids = np.nonzero(np.in1d(np.array(Z75_data.columns),np.array(featureset_ids)))[0]


different_types = [('degree', 'correlation'), 
                    ('page_rank', 'correlation'),
                    ('degree', 'heatkernel'),
                    ('page_rank', 'heatkernel'), ]


for centrality_type, similarity in different_types:

    my_other_clpe = GLPE.CLPE('degree', 
                        'correlation',
                        better_pathway_data, 
                        heat_kernel_param = 100,
                        normalize_rows = False)

    my_other_clpe.fit(np.array(Z75_data))

    simple_centrality_scores = my_other_clpe.simple_transform(featureset_transition_matrix_ids, n_null_trials = 500)

    simple_centrality_scores.to_csv('/data4/mankovic/ZOETIS/pathway_ranking/Z75/pathway_scores/Z75_pathway_scores_'+similarity+'_'+centrality_type+'.csv', index = False)

    print('simple_centrality '+centrality_type+' '+similarity+' done!')