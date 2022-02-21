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

different_types = [('degree', 'correlation'), 
                    ('page_rank', 'correlation'),
                    ('degree', 'heatkernel'),
                    ('page_rank', 'heatkernel'), ]


for centrality_type, similarity in different_types:
    clpe = GLPE.CLPE(centrality_type, 
                    similarity,
                    better_pathway_data, 
                    heat_kernel_param = 100,
                    normalize_rows = False)
    clpe.fit(np.array(Z75_data))

    pathway_transition_matrix = pandas.DataFrame(data = clpe.pathway_transition_matrix_,
                                                index = clpe.pathway_names_, 
                                                columns = list(Z75_data.columns))

    pathway_transition_matrix.to_csv('/data4/mankovic/ZOETIS/pathway_ranking/Z75/fixed/Z75_pathway_matrix_'+similarity+'_'+centrality_type+'.csv')



    

