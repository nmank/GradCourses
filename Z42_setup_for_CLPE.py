import GLPE
#https://github.com/ekehoe32/orthrus
import sys
sys.path.append('/home/katrina/a/mankovic/ZOETIS/Fall2021/Orthrus/orthrus')
import orthrus
from orthrus import core
from orthrus.core import dataset
import numpy as np
# from NetworkDataAnalysis import graph_tools_construction as gt
from matplotlib import pyplot as plt
import pandas
# from orthrus.core.pipeline import *
from sklearn.preprocessing import FunctionTransformer
from orthrus.preprocessing.imputation import HalfMinimum
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from orthrus.core.helper import load_object

#load Kartikay's gene feature set and store RandIDs for genes
Z42_features = pandas.read_csv('/data4/mankovic/De-Identified_CZ/z42_features.csv', index_col = 0)
featureset_randIDs = [str(r) for r in list(Z42_features.index)]

#load pathway data
pathway_data = pandas.read_csv('/data4/mankovic/De-Identified_CZ/deidentified_fcpw.csv')

#load the entire Z42 dataset
ds = dataset.load_dataset('/data4/mankovic/De-Identified_CZ/DeId_TPM_C1_Z40_Z34_Z42.ds')
sample_ids = (ds.metadata['Project'] == 'Z42') & (ds.metadata['Time'] != -21)
Z42_dataset = ds.slice_dataset(sample_ids = sample_ids)

#normalize the dataset
transform = make_pipeline(HalfMinimum(missing_values=0), FunctionTransformer(np.log2))
Z42_dataset.normalize(transform, norm_name='HalfMinimum + Log2')

Z42_data = Z42_dataset.data

#restrict pathway data to to genes that are actually there
pathway_data = pathway_data[['RandID']+list(set(pathway_data.columns).intersection(set(Z42_data.columns)))]

#make a numpy array of genes and pathwayls fpr CLPE
better_pathway_data=pandas.DataFrame(columns = ['feature_id', 'pathway_id'])
gene_names = pathway_data.columns
for row in np.array(pathway_data):
    idx = np.where(row == True)
    # print(row)
    for g in gene_names[idx]:
        better_pathway_data = better_pathway_data.append({'feature_id': int(g), 'pathway_id':row[0]}, ignore_index = True)
better_pathway_data = np.array(better_pathway_data)


node_ids = np.unique(better_pathway_data[:,0])
translate_dict = { node_ids[i] :i  for i in range(len(node_ids))}
better_pathway_data[:,0] = np.vectorize(translate_dict.get)(better_pathway_data[:,0])


my_other_clpe = GLPE.CLPE('degree', 
                    'correlation',
                    better_pathway_data, 
                    heat_kernel_param = 2)
print(my_other_clpe)


#restrict to node ids within pathway file
small_dataset = np.array(Z42_data[node_ids.astype(str)])

#fit to dataset
print(my_other_clpe.fit(small_dataset))

#transform the small dataset for show
print(my_other_clpe.transform(small_dataset).shape)