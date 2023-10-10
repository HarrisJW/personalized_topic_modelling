from quant_eval import get_dataset
from model import MLModel
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go

config = {'num_iters': 50,
          'model_path': 'sentence-transformers/all-MiniLM-L6-v2',
          'min_cluster_size': 20,
          'clus_method': 'gm',
          'epoch': 1,
          'loss': 'cos',
          'umap': True,
          'workdir': 'workdirs/run_testv1v2gm_1_cos_False',
          'oracle': 'v1v2',
          'dataset': 'test'}

#target_dataset = config.get('dataset', '20newsgroups')
target_dataset = '20newsgroups'
DATASET, GROUND_TRUTH = get_dataset(target_dataset)

#Get a sample subset of data points for faster development/testing.
DATASET_DF = pd.DataFrame(DATASET)
NUM_SAMPLES = 500
sample_indices = pd.Index(np.random.choice(DATASET_DF.index, NUM_SAMPLES))

SAMPLE_DATASET = [DATASET[i] for i in sample_indices]
SAMPLE_GROUND_TRUTH = [GROUND_TRUTH[i] for i in sample_indices]

model_path = config["model_path"]

# Convert data to lowercase.
for i, d in enumerate(SAMPLE_DATASET):
    SAMPLE_DATASET[i] = d.lower()

m = MLModel(SAMPLE_DATASET, config)

m.run_all(model_path, config['min_cluster_size'])

df = pd.DataFrame(data=m.umap_document_embeddings_data_viz)
df.columns = ['x', 'y']
df['cluster_id'] = m.clusters.tolist()
#df['document_text'] = m.documents
df['document_text'] = [doc[0:30] for doc in m.documents] # Just the first 30 characters of each document

words = [i[0:5] for i in m.documentWordProbabilitiesForVisualization["words"]]
probabilities = [i[0:5] for i in m.documentWordProbabilitiesForVisualization["probabilities"]]

#Reference: https://plotly.com/python/hover-text-and-formatting/
#fig = px.scatter(df, x="x", y="y", color="cluster_id", hover_data=['cluster_id', 'document_text'])

fig = px.scatter(df, x="x", y="y", color="cluster_id", hover_data={'cluster_id': True,
                                                                   'document_text':True,
                                                                   'words': words,
                                                                   'probabilities': probabilities})
#Plot Topic Vectors
# TODO: Topic vectors do not share same space as document vectors. I expected a given topic to be plotted in the centre of its documents.
# Do I need to calculate document and topic vectors in same step?
#topic_vectors_df = pd.DataFrame(m.umap_topic_embeddings_data_viz, columns = ['x','y'])

#fig.add_trace(go.Scatter(x=topic_vectors_df['x'],
#                            y=topic_vectors_df['y'],
#                            mode="markers",
#                            marker=dict(size=20),
#                            showlegend=False))

fig.show()


print("Testing complete.")