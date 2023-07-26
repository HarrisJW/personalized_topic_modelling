from quant_eval import get_dataset
from model import MLModel
import plotly.express as px
import pandas as pd

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

DATASET, GROUND_TRUTH = get_dataset(config.get('dataset', '20newsgroups'))
model_path = config["model_path"]

# Convert data to lowercase.
for i, d in enumerate(DATASET):
    DATASET[i] = d.lower()

m = MLModel(DATASET, config)

m.run_all(model_path, config['min_cluster_size'])

df = pd.DataFrame(data = m.umap_document_embeddings_data_viz)
df.columns = ['x', 'y']
df['cluster_id'] = m.clusters.tolist()
#df['document_text'] = m.documents
df['document_text'] = [doc[0:30] for doc in m.documents] # Just the first 30 characters of each document

words = [i[0:5] for i in m.mostProbableWordsForAllDocumentsAsList["words"]]
probabilities = [i[0:5] for i in m.mostProbableWordsForAllDocumentsAsList["probabilities"]]

#Reference: https://plotly.com/python/hover-text-and-formatting/
#fig = px.scatter(df, x="x", y="y", color="cluster_id", hover_data=['cluster_id', 'document_text'])

fig = px.scatter(df, x="x", y="y", color="cluster_id", hover_data={'cluster_id': True,
                                                                   'document_text':True,
                                                                   'words': words,
                                                                   'probabilities': probabilities})



fig.show()


print("Testing complete.")