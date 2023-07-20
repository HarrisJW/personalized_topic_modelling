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
df['document_text'] = m.documents

fig = px.scatter(df, x="x", y="y", color="cluster_id", hover_data=['cluster_id', 'document_text'])
fig.show()

print("Testing complete.")