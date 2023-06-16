#Dash tutorial at:
#https://dash.plotly.com/tutorial

from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px

app = Dash(__name__)

top2VecTopics = ["cats", "dogs", "trees"]
top2VecProbablities = [0.15, 0.26, 0.35]

top2VecData = {"topics": top2VecTopics, "probabilities": top2VecProbablities}

probTop2VecTopics = ["bats", "deer", "shrubs"]
probTop2VecProbablities = [0.88, 0.55, 0.80]

probTop2VecVecData = {"topics": probTop2VecTopics, "probabilities": probTop2VecProbablities}

data={"probTop2Vec":  probTop2VecVecData,
     "top2Vec": top2VecData}

df = pd.DataFrame(data=data["probTop2Vec"])

app.layout = html.Div([
    html.Div(children='Topic Modeling'),
    html.Hr(),
    dash_table.DataTable(data=df.to_dict('records'), page_size=10),
    dcc.RadioItems(options=['top2Vec', 'probTop2Vec'], value='top2Vec', id='model-selector-controls'),
    dcc.Graph(figure={},id='graph')
])

@callback(
    Output(component_id='graph', component_property='figure'),
    Input(component_id='model-selector-controls', component_property='value')
)
    
def update_graph(model_chosen):
    fig = px.histogram(data[model_chosen], x='topics', y='probabilities')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)