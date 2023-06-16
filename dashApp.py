#Dash tutorial at:
#https://dash.plotly.com/tutorial

from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd

app = Dash(__name__)

topics = ["cats", "dogs", "trees"]
probablities = [0.15, 0.26, 0.35]
data = {"topics": topics, "probabilities": probablities}

df = pd.DataFrame(data=data)

app.layout = html.Div([
    html.Div(children='Topic Modeling'),
    dash_table.DataTable(data=df.to_dict('records'), page_size=10)
])

if __name__ == '__main__':
    app.run_server(debug=True)