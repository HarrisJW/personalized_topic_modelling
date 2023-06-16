#Dash tutorial at:
#https://dash.plotly.com/tutorial

from dash import Dash, html, dash_table, dcc, callback, Output, Input

app = Dash(__name__)

app.layout = html.Div([
    html.Div(children='Hello World')
])

if __name__ == '__main__':
    app.run_server(debug=True)