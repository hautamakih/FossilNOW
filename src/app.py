from dash import Dash, html, dcc, callback, Output, Input
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.express as px
from utils.dataframe import *


app = Dash(__name__)

color_discrete_map = {'0': 'rgb(255,0,0)', '0.1': 'rgb(0,255,0)', '0.7': 'rgb(0,0,255)', '0.9': 'rgb(255,255,0)', '1': 'rgb(255,0,255)'}


app.layout=html.Div([
    html.H1(children='FossilNOW', style={'textAlign':'center'}),
    dcc.Upload(
        id="upload-data",
        children=html.Div(([
            "Drag and Drop or ",
            html.A("Select Files")
        ])),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
        },
    ),
    dcc.Store("data"),
    "Species",
    dcc.Dropdown([], id='dropdown-species'),
    "Threshold",
    dcc.Slider(0, 1, 0.1,value=0, id='threshold'),
    dcc.Graph(id='graph-content')
])

@callback(
    Output("dropdown-species", "options"),
    Output("dropdown-species", "value"),
    Input("data", "data"),
    prevent_initial_call=True
)
def update_options(df):
    df = pd.DataFrame(df)
    return list(df.columns), df.columns[0]

@callback(
    Output('graph-content', 'figure'),
    Input('dropdown-species', 'value'),
    Input('threshold', 'value'),
    Input("data", "data"),
)
def update_graph(genera, threshold, df):
    if df is None:
        raise PreventUpdate
    gdf = create_gdf(pd.DataFrame(df))
    gdff = gdf[[genera, "LATSTR", "LONGSTR", "geometry"]][gdf[genera] >= threshold]
    if len(gdff[genera].unique()) <= 8:
        gdff[genera] = gdff[genera].astype(str)
        fig = px.scatter_mapbox(gdff, lat=gdff.geometry.y, lon=gdff.geometry.x, mapbox_style="open-street-map", zoom=3, color=genera, color_discrete_map=color_discrete_map)    
    else:
        fig = px.scatter_mapbox(gdff, lat=gdff.geometry.y, lon=gdff.geometry.x, mapbox_style="open-street-map", zoom=3, color=genera)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_traces(marker={'size': 15, 'opacity': 0.6})
    return fig

@callback(
    Output("data", "data"),
    Input("upload-data", "contents")
)
def update_df(contents_list):
    if contents_list is None:
        raise PreventUpdate
    df = parse_contents(contents_list)
    return df.to_dict("records")

if __name__ == '__main__':
    app.run(debug=True)