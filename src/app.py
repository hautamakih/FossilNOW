from dash import Dash, html, dcc, callback, Output, Input
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.express as px
from utils.dataframe import *
from utils.scatter_mapbox import preprocess_data, create_map_figure, add_convex_hull_to_figure

AGE_SPANS = ["0-0.1", "0.1-0.6", "0.6-1.1", "1.1-1.6", "1.6-2.1", "2.1-2.6"]


app = Dash(__name__)

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
    "Age spans",
    dcc.Dropdown(AGE_SPANS, value=AGE_SPANS[0], multi=True, id="age_span"),
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
    return list(df.columns), df.columns[7]

@callback(
    Output('graph-content', 'figure'),
    Input('dropdown-species', 'value'),
    Input('threshold', 'value'),
    Input("data", "data"),
    Input("age_span", "value"),
)
def update_graph(genera, threshold, df, age_spans):
    if df is None:
        raise PreventUpdate
    
    # Preprocess data
    gdff = preprocess_data(df, genera, threshold)

    # Create map figure
    fig = create_map_figure(gdff, genera)

    # Add convex hull to the map if applicable
    if gdff.shape[0] >= 3:
        add_convex_hull_to_figure(fig, gdff, age_spans)
    
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