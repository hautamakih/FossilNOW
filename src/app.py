import dash_mantine_components as dmc
import dash_daq as daq
import dash
from dash import Dash, html, dcc, callback, Output, Input, State
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.express as px
from utils.dataframe import *
from utils.scatter_mapbox import preprocess_data, create_map_figure, add_convex_hull_to_figure, create_histo, add_top_n
import numpy as np


AGE_SPANS = ["0-0.1", "0.1-0.6", "0.6-1.1", "1.1-1.6", "1.6-2.1", "2.1-2.6"]

species_in_sites = pd.read_parquet("../data/species_in_sites.parquet")
rec_species = pd.read_parquet("../data/rec_species.parquet")
content_base = pd.read_csv("../data/content-based-filtering.csv")

app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = dmc.NotificationsProvider(
    html.Div([
        html.Div(id='notifications-container'),
        html.H1(children='FossilNOW', style={'textAlign': 'center'}),
        html.Div([
            dcc.Upload(
                id="upload-data",
                children=html.Div([
                    "Drag and Drop or ",
                    html.A("Select Files")
                ]),
            ),
            dcc.Dropdown(id="df-dropdown", options=["Genera occurrences at sites", "Genera information"], value="Genera occurrences at sites", clearable=False),
            dcc.Input(
                id="n-metacolumns",
                type="number",
                placeholder="n-meta data columns",
                value=0,
            ),
        ]),
        dcc.Store(id="genera-occurrence-data"),
        dcc.Store(id="genera-info-data"),
        dcc.Store(id="sites-meta-data"),
        dcc.Store(id="prediction-data"),
        dcc.Tabs(id="tabs", value="visualization", children=[
            dcc.Tab(label="Visualization", value="visualization"),
            dcc.Tab(label="Recommender systems", value="recommender-model")
        ]),
        html.Div(id="tab-content"),
    ]), position='top-right'
)

@callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
)
def render_content(tab):
    if tab == "visualization":
        return html.Div([
            html.Div([
                html.Div([
                    html.Label("Genus"),
                    dcc.Dropdown(id='dropdown-species', clearable=False),
                ], className='one-fourth-column'),
                html.Div([
                    html.Label("Age spans"),
                    dcc.Dropdown(options=AGE_SPANS, value=AGE_SPANS[0], multi=True, id="age_span"),
                ], className='one-fourth-column'),
                html.Div([
                    html.Label("Threshold"),
                    dcc.Slider(min=0, max=1, step=0.1, value=0, id='threshold'),
                ], className='one-fourth-column'),
                html.Div([
                    html.Label("n highest recommendation scores"),
                    dcc.Input(
                        id="n-highest-rec-scores",
                        value=0,
                        type="number"
                    )
                ], className='n-highest')
            ], className='row'),
            html.Div([
                html.Div([dcc.Graph(id='graph-content'), html.Div([], id='site-summary')], className='half-column'),
                html.Div([html.Div(id='site-info')], className='half-column'),
            ], className='row'),
        ])

@callback(
    Output("dropdown-species", "options"),
    Output("dropdown-species", "value"),
    Input("genera-occurrence-data", "data"),
)
def update_options(df):
    if df is None:
        raise PreventUpdate
    df = pd.DataFrame(df)
    return list(df.columns), df.columns[7]

@callback(
    Output('graph-content', 'figure'),
    Output('notifications-container', 'children'),
    Input('dropdown-species', 'value'),
    Input('threshold', 'value'),
    Input("genera-occurrence-data", "data"),
    Input("sites-meta-data", "data"),
    Input("age_span", "value"),
    Input("n-highest-rec-scores", "value"),
)
def update_graph(genera, threshold, df, sites_df, age_spans, n):
    if df is None:
        raise PreventUpdate
    
    dff = pd.concat([pd.DataFrame(df), pd.DataFrame(sites_df)], axis=1)
    
    # Preprocess data
    gdff = preprocess_data(dff, genera, threshold)

    # Create map figure
    fig = create_map_figure(gdff, genera)

    # Add convex hull to the map if applicable
    errors = add_convex_hull_to_figure(fig, gdff, age_spans)
    
    # Add n highest recommendation scores

    if n > 0:
        add_top_n(gdff, genera, n, fig)

    if len(errors) == 0:
        return fig, dash.no_update

    return fig, dmc.Notification(
        title='Warning!',
        action='show',
        message=f'For age spans {errors} convex hulls could not been generated, since there are less than 3 points',
        autoClose=4000,
        id='not-enough-points-notification',
        color='yellow',
    )

@callback(
    Output("genera-occurrence-data", "data"),
    Output("genera-info-data", "data"),
    Output("sites-meta-data", "data"),
    Input("upload-data", "contents"),
    State("df-dropdown", "value"),
    State("n-metacolumns", "value")
)
def update_df(contents_list, df_type, n_meta):
    if contents_list is None:
        raise PreventUpdate
    
    df = parse_contents(contents_list)

    if df_type == "Genera occurrences at sites":
        if n_meta == 0:
            return df.to_dict("records"), dash.no_update, dash.no_update
        occ_df = df.iloc[:, :-n_meta]
        sites_meta_df = df.iloc[:, -n_meta:]
        return occ_df.to_dict("records"), dash.no_update, sites_meta_df.to_dict("records")

    if df_type == "Genera information":
        return dash.no_update, df.to_dict("records"), dash.no_update

@callback(
    Output("site-info", "children"),
    Output("site-summary", "children"),
    Input("graph-content", "clickData")
)
def update_site_info(clickData):
    if clickData is None:
        return html.P('Click on a site to view more information.'), dash.no_update
    
    site_name, mass_bar_fig, dent_fig = create_histo(clickData, species_in_sites, rec_species)
    
    i = content_base[content_base["SITE_NAME"]==site_name].index.tolist()[0]
    recommendations = content_base.iloc[i, 1:].T.sort_values(ascending=False)[:10]

    recommendations_html = [
        html.Div([
            html.Li(str(index) + " " + str(value)) for index, value in recommendations.items()
        ])
    ]

    return [
        html.H3(f'Site: {site_name}'),
        dcc.Graph(id='site-bar-plot', figure=mass_bar_fig),
        dcc.Graph(id='site-dent-plot', figure=dent_fig)
    ], recommendations_html

if __name__ == '__main__':
    app.run(debug=True, port=8010)