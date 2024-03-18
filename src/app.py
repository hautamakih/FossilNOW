from dash import Dash, html, dcc, callback, Output, Input
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.express as px
from utils.dataframe import *
from utils.scatter_mapbox import preprocess_data, create_map_figure, add_convex_hull_to_figure, create_histo


AGE_SPANS = ["0-0.1", "0.1-0.6", "0.6-1.1", "1.1-1.6", "1.6-2.1", "2.1-2.6"]

dent_genus = pd.read_csv("../data/DentalTraits_Genus_PPPA.csv")
dent_species = pd.read_csv("../data/DentalTraits_Species_PPPA.csv", encoding='unicode_escape')
mass_diet = pd.read_csv("../data/FossilGenera_MammalMassDiet_Jan24.csv")
sites = pd.read_csv("../data/AllSites_SiteOccurrences_AllGenera_26.1.24.csv")
rec = pd.read_csv("../data/Collaborative_filtering_data.csv")

species_in_sites, rec_species = preprocess_sites_df(sites, mass_diet, dent_genus, rec)

app = Dash(__name__)

app.layout = html.Div([
    html.H1(children='FossilNOW', style={'textAlign': 'center'}),
    dcc.Upload(
        id="upload-data",
        children=html.Div([
            "Drag and Drop or ",
            html.A("Select Files")
        ]),
    ),
    dcc.Store(id="data"),
    html.Div([
        html.Div([
            html.Label("Species"),
            dcc.Dropdown(id='dropdown-species'),
        ], className='one-third-column'),
        html.Div([
            html.Label("Age spans"),
            dcc.Dropdown(options=AGE_SPANS, value=AGE_SPANS[0], multi=True, id="age_span"),
        ], className='one-third-column'),
        html.Div([
            html.Label("Threshold"),
            dcc.Slider(min=0, max=1, step=0.1, value=0, id='threshold'),
        ], className='one-third-column'),
    ], className='row'),
    html.Div([
        html.Div([dcc.Graph(id='graph-content')], className='half-column'),
        html.Div([html.Div(id='site-info')], className='half-column'),
    ], className='row'),
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

@callback(
    Output("site-info", "children"),
    Input("graph-content", "clickData")
)
def update_site_info(clickData):
    if clickData is None:
        return html.P('Click on a site to view more information.')
    
    site_name, mass_bar_fig, dent_fig = create_histo(clickData, species_in_sites, rec_species)
    return [
        html.H3(f'Site: {site_name}'),
        dcc.Graph(id='site-bar-plot', figure=mass_bar_fig),
        dcc.Graph(id='site-dent-plot', figure=dent_fig)
    ]

if __name__ == '__main__':
    app.run(debug=True, port=8050)