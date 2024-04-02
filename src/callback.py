import dash
from dash import html, dcc, callback, Output, Input, State, dash_table
from dash.exceptions import PreventUpdate
import pandas as pd
import dash_mantine_components as dmc
from utils.dataframe import *
from utils.scatter_mapbox import preprocess_data, create_map_figure, add_convex_hull_to_figure, create_histo, add_top_n
from models.models import get_recommend_list_mf, get_recommend_list_knn

species_in_sites = pd.read_parquet("../data/species_in_sites.parquet")
rec_species = pd.read_parquet("../data/rec_species.parquet")
content_base = pd.read_csv("../data/content-based-filtering.csv")


def register_callbacks():
    @callback(
        Output("div-data", "style"),
        Output("div-visualization", "style"),
        Output("div-recommender", "style"),
        Input("tabs", "value"),
    )
    def render_content(tab):
        hide = dict(display="none")
        show = dict(display=True)
        
        if tab == "data":
            return show, hide, hide

        if tab == "visualization":
            return hide, show, hide
        
        if tab == "recommender-model":
            return hide, hide, show
    
    @callback(
        Output("div-datatables", "children"),
        Input("genera-occurrence-data", "data"),
        Input("genera-info-data", "data"),
        Input("sites-meta-data", "data"),
        Input("prediction-data", "data"),
    )
    def render_datatables(occ_df, sites_df, meta_df, pred_df):
        occ_df = pd.DataFrame(occ_df) if occ_df is not None else None
        sites_df = pd.DataFrame(sites_df) if sites_df is not None else None
        meta_df = pd.DataFrame(meta_df) if meta_df is not None else None
        pred_df = pd.DataFrame(pred_df) if pred_df is not None else None

        div_tables = html.Div([
            dash_table.DataTable(occ_df.to_dict("records"), [{"name": i, "id": i} for i in occ_df.columns[:10]], page_size= 10,) if occ_df is not None else "",
            dash_table.DataTable(sites_df.to_dict("records"), [{"name": i, "id": i} for i in sites_df.columns[:10]], page_size= 10,) if sites_df is not None else "",
            dash_table.DataTable(meta_df.to_dict("records"), [{"name": i, "id": i} for i in meta_df.columns[:10]], page_size= 10,) if meta_df is not None else "",
            dash_table.DataTable(pred_df.to_dict("records"), [{"name": i, "id": i} for i in pred_df.columns[:10]], page_size= 10,) if pred_df is not None else "",
        ])

        return div_tables

    @callback(
        Output("dropdown-species", "options"),
        Output("dropdown-species", "value"),
        Input("genera-occurrence-data", "data"),
    )
    def update_options(df):
        if df is None:
            raise PreventUpdate
        df = pd.DataFrame(df)
        return list(df.columns), df.columns[0]

    @callback(
        Output('graph-content', 'figure'),
        Output('notifications-container', 'children'),
        Input('dropdown-species', 'value'),
        Input('threshold', 'value'),
        State("prediction-data", "data"),
        State("sites-meta-data", "data"),
        Input("age_span", "value"),
        Input("n-highest-rec-scores", "value"),
        Input("div-visualization", "style"),
    )
    def update_graph(genera, threshold, df, sites_df, age_spans, n, viz_style):
        if df is None or sites_df is None or viz_style == dict(style="none"):
            raise PreventUpdate
        
        dff = pd.concat([pd.DataFrame(df), pd.DataFrame(sites_df)], axis=1)

        # Preprocess data
        gdff = preprocess_data(dff, genera, threshold)

        # Create map figure
        fig = create_map_figure(gdff, genera)

        # Add convex hull to the map if applicable
        errors = add_convex_hull_to_figure(fig, gdff, age_spans)
        
        # Add n highest recommendation scores

        if n and n > 0:
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

    @callback(
        Output("div-mf", "style"),
        Output("div-knn", "style"),
        Input("dropdown-algorithm", "value")
    )
    def render_params(algorithm):
        if algorithm == "Matrix Factorization":
            return dict(display=True), dict(display="none")
        
        if algorithm == "kNN":
            return dict(display="none"), dict(display=True)
        
    @callback(
        Output("prediction-data", "data"),
        Input("button-mf-run", "n_clicks"),
        Input("button-knn-run", "n_clicks"),
        State("input-mf-epochs", "value"),
        State("radio-mf-output-prob", "value"),
        State("radio-knn-output-prob", "value"),
        State("input-knn-k", "value"),
        State("dropdown-algorithm", "value"),
        State("genera-occurrence-data", "data"),
    )
    def run_recommender(n_clicks_mf, n_clicks_knn, epochs, output_prob_mf, output_prob_knn, k, model, df):
        if df is None:
            raise PreventUpdate
        dff = pd.DataFrame(df)

        if model == "Matrix Factorization":
            if n_clicks_mf == 0:
                raise PreventUpdate
            df_output = get_recommend_list_mf(dff, output_prob_mf, epochs)

        elif model == "kNN":
            if n_clicks_knn == 0:
                raise PreventUpdate
            df_output = get_recommend_list_knn(dff, output_prob_knn, k)

        return df_output.to_dict("records")