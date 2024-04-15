import dash
from dash import html, dcc, callback, Output, Input, State, dash_table, callback_context
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np
import dash_mantine_components as dmc
from utils.dataframe import *
from utils.scatter_mapbox import (
    preprocess_data,
    create_map_figure,
    add_convex_hull_to_figure,
    create_histo,
    add_top_n,
    add_column_and_average
)
from models.models import get_recommend_list_mf, get_recommend_list_knn

#species_in_sites = pd.read_parquet("../data/species_in_sites.parquet")
#rec_species = pd.read_parquet("../data/rec_species.parquet")
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

        div_tables = html.Div(
            [
                html.Div(
                    [
                        html.Label("Occurrence data: "),
                        dash_table.DataTable(
                            occ_df.to_dict("records"),
                            [{"name": i, "id": i, "hideable": True} for i in occ_df.columns],
                            hidden_columns=[i for i in occ_df.columns[10:]],
                            page_size=10,
                        )
                        if occ_df is not None
                        else "empty",
                    ], style={"margin-bottom": 10}
                ),
                html.Div(
                    [
                        html.Label("Sites data: "),
                        dash_table.DataTable(
                            meta_df.to_dict("records"),
                            [{"name": i, "id": i, "hideable": True} for i in meta_df.columns],
                            hidden_columns=[i for i in meta_df.columns[10:]],
                            page_size=10,
                        )
                        if meta_df is not None
                        else "empty",
                        html.Div([
                            "Extract N last meta data columns from the occurrence data: ",
                            dcc.Input(
                                id="n-metacolumns",
                                type="number",
                                placeholder="n-meta data columns",
                                value=0,
                            ),
                            html.Button(
                                "Extract", id="split-df-button", n_clicks=0,
                            ), 
                        ], style={"margin-bottom": 10})
                    ]
                ),
                html.Div(
                    [
                        html.Label("Genera data: "),
                        dash_table.DataTable(
                            sites_df.to_dict("records"),
                            [{"name": i, "id": i, "hideable": True} for i in sites_df.columns],
                            hidden_columns=[i for i in sites_df.columns[10:]],
                            page_size=10,
                        )
                        if sites_df is not None
                        else "empty",
                    ], style={"margin-bottom": 10}
                ),
                html.Div(
                    [
                        html.Label("Prediction data: "),
                        dash_table.DataTable(
                            pred_df.to_dict("records"),
                            [{"name": i, "id": i, "hideable": True} for i in pred_df.columns],
                            hidden_columns=[i for i in pred_df.columns[10:]],
                            page_size=10,
                        )
                        if pred_df is not None
                        else "empty",
                    ]
                ),
            ], id="data_tables"
        )

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
        Output("graph-content", "figure"),
        Output("notifications-container", "children"),
        Input("dropdown-species", "value"),
        Input("threshold", "value"),
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
            title="Warning!",
            action="show",
            message=f"For age spans {errors} convex hulls could not been generated, since there are less than 3 points",
            autoClose=4000,
            id="not-enough-points-notification",
            color="yellow",
        )

    @callback(
        Output("genera-occurrence-data", "data"),
        Output("genera-info-data", "data"),
        Output("sites-meta-data", "data"),
        Input("upload-data", "contents"),
        Input("split-df-button", "n_clicks"),
        State("n-metacolumns", "value"),
        State("df-dropdown", "value"),
        State("genera-occurrence-data", "data"),
    )
    def update_df(contents_list, n_clicks, n, df_type, occ_df):
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
<<<<<<< HEAD

        if triggered_id == "upload-data":
            if contents_list is None:
                raise PreventUpdate

=======

        if triggered_id == "upload-data":
            if contents_list is None:
                raise PreventUpdate

>>>>>>> 32cf3b59abd7ec0cd30c5e1b628363c27c370203
            df = parse_contents(contents_list)

            if df_type == "Genera occurrences at sites":
                return df.to_dict("records"), dash.no_update, dash.no_update

            if df_type == "Genera information":
                return dash.no_update, df.to_dict("records"), dash.no_update

        elif triggered_id == "split-df-button":
            if n is None or n == 0 or occ_df is None:
                raise PreventUpdate

            occ_df = pd.DataFrame(occ_df)

            return (
                occ_df.iloc[:, :-n].to_dict("records"),
                dash.no_update,
                occ_df.iloc[:, -n:].to_dict("records"),
            )

<<<<<<< HEAD

    @callback(
        Output("visualize-true-data", "data"),
        Output("visualize-recommendations-data", "data"),
        Input("genera-occurrence-data", "data"),
        Input("sites-meta-data","data"),
        Input("prediction-data", "data"),
        Input("genera-info-data", "data")
    )

    def visualization_data(occurences, site_data, recommendations, meta_data):  
        if occurences is None or site_data is None or recommendations is None or meta_data is None:
            raise PreventUpdate   
        #use genera_occurence_data, genera_inof_data and recommendation_data
        occurences = pd.DataFrame(occurences)
        site_data = pd.DataFrame(site_data)
        sites = site_data.iloc[:,-1:]
        recommendations = pd.DataFrame(recommendations)
        recommendations.insert(loc=0, column='SITE_NAME', value=sites)
        meta_data = pd.DataFrame(meta_data)
        if 'SITE_NAME' in occurences:
            site = 'SITE_NAME'
        elif 'NAME' in occurences:
            site = 'NAME'
        else:
            print('no site column')
        species_in_sites = occurences[[site]].copy()
        species_in_sites['genus_list'] = occurences.iloc[:, :100].apply(lambda row: list(row.index[row > 0]), axis=1)
        #meta_data
        meta_data = meta_data[['Genus', 'LogMass', 'HYP_Mean', 'LOP_Mean']]
        #mass:
        species_in_sites = add_column_and_average(species_in_sites,'LogMass', meta_data)
        #dental info
        meta_data['HYP_Mean'] = meta_data['HYP_Mean'].fillna(-1)
        meta_data['LOP_Mean'] = meta_data['LOP_Mean'].fillna(-1)
        species_in_sites = add_column_and_average(species_in_sites,'HYP_Mean',meta_data)
        species_in_sites = add_column_and_average(species_in_sites,'LOP_Mean',meta_data)
        #recommendations:
        if 'SITE_NAME' in recommendations:
            site = 'SITE_NAME'
        elif 'NAME' in recommendations:
            site = 'NAME'
        rec_species = recommendations[[site]].copy()
        rec_species['genus_list'] = recommendations.iloc[:, 1::].apply(lambda row: list(row.index[row > 0.5]), axis=1)
        rec_species = add_column_and_average(rec_species, 'LogMass', meta_data)
        rec_species = add_column_and_average(rec_species, 'HYP_Mean', meta_data)
        rec_species = add_column_and_average(rec_species, 'LOP_Mean', meta_data)
        rec_species = rec_species[[site, 'genus_list','LogMass','HYP_Mean','LOP_Mean']]
        return species_in_sites.to_dict("records"), rec_species.to_dict("records")

=======
>>>>>>> 32cf3b59abd7ec0cd30c5e1b628363c27c370203

    @callback(
        Output("site-info", "children"),
        Output("site-summary", "children"),
        Input("graph-content", "clickData"),
        State("visualize-true-data", "data"),
        State("visualize-recommendations-data", "data")
    )
    def update_site_info(clickData, species_in_sites, rec_species):
        if clickData is None:
            return html.P("Click on a site to view more information."), dash.no_update

        species_in_sites = pd.DataFrame(species_in_sites)
        rec_species = pd.DataFrame(rec_species)
        site_name, mass_bar_fig, dent_fig = create_histo(
            clickData, species_in_sites, rec_species
        )

        i = content_base[content_base["SITE_NAME"] == site_name].index.tolist()[0]
        recommendations = content_base.iloc[i, 1:].T.sort_values(ascending=False)[:10]

        recommendations_html = [
            html.Div(
                [
                    html.Li(str(index) + " " + str(np.round(value, 2)))
                    for index, value in recommendations.items()
                ]
            )
        ]

        return [
            html.H3(f"Site: {site_name}"),
            dcc.Graph(id="site-bar-plot", figure=mass_bar_fig),
            dcc.Graph(id="site-dent-plot", figure=dent_fig),
        ], recommendations_html

    @callback(
        Output("div-mf", "style"),
        Output("div-knn", "style"),
        Input("dropdown-algorithm", "value"),
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
    def run_recommender(
        n_clicks_mf, n_clicks_knn, epochs, output_prob_mf, output_prob_knn, k, model, df
    ):
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
