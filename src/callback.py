import uuid
import dash
from dash import html, dcc, callback, Output, Input, State, dash_table, callback_context
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np
import dash_mantine_components as dmc
import plotly.express as px
from utils.dataframe import *
from utils.scatter_mapbox import (
    preprocess_data,
    create_map_figure,
    add_convex_hull_to_figure,
    create_histo,
    add_top_n,
    add_column_and_average,
    add_true_occurrences,
)
from models.models import (
    get_recommend_list_mf,
    get_recommend_list_knn,
    get_recommend_list_content_base,
    get_recommend_list_colab,
    get_recommend_list_hybrid,
)

# species_in_sites = pd.read_parquet("../data/species_in_sites.parquet")
# rec_species = pd.read_parquet("../data/rec_species.parquet")
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
                            [
                                {"name": i, "id": i, "hideable": True}
                                for i in occ_df.columns
                            ],
                            hidden_columns=[i for i in occ_df.columns[10:]],
                            page_size=10,
                        )
                        if occ_df is not None
                        else "empty",
                    ],
                    style={"margin-bottom": 10},
                ),
                html.Div(
                    [
                        html.Label("Sites data: "),
                        dash_table.DataTable(
                            meta_df.to_dict("records"),
                            [
                                {"name": i, "id": i, "hideable": True}
                                for i in meta_df.columns
                            ],
                            hidden_columns=[i for i in meta_df.columns[10:]],
                            page_size=10,
                        )
                        if meta_df is not None
                        else "empty",
                        html.Div(
                            [
                                "Extract N last meta data columns from the occurrence data: ",
                                dcc.Input(
                                    id="n-metacolumns",
                                    type="number",
                                    placeholder="n-meta data columns",
                                    value=0,
                                ),
                                html.Button(
                                    "Extract",
                                    id="split-df-button",
                                    n_clicks=0,
                                ),
                            ],
                            style={"margin-bottom": 10},
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Genera data: "),
                        dash_table.DataTable(
                            sites_df.to_dict("records"),
                            [
                                {"name": i, "id": i, "hideable": True}
                                for i in sites_df.columns
                            ],
                            hidden_columns=[i for i in sites_df.columns[10:]],
                            page_size=10,
                        )
                        if sites_df is not None
                        else "empty",
                    ],
                    style={"margin-bottom": 10},
                ),
                html.Div(
                    [
                        html.Label("Prediction data: "),
                        dash_table.DataTable(
                            pred_df.to_dict("records"),
                            [
                                {"name": i, "id": i, "hideable": True}
                                for i in pred_df.columns
                            ],
                            hidden_columns=[i for i in pred_df.columns[10:]],
                            page_size=10,
                        )
                        if pred_df is not None
                        else "empty",
                    ]
                ),
            ],
            id="data_tables",
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
        State("genera-occurrence-data", "data"),
        Input("age_span", "value"),
        Input("n-highest-rec-scores", "value"),
        Input("div-visualization", "style"),
    )
    def update_graph(
        genera, threshold, pred_df, sites_df, occ_df, age_spans, n, viz_style
    ):
        if pred_df is None or sites_df is None or viz_style == dict(style="none"):
            raise PreventUpdate

        dff = pd.concat([pd.DataFrame(pred_df), pd.DataFrame(sites_df)], axis=1)

        # Preprocess data
        gdff = preprocess_data(dff, genera, threshold)

        # Create map figure

        fig = create_map_figure(gdff, genera)

        # Add convex hull to the map if applicable
        errors = add_convex_hull_to_figure(fig, gdff, age_spans)

        # Add n highest recommendation scores

        if n and n > 0:
            add_top_n(gdff, genera, n, fig)

        add_true_occurrences(fig, occ_df, sites_df, genera)

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

        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if triggered_id == "upload-data":
            if contents_list is None:
                raise PreventUpdate

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

    @callback(
        Output("visualize-true-data", "data"),
        Output("visualize-recommendations-data", "data"),
        Input("genera-occurrence-data", "data"),
        Input("sites-meta-data", "data"),
        Input("prediction-data", "data"),
        Input("genera-info-data", "data"),
        Input("threshold", "value"),
    )
    def visualization_data(
        occurences, site_data, recommendations, meta_data, threshold
    ):
        if (
            occurences is None
            or site_data is None
            or recommendations is None
            or meta_data is None
            or threshold is None
        ):
            raise PreventUpdate
        # use genera_occurence_data, genera_inof_data and recommendation_data
        occurences = pd.DataFrame(occurences)
        site_data = pd.DataFrame(site_data)
        if "SITE_NAME" in site_data.columns:
            site = "SITE_NAME"
        elif "NAME" in site_data.columns:
            site = "NAME"
        else:
            print("no site column")
        sites = site_data[[site]]
        recommendations = pd.DataFrame(recommendations)
        recommendations.insert(loc=0, column="SITE_NAME", value=sites)
        occurences.insert(loc=0, column="SITE_NAME", value=sites)
        meta_data = pd.DataFrame(meta_data)
        # print(occurences.columns)
        species_in_sites = occurences[["SITE_NAME"]].copy()
        species_in_sites["genus_list"] = occurences.iloc[:, 1:].apply(
            lambda row: list(row.index[row > 0]), axis=1
        )
        # meta_data
        meta_data = meta_data[["Genus", "LogMass", "HYP_Mean", "LOP_Mean"]]
        # mass:
        species_in_sites = add_column_and_average(
            species_in_sites, "LogMass", meta_data
        )
        # dental info
        meta_data["HYP_Mean"] = meta_data["HYP_Mean"].fillna(-1)
        meta_data["LOP_Mean"] = meta_data["LOP_Mean"].fillna(-1)
        species_in_sites = add_column_and_average(
            species_in_sites, "HYP_Mean", meta_data
        )
        species_in_sites = add_column_and_average(
            species_in_sites, "LOP_Mean", meta_data
        )
        # recommendations:
        rec_species = recommendations[["SITE_NAME"]].copy()
        rec_species["genus_list"] = recommendations.iloc[:, 1::].apply(
            lambda row: list(row.index[row > threshold]), axis=1
        )

        def score_tuples(row):
            return [(col, score) for col, score in row.items() if score > threshold]

        rec_species["scores"] = recommendations.iloc[:, 1:].apply(score_tuples, axis=1)
        rec_species = add_column_and_average(rec_species, "LogMass", meta_data)
        rec_species = add_column_and_average(rec_species, "HYP_Mean", meta_data)
        rec_species = add_column_and_average(rec_species, "LOP_Mean", meta_data)
        rec_species = rec_species[
            ["SITE_NAME", "genus_list", "scores", "LogMass", "HYP_Mean", "LOP_Mean"]
        ]
        return species_in_sites.to_dict("records"), rec_species.to_dict("records")

    @callback(
        Output("site-info", "children"),
        Output("site-summary", "children"),
        Input("graph-content", "clickData"),
        Input("visualize-true-data", "data"),
        Input("visualize-recommendations-data", "data"),
    )
    def update_site_info(clickData, species_in_sites, rec_species):
        if species_in_sites is None or rec_species is None:
            raise PreventUpdate
        if clickData is None:
            return html.P("Click on a site to view more information."), dash.no_update

        species_in_sites = pd.DataFrame(species_in_sites)
        rec_species = pd.DataFrame(rec_species)
        site_name, mass_bar_fig, dent_fig, true_occ, rec_occ = create_histo(
            clickData, species_in_sites, rec_species
        )

        rec_occ = sorted(rec_occ, key=lambda x: x[1], reverse=True)
        recommendations_html = [
            html.Div(
                children=[
                    html.P(
                        "Recommendation scores and true occurences",
                        style={"text-align": "center"},
                    ),
                    html.Div(
                        children=[
                            html.Li(str(gen) + " " + str(np.round(scr, 2)))
                            for gen, scr in rec_occ
                        ],
                        style={"width": "50%", "float": "left"},
                    ),
                    # html.P("True occurrences", style={"margin-left": "50%"}),
                    html.Div(
                        children=[html.Li(str(gen)) for gen in true_occ],
                        style={"margin-left": "50%"},
                    ),
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
        Output("div-content", "style"),
        Output("div-collab", "style"),
        Output("div-hybrid", "style"),
        Input("dropdown-algorithm", "value"),
    )
    def render_params(algorithm):
        hide = dict(display="none")
        show = dict(display=True)
        if algorithm == "Matrix Factorization":
            return show, hide, hide, hide, hide

        if algorithm == "kNN":
            return hide, show, hide, hide, hide
        if algorithm == "Content-based Filtering":
            return hide, hide, show, hide, hide
        if algorithm == "Collaborative Filtering":
            return hide, hide, hide, show, hide
        if algorithm == "Hybrid: Content-based x Collaborative":
            return hide, hide, hide, hide, show

    @callback(
        Output("button-mf-run", "value"),
        # Output("button-knn-run", "value"),
        # Output("button-content-run", "value"),
        Output("notification-mf", "children"),
        Input("button-mf-run", "n_clicks"),
        Input("button-knn-run", "n_clicks"),
        Input("button-content-run", "n_clicks"),
        Input("button-collab-run", "n_clicks"),
        Input("button-hybrid-run", "n_clicks"),
        State("recommender-compute-done", "children"),
        State("button-mf-run", "value"),
    )
    def start_recommender(
        n_clicks_mf,
        n_clicks_knn,
        n_clicks_content,
        n_clicks_collab,
        n_clicks_hybrid,
        done,
        doing,
    ):
        if (
            n_clicks_mf == 0
            and n_clicks_knn == 0
            and n_clicks_content == 0
            and n_clicks_collab == 0
            and n_clicks_hybrid == 0
        ):
            raise PreventUpdate
        if done != doing:
            return dash.no_update, dmc.Notification(
                id=str(uuid.uuid4()),
                color="yellow",
                title="Warning",
                message=("The recommender system computing has not yet finished."),
                action="show",
                autoClose=10000,
            )

        process_id = str(uuid.uuid4())

        message = "The recommender system process has started."

        return process_id, dmc.Notification(
            id=process_id,
            color="blue",
            title="Processing",
            message=message,
            action="show",
            loading=True,
            autoClose=False,
            disallowClose=True,
        )

    @callback(
        Output("prediction-data", "data"),
        Output("recommender-compute-done", "children"),
        Output("notification-mf-done", "children"),
        Input("button-mf-run", "value"),
        Input("button-knn-run", "value"),
        Input("button-content-run", "value"),
        Input("button-collab-run", "value"),
        Input("button-hybrid-run", "value"),
        State("test-train-split", "value"),
        State("input-mf-epochs", "value"),
        State("input-mf-dim-hid", "value"),
        State("radio-mf-output-prob", "value"),
        State("radio-knn-output-prob", "value"),
        State("input-knn-k", "value"),
        State("input-collab-k", "value"),
        State("input-collab-min_k", "value"),
        State("input-hybrid-k", "value"),
        State("input-hybrid-min_k", "value"),
        State("input-hybrid-weight", "value"),
        State("input-hybrid-threshold", "value"),
        State("method-hybrid", "value"),
        State("dropdown-algorithm", "value"),
        State("genera-occurrence-data", "data"),
        State("genera-info-data", "data"),
        State("sites-meta-data", "data"),
    )
    def run_recommender(
        process_id,
        n_clicks_knn,
        n_clicks_content,
        n_clicks_collab,
        n_clicks_hybrid,
        test_train_split,
        epochs,
        dim_hid,
        output_prob_mf,
        output_prob_knn,
        k,
        k_collab,
        min_k_collab,
        k_hybrid,
        min_k_hybrid,
        weight_hybrid,
        threshold_hybrid,
        hybrid_method,
        model,
        df,
        genera,
        sites,
    ):
        if df is None or sites is None:
            raise PreventUpdate
        dff = pd.DataFrame(df)
        sites = pd.DataFrame(sites)
        site_name = "SITE_NAME" if "SITE_NAME" in sites.columns else "NAME"
        if sites.columns[0] != site_name:
            site_name_data = sites[site_name]
            sites = sites.drop([site_name], axis=1)
            sites.insert(loc=0, column=site_name, value=site_name_data)

        if model == "Matrix Factorization":
            df_output = get_recommend_list_mf(dff, output_prob_mf, epochs, dim_hid)

        elif model == "kNN":
            if n_clicks_knn == 0:
                raise PreventUpdate
            df_output = get_recommend_list_knn(dff, output_prob_knn, k)

        elif model == "Content-based Filtering":
            if n_clicks_content == 0 or genera is None:
                raise PreventUpdate
            genera = pd.DataFrame(genera)

            dff.insert(loc=0, column="SITE_NAME", value=sites[site_name])

            # if 'COUNTRY' in sites.columns:
            sites = sites.drop(["COUNTRY"], axis=1)
            dff.insert(loc=0, column="SITE_NAME", value=sites[sites.columns[0]])

            df_output = get_recommend_list_content_base(dff, sites, genera)

        elif model == "Collaborative Filtering":
            if n_clicks_collab == 0:
                raise PreventUpdate
            dff.insert(loc=0, column="SITE_NAME", value=sites[site_name])
            df_output = get_recommend_list_colab(dff, k_collab, min_k_collab)

        elif model == "Hybrid: Content-based x Collaborative":
            if n_clicks_hybrid == 0:
                raise PreventUpdate
            dff.insert(loc=0, column="SITE_NAME", value=sites[site_name])
            sites = sites.drop(["COUNTRY"], axis=1)
            df_output = get_recommend_list_hybrid(
                dff,
                sites,
                genera,
                k=k_hybrid,
                min_k=min_k_hybrid,
                method=hybrid_method,
                content_based_weight=weight_hybrid,
                filter_threshold=threshold_hybrid,
            )

        return (
            df_output.to_dict("records"),
            process_id,
            dmc.Notification(
                id=process_id or str(uuid.uuid4()),
                color="green",
                title="Success",
                message="The recommendation scores computed successfully!",
                action="update" if process_id else "show",
                autoClose=5000,
                disallowClose=False,
            ),
        )
