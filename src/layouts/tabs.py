from dash import dcc, html

def get_tabs():
    return dcc.Tabs(
        id="tabs",
        value="data",
        children=[
            get_data_tab(),
            get_recommender_tab(),
            get_viz_tab(),
        ],
    )

def get_data_tab():
    return dcc.Tab(
                label="Data",
                value="data",
                children=[
                    html.Div(
                        [
                            # Istructions for data tab:
                            html.H2(
                                children="Instructions",
                                style={
                                    "textAlign": "center",
                                    "font-size": "medium",
                                },
                            ),
                            html.P(
                                "Upload data and select the number of metadata columns (these should be at the end of the data)"
                            ),
                            html.P(
                                "The site name column should be named as 'SITE_NAME' or 'NAME'"
                            ),
                            html.P(
                                "The Genera information data should contain columns: 'Genus', 'LogMass', 'HYP_Mean' and 'LOP_Mean'"
                            ),
                            html.P(
                                "Categorical values in Site and Genera information data should be one-hot-encoded (dummy variables)"
                            ),
                            html.P(
                                "When using Conten-based or hybrid algorithm, the occurence data should not contain any extra data apart for the columns that are used for the predictions."
                            ),
                        ]
                    )
                ],
            )

def get_recommender_tab():
    return dcc.Tab(
                label="Recommender systems",
                value="recommender-model",
                children=[
                    html.Div(
                        [
                            # Instructions for recommender tab:
                            html.H2(
                                children="Instructions",
                                style={
                                    "textAlign": "center",
                                    "font-size": "medium",
                                },
                            ),
                            html.P(
                                "Select the wanted recommender system algorithm and the parameters"
                            ),
                        ]
                    )
                ],
            )

def get_viz_tab():
    return dcc.Tab(
                label="Visualization",
                value="visualization",
                children=[
                    html.Div(
                        [
                            # Instructions for visulaization tab:
                            html.H2(
                                children="1. Instructions for the map:",
                                style={
                                    "textAlign": "center",
                                    "font-size": "medium",
                                },
                            ),
                            html.Ul(
                                style={
                                    "text-align": "center",
                                    "list-style-position": "inside",
                                },
                                children=[
                                    html.Li(
                                        "Select the Genus, age spans and threshold."
                                    ),
                                    html.Li(
                                        "The map will show you convex hulls for each timespan fro the selected genus."
                                    ),
                                ],
                            ),
                            html.H2(
                                children="2. Instructions for the histograms:",
                                style={
                                    "textAlign": "center",
                                    "font-size": "medium",
                                },
                            ),
                            html.Ul(
                                style={
                                    "text-align": "center",
                                    "list-style-position": "inside",
                                },
                                children=[
                                    html.Li("Select a site from the map."),
                                    html.Li(
                                        "Get histograms about the logmass, hypsodonty, and loph of the true and recommended genera on that site."
                                    ),
                                    html.Li("-1 values in histograms represents missing values in the genera data")
                                ],
                            ),
                        ]
                    )
                ],
            )