import dash_mantine_components as dmc
from dash import html, dcc

AGE_SPANS = ["0-0.1", "0.1-0.6", "0.6-1.1", "1.1-1.6", "1.6-2.1", "2.1-2.6"]


def get_layout():
    return dmc.NotificationsProvider(
        html.Div(
            [
                html.Div(id="notifications-container"),
                html.H1(children="FossilNOW", style={"textAlign": "center"}),
                dcc.Store(id="genera-occurrence-data"),
                dcc.Store(id="genera-info-data"),
                dcc.Store(id="sites-meta-data"),
                dcc.Store(id="prediction-data"),
                dcc.Store(id="visualize-recommendations-data"),
                dcc.Store(id="visualize-true-data"),
                dcc.Tabs(
                    id="tabs",
                    value="data",
                    children=[
                        dcc.Tab(
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
                                            "Select the number of metadata columns (these should be at the end of the data) and upload data"
                                        ),
                                    ]
                                )
                            ],
                        ),
                        dcc.Tab(
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
                        ),
                        dcc.Tab(
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
                                            ],
                                        ),
                                    ]
                                )
                            ],
                        ),
                    ],
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                dcc.Upload(
                                    id="upload-data",
                                    children=html.Div(
                                        ["Drag and Drop or ", html.A("Select Files")]
                                    ),
                                ),
                                dcc.Dropdown(
                                    id="df-dropdown",
                                    options=[
                                        "Genera occurrences at sites",
                                        "Genera information",
                                    ],
                                    value="Genera occurrences at sites",
                                    clearable=False,
                                ),
                                dcc.Input(
                                    id="n-metacolumns",
                                    type="number",
                                    placeholder="n-meta data columns",
                                    value=0,
                                    style={"display": "none"}
                                ),
                                html.Div(id="div-datatables"),
                                html.Button(
                                    "Extract", id="split-df-button", n_clicks=0, style={"display": "none"}
                                ),
                            ],
                            id="div-data",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label("Genus"),
                                                dcc.Dropdown(
                                                    id="dropdown-species",
                                                    clearable=False,
                                                ),
                                            ],
                                            className="one-fourth-column",
                                        ),
                                        html.Div(
                                            [
                                                html.Label("Age spans"),
                                                dcc.Dropdown(
                                                    options=AGE_SPANS,
                                                    value=AGE_SPANS[0],
                                                    multi=True,
                                                    id="age_span",
                                                ),
                                            ],
                                            className="one-fourth-column",
                                        ),
                                        html.Div(
                                            [
                                                html.Label("Threshold"),
                                                dcc.Slider(
                                                    min=0,
                                                    max=1,
                                                    step=0.1,
                                                    value=0,
                                                    id="threshold",
                                                ),
                                            ],
                                            className="one-fourth-column",
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "n highest recommendation scores"
                                                ),
                                                dcc.Input(
                                                    id="n-highest-rec-scores",
                                                    value=0,
                                                    type="number",
                                                ),
                                            ],
                                            className="n-highest",
                                        ),
                                    ],
                                    className="row",
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                dcc.Graph(id="graph-content"),
                                                html.Div([], id="site-summary"),
                                            ],
                                            className="half-column",
                                        ),
                                        html.Div(
                                            [html.Div(id="site-info")],
                                            className="half-column",
                                        ),
                                    ],
                                    className="row",
                                ),
                            ],
                            id="div-visualization",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label("Choose algorithm"),
                                        dcc.Dropdown(
                                            id="dropdown-algorithm",
                                            options=[
                                                "Matrix Factorization",
                                                "kNN",
                                                "Content-based Filtering",
                                            ],
                                            value="Matrix Factorization",
                                            clearable=False,
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Label("Epochs"),
                                                        dcc.Input(
                                                            id="input-mf-epochs",
                                                            value=100,
                                                            type="number",
                                                        ),
                                                    ]
                                                ),
                                                html.Div(
                                                    [
                                                        html.Label(
                                                            "Output probabilities"
                                                        ),
                                                        dcc.RadioItems(
                                                            ["Yes", "No"],
                                                            "Yes",
                                                            id="radio-mf-output-prob",
                                                        ),
                                                    ]
                                                ),
                                                html.Div(
                                                    [
                                                        html.Button(
                                                            "Run",
                                                            id="button-mf-run",
                                                            n_clicks=0,
                                                        )
                                                    ]
                                                ),
                                            ],
                                            id="div-mf",
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Label("Top k"),
                                                        dcc.Input(
                                                            id="input-knn-k",
                                                            value=15,
                                                            type="number",
                                                        ),
                                                    ]
                                                ),
                                                html.Div(
                                                    [
                                                        html.Label(
                                                            "Output probabilities"
                                                        ),
                                                        dcc.RadioItems(
                                                            ["Yes", "No"],
                                                            "Yes",
                                                            id="radio-knn-output-prob",
                                                        ),
                                                    ]
                                                ),
                                                html.Div(
                                                    [
                                                        html.Button(
                                                            "Run",
                                                            id="button-knn-run",
                                                            n_clicks=0,
                                                        )
                                                    ]
                                                ),
                                            ],
                                            id="div-knn",
                                        ),
                                    ],
                                    id="recommender-params",
                                ),
                            ],
                            id="div-recommender",
                        ),
                    ],
                    id="tab-content",
                ),
            ]
        ),
        position="top-right",
    )
