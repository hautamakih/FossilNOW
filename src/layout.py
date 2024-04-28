import dash_mantine_components as dmc
from dash import html, dcc

AGE_SPANS = ["0-0.1", "0.1-0.6", "0.6-1.1", "1.1-1.6", "1.6-2.1", "2.1-2.6"]


def get_layout():
    return dmc.NotificationsProvider(
        html.Div(
            [
                html.Div(id="notifications-container"),
                html.Div(id="notification-mf"),
                html.Div(id="notification-mf-done"),
                html.Div(id="recommender-compute-done", style={"display": "none"}),
                html.H1(children="FossilNOW", style={"textAlign": "center"}),
                dcc.Store(id="genera-occurrence-data"),
                dcc.Store(id="genera-info-data"),
                dcc.Store(id="sites-meta-data"),
                dcc.Store(id="prediction-data"),
                dcc.Store(id="visualize-recommendations-data"),
                dcc.Store(id="visualize-true-data"),
                dcc.Store(id="true-negative-data"),
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
                                                html.Li("-1 values in histograms represents missing values in the genera data")
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
                                        "True negatives",
                                    ],
                                    value="Genera occurrences at sites",
                                    clearable=False,
                                ),
                                dcc.Input(
                                    id="n-metacolumns",
                                    type="number",
                                    placeholder="n-meta data columns",
                                    value=0,
                                    style={"display": "none"},
                                ),
                                html.Div(id="div-datatables"),
                                html.Button(
                                    "Extract",
                                    id="split-df-button",
                                    n_clicks=0,
                                    style={"display": "none"},
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
                                                "Collaborative Filtering",
                                                "Hybrid: Content-based x Collaborative"
                                            ],
                                            value="Matrix Factorization",
                                            clearable=False,
                                        ),
                                    ]
                                ),
                                html.Div([
                                    html.Label("Size of train data"),
                                    dcc.Input(id="test-train-split",
                                              value=0.8,
                                              type="number")
                                ]),
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
                                                        html.Label("Dim-hid"),
                                                        dcc.Input(
                                                            id="input-mf-dim-hid",
                                                            value=10,
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
                                                        html.Label(
                                                            "Include true negatives"
                                                        ),
                                                        dcc.RadioItems(
                                                            ["Yes", "No"],
                                                            "No",
                                                            id="radio-mf-true-neg",
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
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Label("Occurence threshold"),
                                                        dcc.Input(
                                                            id="input-content-oc-threshold",
                                                            value=0.8,
                                                            type="number",
                                                        ),
                                                    ]
                                                ),
                                                html.Div(
                                                    [
                                                        html.Button(
                                                            "Run",
                                                            id="button-content-run",
                                                            n_clicks=0,
                                                        )
                                                    ]
                                                ),
                                            ],
                                            id="div-content",
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Label("Top k"),
                                                        dcc.Input(
                                                            id="input-collab-k",
                                                            value=15,
                                                            type="number",
                                                        ),
                                                    ]
                                                ),
                                                html.Div(
                                                    [
                                                        html.Label("Min k"),
                                                        dcc.Input(
                                                            id="input-collab-min_k",
                                                            value=2,
                                                            type="number",
                                                        ),
                                                    ]
                                                ),
                                                html.Div(
                                                    [
                                                        html.Button(
                                                            "Run",
                                                            id="button-collab-run",
                                                            n_clicks=0,
                                                        )
                                                    ]
                                                ),
                                            ],
                                            id="div-collab",
                                        ),
                                    html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Label("Occurence threshold"),
                                                        dcc.Input(
                                                            id="input-hybrid-oc-threshold",
                                                            value=0.8,
                                                            type="number",
                                                        ),
                                                    ]
                                                ),
                                                html.Div(
                                                    [
                                                        html.Label("Method"),
                                                        dcc.RadioItems(
                                                            ['average', 'filter','filter_average'],
                                                            "average",
                                                            id="method-hybrid",
                                                        ),
                                                    ]
                                                ),
                                                html.Div(
                                                    [
                                                        html.Label("Weight"),
                                                        dcc.Input(
                                                            id="input-hybrid-weight",
                                                            value=0.5,
                                                            type="number",
                                                        ),
                                                    ]
                                                ),
                                                html.Div(
                                                    [
                                                        html.Label("Threshold"),
                                                        dcc.Input(
                                                            id="input-hybrid-threshold",
                                                            value=0.01,
                                                            type="number",
                                                        ),
                                                    ]
                                                ),
                                                html.Div(
                                                    [
                                                        html.Label("Top k"),
                                                        dcc.Input(
                                                            id="input-hybrid-k",
                                                            value=15,
                                                            type="number",
                                                        ),
                                                    ]
                                                ),
                                                html.Div(
                                                    [
                                                        html.Label("Min k"),
                                                        dcc.Input(
                                                            id="input-hybrid-min_k",
                                                            value=2,
                                                            type="number",
                                                        ),
                                                    ]
                                                ),
                                                html.Div(
                                                    [
                                                        html.Button(
                                                            "Run",
                                                            id="button-hybrid-run",
                                                            n_clicks=0,
                                                        )
                                                    ]
                                                ),
                                            ],
                                            id="div-hybrid",
                                        ),
                                    ],
                                    id="recommender-params",
                                ),
                                html.Div(id="recommender-metrics")
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
