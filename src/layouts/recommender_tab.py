from dash import html, dcc


def get_recommender_tab_components():
    return html.Div(
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
                                    html.Label(
                                        "Include true negatives"
                                    ),
                                    dcc.RadioItems(
                                        ["Yes", "No"],
                                        "No",
                                        id="radio-content-true-neg",
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
                                    html.Label(
                                        "Include true negatives"
                                    ),
                                    dcc.RadioItems(
                                        ["Yes", "No"],
                                        "No",
                                        id="radio-collab-true-neg",
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
                                    html.Label(
                                        "Include true negatives"
                                    ),
                                    dcc.RadioItems(
                                        ["Yes", "No"],
                                        "No",
                                        id="radio-hybrid-true-neg",
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
        style={"display": "none"},
    )