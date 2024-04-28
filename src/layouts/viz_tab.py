from dash import html, dcc

AGE_SPANS = ["0-0.1", "0.1-0.6", "0.6-1.1", "1.1-1.6", "1.6-2.1", "2.1-2.6"]

def get_viz_tab_components():
    return html.Div(
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
        style={"display": "none"},
    )