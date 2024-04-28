from dash import html, dcc


def get_data_tab_components():
    return html.Div(
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
    )