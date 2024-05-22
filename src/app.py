import dash_mantine_components as dmc
import dash_daq as daq
import dash
from dash import Dash, html, dcc, callback, Output, Input, State
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.express as px
from utils.dataframe import *
from utils.scatter_mapbox import (
    preprocess_data,
    create_map_figure,
    add_convex_hull_to_figure,
    create_histo,
    add_top_n,
)
import numpy as np
from models.models import get_recommend_list_mf, get_recommend_list_knn
from layout import get_layout
from callback import register_callbacks


app = Dash(__name__, suppress_callback_exceptions=True)

server = app.server

app.layout = get_layout()

register_callbacks()

if __name__ == "__main__":
    app.run(debug=True, port=8050, host="0.0.0.0")
