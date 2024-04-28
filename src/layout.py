import dash_mantine_components as dmc
from dash import html, dcc
import layouts.hidden as hidden_layouts
import layouts.tabs as tabs
import layouts.data_tab as data_tab
import layouts.recommender_tab as recommender_tab
import layouts.viz_tab as viz_tab


def get_layout():
    return dmc.NotificationsProvider(
        html.Div(
            [
                hidden_layouts.get_notification_div(),
                html.Div(id="recommender-compute-done", style={"display": "none"}),
                html.H1(children="FossilNOW", style={"textAlign": "center"}),
                hidden_layouts.get_data_stores(),
                tabs.get_tabs(),
                html.Div(
                    [
                        data_tab.get_data_tab_components(),
                        recommender_tab.get_recommender_tab_components(),
                        viz_tab.get_viz_tab_components(),
                    ],
                    id="tab-content",
                ),
            ]
        ),
        position="top-right",
    )
