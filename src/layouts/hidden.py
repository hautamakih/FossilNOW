from dash import html, dcc

def get_notification_div():
    return html.Div([
        html.Div(id="notifications-container"),
        html.Div(id="notification-mf"),
        html.Div(id="notification-mf-done"),
    ], id="notifications")

def get_data_stores():
    return html.Div([
        dcc.Store(id="genera-occurrence-data"),
        dcc.Store(id="genera-info-data"),
        dcc.Store(id="genera-dental-data"),
        dcc.Store(id="sites-meta-data"),
        dcc.Store(id="prediction-data"),
        dcc.Store(id="visualize-recommendations-data"),
        dcc.Store(id="visualize-true-data"),
        dcc.Store(id="true-negative-data"),
    ], id="data-stores")