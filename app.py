from dash import Dash, html, dcc, callback, Output, Input
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import plotly.express as px

app = Dash(__name__)

df = pd.read_csv("fossilrec/3_data/data_occ.csv", sep="\t")

def parse_coord(coord_str):
    parts = coord_str.split()
    degrees = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2]) if len(parts) > 2 else 0.0
    direction = parts[-1]
    
    decimal_degrees = degrees + minutes / 60 + seconds / 3600
    if direction in ['S', 'W']:
        decimal_degrees *= -1
    
    return decimal_degrees

def create_gdf(df):

    latitudes = [parse_coord(lat_str) for lat_str in df["LATSTR"]]
    longitudes = [parse_coord(lon_str) for lon_str in df["LONGSTR"]]
    # Create a GeoDataFrame
    geometry = [Point(lon, lat) for lon, lat in zip(longitudes, latitudes)]
    df["geometry"] = geometry
    gdf = gpd.GeoDataFrame(df)
    return gdf

gdf = create_gdf(df)
color_discrete_map = {'0': 'rgb(255,0,0)', '0.1': 'rgb(0,255,0)', '0.7': 'rgb(0,0,255)', '0.9': 'rgb(255,255,0)', '1': 'rgb(255,0,255)'}


app.layout=html.Div([
    html.H1(children='FossilNOW', style={'textAlign':'center'}),
    dcc.Dropdown(list(df.columns)[:104], 'Eotragus', id='dropdown-selection'),
    "Threshold",
    dcc.Slider(0, 1, 0.1,value=0, id='threshold'),
    dcc.Graph(id='graph-content')
])

@callback(
    Output('graph-content', 'figure'),
    Input('dropdown-selection', 'value'),
    Input('threshold', 'value'),
)
def update_graph(genera, threshold):
    gdff = gdf[[genera, "LATSTR", "LONGSTR", "geometry"]][gdf[genera] >= threshold]
    gdff[genera] = gdff[genera].astype(str)
    fig = px.scatter_mapbox(gdff, lat=gdff.geometry.y, lon=gdff.geometry.x, mapbox_style="open-street-map", zoom=3, color=genera, color_discrete_map=color_discrete_map)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_traces(marker={'size': 15, 'opacity': 0.6})
    return fig

if __name__ == '__main__':
    app.run(debug=True)