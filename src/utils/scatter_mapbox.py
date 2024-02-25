from utils.dataframe import create_gdf
import pandas as pd
import plotly.express as px


def preprocess_data(df, genera, threshold):
    gdf = create_gdf(pd.DataFrame(df))
    gdff = gdf[[genera, "LATSTR", "LONGSTR", "geometry", "COUNTRY", "NAME"]][gdf[genera] >= threshold]
    return gdff

def create_map_figure(gdff, genera):
    if len(gdff[genera].unique()) <= 8:
        gdff[genera] = gdff[genera].astype(str)
        fig = px.scatter_mapbox(gdff, lat=gdff.geometry.y, lon=gdff.geometry.x, mapbox_style="open-street-map", zoom=3, color=genera, hover_data=["COUNTRY", "NAME"])
    else:
        fig = px.scatter_mapbox(gdff, lat=gdff.geometry.y, lon=gdff.geometry.x, mapbox_style="open-street-map", zoom=3, color=genera, hover_data=["COUNTRY", "NAME"])
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_traces(marker={'size': 15, 'opacity': 0.6})
    return fig

def add_convex_hull_to_figure(fig, convex_hull, gdff):
    fig.add_scattermapbox(
        lat=list(convex_hull.exterior.xy[1]),
        lon=list(convex_hull.exterior.xy[0]),
        fill="toself",
        fillcolor="rgba(255, 0, 0, 0.3)",
        mode="lines",
        line=dict(color="red", width=2),
        name="Convex Hull"
    )