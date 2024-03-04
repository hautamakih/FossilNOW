from utils.dataframe import create_gdf
import pandas as pd
import plotly.express as px

COLORS = ["rgba(255, 0, 0, 0.3)", "rgba(0, 0, 255, 0.3)", "rgba(255, 255, 0, 0.3)", "rgba(0, 255, 0, 0.3)", "rgba(0, 255, 255, 0.3)", "rgba(255, 0, 255, 0.3)"]


def preprocess_data(df, genera, threshold):
    gdf = create_gdf(pd.DataFrame(df))
    try:
        gdff = gdf[[genera, "LATSTR", "LONGSTR", "geometry", "COUNTRY", "NAME", "MIN_AGE", "MAX_AGE"]][gdf[genera] >= threshold]
    except KeyError:
        gdff = gdf[[genera, "LAT", "LONG", "geometry", "COUNTRY", "SITE_NAME", "MIN_AGE", "MAX_AGE"]][gdf[genera] >= threshold]
    return gdff

def create_map_figure(gdff, genera):
    if len(gdff[genera].unique()) <= 8:
        gdff[genera] = gdff[genera].astype(str)
        fig = px.scatter_mapbox(gdff, lat=gdff.geometry.y, lon=gdff.geometry.x, mapbox_style="open-street-map", zoom=1, color=genera, hover_data=["COUNTRY", "SITE_NAME", "MIN_AGE", "MAX_AGE"])
    else:
        fig = px.scatter_mapbox(gdff, lat=gdff.geometry.y, lon=gdff.geometry.x, mapbox_style="open-street-map", zoom=1, color=genera, hover_data=["COUNTRY", "SITE_NAME", "MIN_AGE", "MAX_AGE"])
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_traces(marker={'size': 15, 'opacity': 0.6})
    return fig

def add_convex_hull_to_figure(fig, gdff, age_spans):
    gdffs = []
    
    if isinstance(age_spans, str):
        age_spans = [age_spans]
    
    for a in age_spans:
        start, end = a.split("-")
        gdffs.append(
            gdff[(gdff["MIN_AGE"] >= float(start)) & (gdff["MAX_AGE"] < float(end))]
        )

    for i, gdf in enumerate(gdffs):
        convex_hull = gdf.unary_union.convex_hull
        
        fig.add_scattermapbox(
            lat=list(convex_hull.exterior.xy[1]),
            lon=list(convex_hull.exterior.xy[0]),
            fill="toself",
            fillcolor=COLORS[i],
            mode="lines",
            line=dict(color=COLORS[i], width=2),
            name="Convex Hull"
        )