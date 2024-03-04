import geopandas as gpd
from shapely.geometry import Point

import pandas as pd
import io
import base64



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
    try:
        latitudes = [parse_coord(lat_str) for lat_str in df["LATSTR"]]
        longitudes = [parse_coord(lon_str) for lon_str in df["LONGSTR"]]
        geometry = [Point(lon, lat) for lon, lat in zip(longitudes, latitudes)]
    except KeyError:
        geometry = [Point(lon, lat) for lon, lat in zip(df["LONG"], df["LAT"])]
    df["geometry"] = geometry
    gdf = gpd.GeoDataFrame(df)
    return gdf

def parse_contents(contents):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(
            io.StringIO(decoded.decode("utf-8")),
            sep="\t"
        )
        if df.shape[1] == 1:
            df = pd.read_csv(
                io.StringIO(decoded.decode("utf-8"))
            )
    except Exception as e:
        print(e)
    
    return df