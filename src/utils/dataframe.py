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

def preprocess_sites_df(sites, mass_diet, dent_genus, rec):
    species_in_sites = sites[['SITE_NAME']].copy()  # Copy the 'SITE_NAME' column to the new DataFrame
    species_in_sites['lat'] = sites['LAT']
    species_in_sites['lon'] = sites['LONG']
    species_in_sites['genus_list'] = sites.iloc[:, 1:453].apply(lambda row: list(row.index[row == 1]), axis=1)

    species_in_sites = add_column_and_average(species_in_sites,'LogMass', mass_diet)

    dent_genus['HYP_Mean'] = dent_genus['HYP_Mean'].fillna(-1)
    dent_genus['LOP_Mean'] = dent_genus['LOP_Mean'].fillna(-1)
    species_in_sites = add_column_and_average(species_in_sites,'HYP_Mean',dent_genus)
    species_in_sites = add_column_and_average(species_in_sites,'LOP_Mean',dent_genus)

    rec['genus_list'] = rec.iloc[:, 1::].apply(lambda row: list(row.index[row == 1]), axis=1)

    rec = add_column_and_average(rec, 'LogMass', mass_diet)
    rec = add_column_and_average(rec, 'HYP_Mean', dent_genus)
    rec = add_column_and_average(rec, 'LOP_Mean', dent_genus)
    rec_species = rec[['SITE_NAME', 'genus_list','LogMass','HYP_Mean','LOP_Mean']]

    return species_in_sites, rec_species

def add_column_and_average(df_to_add, column_name, df):
    '''
    input:
    df_to_add: the dataframe where we will add the column. Should have a list of genera
    column name: the name of the column that will be added to species_in_sites dataframe
    df: dataframe where the information to the species_in_sites dataframe will be taken. Should have column: 'Genus'
    '''
    list_of_genera = list(df['Genus'])
    list_to_add = []
    for ind in df_to_add.index:
        l = []
        for genus in df_to_add['genus_list'].iloc[ind]:
            if genus in list_of_genera:
                l.append(df[column_name].loc[df['Genus'] == genus].iloc[0])
            else:
                l.append(-1)
        list_to_add.append(l)
    df_to_add[column_name] = list_to_add
    #get the average:
    average = []
    for ind in df_to_add.index:
        total = sum(df_to_add[column_name].iloc[ind])
        num_genera = len(df_to_add['genus_list'].iloc[ind])
        avg = total / num_genera if num_genera != 0 else 0
        average.append(avg)

    df_to_add['average_{}'.format(column_name)] = average
    return df_to_add