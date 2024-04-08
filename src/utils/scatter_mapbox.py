from utils.dataframe import create_gdf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = ["rgba(255, 0, 0, 0.3)", "rgba(0, 0, 255, 0.3)", "rgba(255, 255, 0, 0.3)", "rgba(0, 255, 0, 0.3)", "rgba(0, 255, 255, 0.3)", "rgba(255, 0, 255, 0.3)"]


def preprocess_data(df, genera, threshold):
    gdf = create_gdf(df)
    lat = "LAT" if "LAT" in gdf.columns else "LATSTR"
    long = "LONG" if "LONG" in gdf.columns else "LONGSTR"
    try:
        gdff = gdf[[genera, "LATSTR", "LONGSTR", "geometry", "COUNTRY", "NAME", "MIN_AGE", "MAX_AGE"]][gdf[genera] >= threshold]
    except KeyError:
        gdff = gdf[[genera, "LAT", "LONG", "geometry", "COUNTRY", "SITE_NAME", "MIN_AGE", "MAX_AGE"]][gdf[genera] >= threshold]
    return gdff

def create_map_figure(gdff, genera):
    fig = go.Figure()

    site_name = "SITE_NAME" if "SITE_NAME" in gdff.columns else "NAME"

    try:
        if len(gdff[genera].unique()) <= 8:
            gdff[genera] = gdff[genera].astype(str)
            fig.add_trace(px.scatter_mapbox(gdff, lat=gdff.geometry.y, lon=gdff.geometry.x, color=genera, hover_data=["COUNTRY", "MIN_AGE", "MAX_AGE"], hover_name=site_name).data[0])
        else:
            fig.add_trace(px.scatter_mapbox(gdff, lat=gdff.geometry.y, lon=gdff.geometry.x, color=genera, hover_data=["COUNTRY", "MIN_AGE", "MAX_AGE"], hover_name=site_name).data[0])
    except IndexError:
        fig.add_trace(px.scatter_mapbox(gdff, lat=gdff.geometry.y, lon=gdff.geometry.x).data[0])
        fig.update_traces(mode='markers', marker=dict(opacity=0))
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_center={"lat": gdff.geometry.y.mean(), "lon": gdff.geometry.x.mean()},
        mapbox_zoom=0,
        coloraxis_colorbar=dict(
            len=0.5,
            yanchor='bottom'
        ),
    )
    fig.update_traces(marker={'size': 7, 'opacity': 0.6})
    return fig

def add_convex_hull_to_figure(fig, gdff, age_spans):
    gdffs = {}
    
    if isinstance(age_spans, str):
        age_spans = [age_spans]
    
    for a in age_spans:
        start, end = a.split("-")
        gdffs[a] = gdff[(gdff["MIN_AGE"] >= float(start)) & (gdff["MAX_AGE"] < float(end))]

    errors = []

    for i, (age_span, gdf) in enumerate(gdffs.items()):
        # Skip drawing a convex hull if it has less than three points
        if len(gdf.drop_duplicates(subset=['LAT', 'LONG'])) < 3:
            errors.append(age_span)
            continue
        
        convex_hull = gdf.unary_union.convex_hull

        fig.add_scattermapbox(
            lat=list(convex_hull.exterior.xy[1]),
            lon=list(convex_hull.exterior.xy[0]),
            fill="toself",
            fillcolor=COLORS[i],
            mode="lines",
            line=dict(color=COLORS[i], width=2),
            name=age_span
        )

    fig.update_layout(
        legend=dict(x=1.025, y=0.5, yanchor="top")
    )

    return errors

def add_top_n(gdff, genera, n, fig):
    n_highest_df = gdff.sort_values(by=genera, ascending=False).iloc[:n, :]

    try:
        fig.add_trace(
            px.scatter_mapbox(
                n_highest_df,
                lat=n_highest_df.geometry.y,
                lon=n_highest_df.geometry.x,
                color=genera,
                hover_data=["COUNTRY", "MIN_AGE", "MAX_AGE"],
                hover_name="SITE_NAME"
            ).update_traces(marker={"size": 25}).data[0]
        )
    except IndexError:
        pass

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

def create_histo(clickData, species_in_sites, rec_species):
    #site data:
    site_name = clickData['points'][0]['hovertext']

    if 'SITE_NAME' in species_in_sites:
        site = 'SITE_NAME'
    elif 'NAME' in species_in_sites:
        site = 'NAME'
    #check if empty
    if len(species_in_sites[species_in_sites[site] == site_name].index) == 0:
        site_data = species_in_sites[species_in_sites[site] == site_name]
    else:   
        site_data = species_in_sites[species_in_sites[site] == site_name].iloc[0]

    # #recommendations:
    if 'SITE_NAME' in rec_species:
        site = 'SITE_NAME'
    elif 'NAME' in rec_species:
        site = 'NAME'
    #check if empty:
    if len(rec_species[rec_species[site] == site_name].index) == 0:
        rec_data = rec_species[rec_species[site] == site_name]
    else:
        rec_data = rec_species[rec_species[site] == site_name].iloc[0]

    # print('site: ', site_data)
    # print('rec: ', rec_data)
    #LOG MASS:
    mass_bar_fig = go.Figure()
    mass_bar_fig.add_trace(go.Histogram(x=site_data["LogMass"],
                                        name='True',
                                        xbins=dict(
                                            start=-1,
                                            end=10.0,
                                            size=0.5
                                        ),
                                        opacity = 0.75,
                                        hovertext=[f'Genus: {genus}' for genus in site_data['genus_list']]))
    mass_bar_fig.add_trace(go.Histogram(x=rec_data["LogMass"], 
                                        name='Recommandations',
                                        xbins=dict(
                                            start=-1,
                                            end=10.0,
                                            size=0.5
                                        ),
                                        opacity = 0.75,
                                        hovertext=[f'Genus: {genus}' for genus in rec_data['genus_list']]))
    mass_bar_fig.update_xaxes(title_text='Mass (log)')
    mass_bar_fig.update_layout(title='Masses (log) of genera in {}'.format(site_name),
                                bargap=0.2)

    mass_bar_fig.add_annotation(
        text="Genus (True): {}".format(site_data['genus_list']),
        align='right',
        showarrow=False,
        xref='paper',
        yref='paper',
        x=0,
        y=1.17,
        bordercolor='black',
        borderwidth=1)
    mass_bar_fig.add_annotation(
        text="Genus (Rec): {}".format(rec_data['genus_list']),
        align='right',
        showarrow=False,
        xref='paper',
        yref='paper',
        x=0,
        y=1.1,
        bordercolor='black',
        borderwidth=1)
    
    #DENTAL DATA:
    dent_fig = make_subplots(rows=1, cols=2)
    #Hyp:
    dent_fig.add_trace(go.Histogram(x=site_data["HYP_Mean"],
                                        name = 'True HYP',
                                        xbins=dict(
                                            start=-1,
                                            end=5.0,
                                            size=0.5
                                        ),
                                        marker_color='skyblue',
                                        opacity = 0.85),
                        row=1, col=1)
    dent_fig.add_trace(go.Histogram(x=rec_data['HYP_Mean'],
                                   name = 'Rec HYP',
                                        xbins=dict(
                                            start=-1,
                                            end=5.0,
                                            size=0.5
                                        ),
                                        marker_color='violet',
                                        opacity = 0.85),
                        row=1, col=1)
    #Lop:
    dent_fig.add_trace(go.Histogram(x=site_data["LOP_Mean"],
                                         name = 'True LOP', 
                                        xbins=dict(
                                            start=-1,
                                            end=5.0,
                                            size=0.5
                                        ),
                                        marker_color='aquamarine',
                                        opacity = 0.85),
                        row=1, col=2)
    dent_fig.add_trace(go.Histogram(x=rec_data["LOP_Mean"],
                                         name = 'Rec LOP', 
                                        xbins=dict(
                                            start=-1,
                                            end=5.0,
                                            size=0.5
                                        ),
                                        marker_color='hotpink',
                                        opacity = 0.85),
                        row=1, col=2)
    
    dent_fig.update_xaxes(title_text='Mean HYP', row=1, col=1)
    dent_fig.update_xaxes(title_text='Mean LOP', row=1, col=2)
    dent_fig.update_layout(title='Mean Hypsodonty and loph of genera in {}'.format(site_name),
                               bargap=0.2)
    
    return site_name, mass_bar_fig, dent_fig