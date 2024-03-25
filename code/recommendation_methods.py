import pandas as pd

def recommend_n(recommendation_and_real: pd.DataFrame, n: int=10):
    def top_n(df, n, column='occurence'):
        return df.nlargest(n, column)
    
    recommendation_and_real = recommendation_and_real.groupby(
        'SITE_NAME', group_keys=False
    ).apply(
        top_n, n
    ).sort_values(
        by=["SITE_NAME", "similarity"], ascending=[True, False]
    )

    return recommendation_and_real

def recommend_using_cutoff(recommendation_and_real: pd.DataFrame, cutoff: float=0.75):
    recommendation_and_real = recommendation_and_real.groupby(
        'SITE_NAME', group_keys=False
    ).apply(
        lambda x: x[x['similarity'] >= cutoff]
    ).sort_values(
        by=["SITE_NAME", "similarity"], ascending=[True, False]
    )
    
    return recommendation_and_real

def recommend_n_more(recommendations_and_real: pd.DataFrame, n: int):
    def filter_rows(group):
        sorted_group = group.sort_values(by='similarity', ascending=False)
        occurence_1 = sorted_group[sorted_group['occurence'] == 1]
        occurence_0_top_n = sorted_group[sorted_group['occurence'] == 0].head(n)
        return pd.concat([occurence_1, occurence_0_top_n])
    
    recommendations_and_real = recommendations_and_real.groupby('SITE_NAME').apply(filter_rows)
    recommendations_and_real.reset_index(drop=True, inplace=True)
    
    return recommendations_and_real

if __name__ == "__main__":
    # Testing the functions
    from ContentBasedFiltering import ContentBasedFiltering

    path = "../data/FossilGenera_MammalMassDiet_Jan24.csv"
    df_mass_diet = pd.read_csv(path, sep=",")

    path = "../data/AllSites_SiteOccurrences_AllGenera_26.1.24.csv"
    df = pd.read_csv(path)

    # Fitting the algorithm
    cbf = ContentBasedFiltering()
    cbf.fit(df, df_mass_diet)

    # Dropping site information columns to do the predictions using the same data
    cols_redundant = [
    'LAT',
    'LONG',
    'ALTITUDE',
    'MAX_AGE',
    'BFA_MAX',
    'BFA_MAX_ABS',
    'MIN_AGE',
    'BFA_MIN',
    'BFA_MIN_ABS',
    'COUNTRY',
    'age_range',
    'Total_Gen_Count',
    'Large_GenCount',
    'Small_GenCount',
    'smallperlarge',
    'smallprop',
    'Herb_GenCount',
    'Nonherb_GenCount',
    'DietRatio',
    'HerbProp',
    'mid_age'
    ]

    df_site_genus_matrix = df.drop(columns=cols_redundant)

    df_test = df_site_genus_matrix.reset_index(drop=True).melt(id_vars='SITE_NAME', var_name='genus', value_name='occurence')
    true_and_pred = cbf.predict(df_test)

    # Testing the functions
    print(recommend_n_more(true_and_pred, 5))