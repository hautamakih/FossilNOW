#%%
import pandas as pd
import numpy as np
import utils_local

from scipy import stats

from surprise import Dataset, Reader
from surprise import KNNBasic, KNNWithMeans
from surprise.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.metrics import root_mean_squared_error, mean_squared_error, mean_absolute_error, f1_score

#%%

class ContentBasedFiltering:
    def __init__(self):
        self.site_genus_matrix = None
        self.site_info = None
        self.genus_info = None
        self.genus_related_site_info = None
        self.genus_info_with_site_info = None
        self.site_info_with_genus_info = None
        self.recommendation_matrix = None
        self.recommendations = None


    def fit(self, site_data, genus_data):
        self.build_site_genus_matrix(site_data)
        self.build_site_info(site_data)
        self.build_genus_info(genus_data)
        self.build_genus_related_site_info()
        self.build_genus_info_with_site_info(genus_data)
        self.build_site_info_with_genus_info(site_data)

        # This has to be edited so that it allways finds recommendations for all
        self.find_recommendations_for_all_sites(site_data, normalization=self.normalize_columns_min_max, n_species_to_recommend=500)
    

    def get_recommendations(self, matrix_form=True):
        if matrix_form:
            return self.recommendation_matrix
        else:
            return self.recommendations


    def predict(self, test_set):
        df_test = test_set.merge(
            self.recommendations, 
            on=["SITE_NAME", "genus"],
            how="left"
        ).sort_values(
            by=["SITE_NAME", "similarity"],
            ascending=[True, False]
        )

        return df_test


    def build_site_genus_matrix(self, df):
        cols_redundant = ['LAT',
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

        df_site_genus = df.drop(columns=cols_redundant).set_index('SITE_NAME')
        self.site_genus_matrix = df_site_genus


    def build_site_info(self, df):
        site_info_cols = [
        'SITE_NAME',
        'LAT',
        'LONG',
        'MAX_AGE',
        'MIN_AGE',
        'age_range',
        'Large_GenCount',
        'Small_GenCount',
        'Herb_GenCount',
        'Nonherb_GenCount',
        'mid_age'
        ]

        df_site_info = df[site_info_cols].set_index('SITE_NAME')
        self.site_info = df_site_info


    def build_genus_info(self, df):
        genus_info_cols = [
            "Genus",
            "Order",
            "Family",
            "Massg",
            "Diet",
        # "DietSource"
        ]
        
        df_genus_info = df[genus_info_cols]

        dummy_cols = [
            "Order",
            "Family",
            "Diet",
        # "DietSource"
        ]

        df_genus_info = pd.get_dummies(df_genus_info, columns=dummy_cols)
        df_genus_info = df_genus_info.replace({False: 0, True: 1})
        df_genus_info = df_genus_info.rename(columns={"Genus": "genus"})

        self.genus_info = df_genus_info


    def build_genus_related_site_info(self):
        genus_info = self.genus_info
        site_genus = self.site_genus_matrix

        site_genus = site_genus.stack().reset_index().rename(columns={"level_1": "genus", 0: "presence"})
        site_genus = site_genus[site_genus["presence"] == 1].drop("presence", axis="columns")

        site_genus = site_genus.merge(genus_info, on="genus", how="left")
        site_genus = site_genus.drop(["genus"], axis=1)
        site_genus = site_genus.groupby('SITE_NAME').mean().reset_index().set_index("SITE_NAME")
        
        self.genus_related_site_info = site_genus


    def build_genus_info_with_site_info(self, df_genus_data):
        site_genus = self.site_genus_matrix
        site_genus = site_genus.stack().reset_index().rename(columns={"level_1": "genus", 0: "presence"})
        site_info = self.site_info

        genus_info = site_genus.merge(site_info, on="SITE_NAME", how="left")
        genus_info = genus_info[genus_info["presence"] == 1]

        genus_info = genus_info.drop(["SITE_NAME", "presence"], axis=1)
        genus_info = genus_info.groupby('genus').mean().reset_index().set_index("genus")

        df_genus_data = self.genus_info
        genus_info = genus_info.merge(df_genus_data, left_index=True, right_on="genus", how="left").reset_index(drop=True).set_index("genus")

        self.genus_info_with_site_info = genus_info


    def build_site_info_with_genus_info(self, df):
        site_info_cols = [
        'SITE_NAME',
        'LAT',
        'LONG',
        'MAX_AGE',
        'MIN_AGE',
        'age_range',
        'Large_GenCount',
        'Small_GenCount',
        'Herb_GenCount',
        'Nonherb_GenCount',
        'mid_age'
        ]

        df_site_info = df[site_info_cols].set_index('SITE_NAME')
        
        df_site_info_by_genera = self.genus_related_site_info
        df_site_info = df_site_info.merge(df_site_info_by_genera, left_index=True, right_on="SITE_NAME", how="left")
        
        self.site_info_with_genus_info = df_site_info
    

    def normalize_columns_min_max(self, df):
        return (df - df.min()) / (df.max() - df.min())
    

    def normalize_columns_mean(self, df):
        return (df - df.mean()) / df.std()


    def get_recommendations_for_site(self, genus_info, site_name, site_indices, genus_site_similarity_matrix, num_recommend = 10):
        idx = site_indices[site_name]

        # Sorted similarity scores
        sim_scores = sorted(list(enumerate(genus_site_similarity_matrix[:,idx])), key=lambda x: x[1], reverse=True)

        # Get the scores of the num_recommend most similar sites
        similar_genus_for_site = sim_scores[:num_recommend]

        # Get the genus indices
        genus_indices = [i[0] for i in similar_genus_for_site]
        genus_site_similarities = [i[1] for i in similar_genus_for_site]

        recommended_genus = genus_info.iloc[genus_indices].index.to_frame(index=False).assign(similarity=genus_site_similarities)
        recommended_genus.insert(0, "SITE_NAME", site_name)

        return recommended_genus
    
    def find_recommendations_for_all_sites(self, df, normalization=None, n_species_to_recommend=500):
        genus_info = self.genus_info_with_site_info
        site_info = self.site_info_with_genus_info
        
        if normalization != None:
            genus_info = normalization(genus_info)
            site_info = normalization(site_info)
        
        if genus_info.isnull().values.any():
            print("WARNING! Genus info data contains nans. Assigning to zeros")
            genus_info = genus_info.fillna(0)
        
        if site_info.isnull().values.any():
            print("WARNING! Site info data contains nans. Assigning to zeros")
            site_info = site_info.fillna(0)


        site_indices = pd.Series(df.index, index=df["SITE_NAME"]).drop_duplicates()
        sim = cosine_similarity(genus_info, site_info)

        recommendations = []
        for site, idx in site_indices.items():
            site_recommendations = self.get_recommendations_for_site(
                genus_info=genus_info,
                site_name=site,
                site_indices=site_indices,
                genus_site_similarity_matrix=sim,
                num_recommend=n_species_to_recommend                        
            )

            recommendations.append(site_recommendations)
        
        self.recommendations = pd.concat(recommendations).reset_index(drop=True)
        self.recommendation_matrix = pd.pivot(self.recommendations, index="SITE_NAME", columns="genus", values="similarity").fillna(0)


#%%
if __name__ == "__main__":
    path = "../data/FossilGenera_MammalMassDiet_Jan24.csv"
    df_mass_diet = pd.read_csv(path, sep=",")

    # Genera and sites
    path = "../data/AllSites_SiteOccurrences_AllGenera_26.1.24.csv"

    df = pd.read_csv(path)

    cbf = ContentBasedFiltering()
    cbf.fit(df, df_mass_diet)

# %%