import sys
sys.path.append("..")

from CBF.ContentBasedFiltering import ContentBasedFiltering
from surprise import KNNBasic, Dataset, Reader

import numpy as np
import pandas as pd

class CbfCfHybrid:
    def __init__(self):
        pass


    def fit(self, occurences: pd.DataFrame, site_data: pd.DataFrame, genus_data: pd.DataFrame, k: int, min_k: int, content_based_weight=0.5, normalization: str="min-max", sim_options: dict={'name': "MSD",'user_based': True}):
        self.fit_content_based(occurences, site_data, genus_data, normalization)
        self.fit_kNN(occurences, k=k, min_k=min_k, sim_options=sim_options)
        self.calculate_hybrid_score(content_based_weight)
    

    def fit_content_based(self, occurences: pd.DataFrame, site_data: pd.DataFrame, genus_data: pd.DataFrame, normalization: str):
        cbf = ContentBasedFiltering()
        cbf.fit(
            occurences = occurences,
            site_data = site_data,
            genus_data = genus_data, 
            normalization = normalization
        )

        self.cbf_scores = cbf.get_recommendations(matrix_form=True)
        self.cbf_scores_table = self.cbf_scores.stack().reset_index().rename(columns={"level_1": "genus", 0: "cbf_similarity"})
    

    def fit_kNN(self, occurences: pd.DataFrame, k, min_k, sim_options):
        # Occurences in stacked form (edit sot that site name is taken from first column)
        occurences_non_matrix = occurences.set_index("SITE_NAME").stack().reset_index().rename(columns={"level_1": "genus", 0: "occurance"})

        # Fit the kNN Collaborative Filtering
        reader = Reader(rating_scale=(0, 1))
        data = Dataset.load_from_df(occurences_non_matrix, reader) # Column order must be user, item, rating

        trainset = data.build_full_trainset()

        knn = KNNBasic(k=5, min_k=1, sim_options=sim_options)
        knn.fit(trainset)

        testset = trainset.build_testset()

        # Get predictions for all user-item pairs
        predictions = knn.test(testset)

        # Get item scores from the predictions
        item_scores = [(prediction.uid, prediction.iid, prediction.est) for prediction in predictions]
        self.knn_scores = pd.DataFrame(item_scores, columns =['SITE_NAME', 'genus', 'knn_similarity'])
    

    def calculate_hybrid_score(self, content_based_weight: float=0.5):
        knn_weight = 1 - content_based_weight

        self.scores = pd.merge(self.cbf_scores_table, self.knn_scores, on=["SITE_NAME", "genus"], how="inner")
        self.scores['hybrid_similarity'] = content_based_weight * self.scores['cbf_similarity'] + knn_weight * self.scores['knn_similarity']

        self.hybrid_scores = self.scores.pivot(index="SITE_NAME", columns="genus", values="hybrid_similarity")
    

    def get_recommendations(self, matrix_form:bool=True):
        """
        Gives the similarity scores for all the genus-site pairs

        Parameters:
        -----------
        matrix_form: bool
            A boolean value. If assigned true the recommendations are returned in matrix form. If assigned False the recommendations are returned in DataFrame that has columns SITE_NAME, genus, similarity. The defauls value is True.

        Returns:
        --------
        pd.DataFrame
        """

        if matrix_form:
            return self.hybrid_scores
        else:
            return self.cbf_scores


    def predict(self):
        pass

if __name__ == "__main__":
    path = "../../../data/FossilGenera_MammalMassDiet_Jan24.csv"
    df_mass_diet = pd.read_csv(path, sep=",")

    # With genus info, give the columns you want to use and convert categorical using one-hot-encoding
    genus_info_cols = [
        "Genus",
        "Order",
        "Family",
        "Massg",
        "Diet",
        ]
        
    df_genus_info = df_mass_diet[genus_info_cols]

    dummy_cols = [
        "Order",
        "Family",
        "Diet",
        ]

    #The genus column must be the first one in genus data
    df_genus_info = pd.get_dummies(df_genus_info, columns=dummy_cols)
    df_genus_info = df_genus_info.replace({False: 0, True: 1})

    # Genera and sites
    path = "../../../data/AllSites_SiteOccurrences_AllGenera_26.1.24.csv"

    # When giving the site-genus matrix, only give dataframe with columns that are used to fit. Spedsify the number of site-info columns at the end.
    # The first column must be the site name
    df = pd.read_csv(path)
    cols = [
        'ALTITUDE',
        'BFA_MAX',
        'BFA_MAX_ABS',
        'BFA_MIN',
        'BFA_MIN_ABS',
        'COUNTRY',
        'Total_Gen_Count',
        'smallperlarge',
        'smallprop',
        'DietRatio',
        'HerbProp',
    ]

    df = df.drop(columns=cols)

    occurences = df.iloc[:,:-10]
    site_info_cols = ["SITE_NAME"] + df.iloc[:,-10:].columns.to_list()
    site_info = df[site_info_cols]

    hybrid = CbfCfHybrid()
    hybrid.fit(occurences, site_info, df_genus_info, k=10, min_k=1, normalization='min-max')

    print(hybrid.get_recommendations())