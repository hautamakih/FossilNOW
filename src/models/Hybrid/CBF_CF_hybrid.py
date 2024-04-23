#%%
import sys
# sys.path.append("..")

from models.CBF.ContentBasedFiltering import ContentBasedFiltering
from surprise import KNNBasic, Dataset, Reader

import numpy as np
import pandas as pd

#%%
class CbfCfHybrid:
    """
    A hybrid recommender algorithm that combines kNN collaborative filtering and Content-based filtering

    Attibutes
    ---------
    None

    Methods
    -------
    fit(occurences, site_data, genus_data, k, min_k, method, content_based_weight, filter_threshold, normalization, sim_options):
        Fits the algorithm on given data

    get_recommendations(matrix_form=True):
        Gives the similarity scores for all the genus-site pairs

    predict(test_set):
        Gives a DataFrame with true values of the test set and predicted values from the fit
    """

    def __init__(self):
        pass


    def fit(
            self, 
            occurences: pd.DataFrame, 
            site_data: pd.DataFrame, 
            genus_data: pd.DataFrame, 
            k: int, 
            min_k: int,
            method: str="average",
            content_based_weight: float=0.5,
            filter_threshold: float=0.01, 
            normalization: str="min-max", 
            occurence_threshold: float = 0.8,
            sim_options: dict={'name': "MSD",'user_based': True}
        ):
        """
        Fits the algorithm on given data

        Parameters:
        -----------
        occurences: pd.DataFrame
            a Pandas DataFrame containing occurences of genera at each site in a matrix form. The first column must be the site name.

        site_data: pd.DataFrame
            a Pandas DataFrame containing the site metadata. The first column must be the site name.
            Categorical variables must be converted to numberival beforehand.

        genus_data: pd.DataFrame
            a Pandas DataFrame containing the information about genera. Categorical features must be converted using one-hot-encoding beforehand. 
            The first column must be the genus name. Categorical variables must be converted to numberival beforehand.

        k: int
            The maximum number of neigbours used by the kNN Collaborative filter algorithm

        min_k: int
            The minimum number of neigbours used by the kNN Collaborative filter algorithm

        method: str
            The method used to combine the scores of the two predictor algorithms. Can be 'average', 'filter' or 'filter_average'. 
            The 'average' calculates the average similarity of the two prediction algorithms. 
            The 'filter' uses the content-based filtering similarity scores if the kNN result is above a given threshold. 
            The 'filter_average' combines both by first filtering and then calculating the average for those values that are above the threshold.
        
        content_based_weight: float
            A weight of content-based similarity values in hybrid results. Must be between 0 and 1. This is used if method is either 'average' or 'filter_average'.
        
        filter_threshold: float
            A threshold value for kNN Collaborative filtering similarity score. Values below the threshold are assigned to 0 in hybrid similarity if using
            'filter' or 'filter_average' method.

        normalization: str
            The type of normalization used to normalize columns before calculating the similarity scores. 
            Possible values: ["min-max", "mean"]. The default value is min-max.

        sim_options: dict
            The similarity parameters used by the kNN Collaborative filtering. See the Surprise documentation if you want to change this.
        
        occurence_threshold: float
            A threshold value that tells the algorithm which values are handled as occurences in the occurence data. This is needed if your occurence data has values between 0 and 1 (not just 0s and 1s).
            The algorithm cannot handle uncertancies in the data and hence values over or equal to the threshold are handled as occurence. The default value is 0.8.

        Returns
        -------
        None

        References:
        -----------
        Surprise documentation for similarity options:
            https://surprise.readthedocs.io/en/stable/similarities.html
        """


        site_data = site_data.rename(columns={site_data.columns[0]: "SITE_NAME"}) # Saving the site data to a class variable and giving name "SITE_NAME" to the first column
        genus_data = genus_data.rename(columns={genus_data.columns[0]: "genus"}) # Saving the genus data to a class variable and giving name "genus" to the first column
        occurences = occurences.rename(columns={occurences.columns[0]: "SITE_NAME"}) # Renaming the first column to site name

        print("Fitting the hybrid algorithm...")
        self.__fit_content_based(occurences, site_data, genus_data, normalization, occurence_threshold)
        self.__fit_kNN(occurences, k=k, min_k=min_k, sim_options=sim_options)

        if method == "average":
            self.__calculate_hybrid_score_mean(content_based_weight)
        elif method == "filter":
            self.__calculate_hybrid_score_filter(filter_threshold)
        elif method == "filter_average":
            self.__calculate_hybrid_score_filter_average(filter_threshold, content_based_weight)
        else:
            raise ValueError("The method must be either 'average' or 'filter'.")
        print("Fitting the hybrid algorithm complete.")
    

    def __fit_content_based(
            self, 
            occurences: pd.DataFrame, 
            site_data: pd.DataFrame, 
            genus_data: pd.DataFrame, 
            normalization: str,
            occurence_threshold: float
        ):
        """
        Fits the Content-based filtering on the given data

        Parameters:
        -----------
        occurences: pd.DataFrame
            a Pandas DataFrame containing occurences of genera at each site in a matrix form. The first column must be the site name.

        site_data: pd.DataFrame
            a Pandas DataFrame containing the site metadata. The first column must be the site name.

        genus_data: pd.DataFrame
            a Pandas DataFrame containing the information about genera. Categorical features must be converted using one-hot-encoding beforehand. The first column must be the genus name.

        normalization: str
            The type of normalization used to normalize columns before calculating the similarity scores. Possible values: ["min-max", "mean"]. The default value is min-max.
 
        occurence_threshold: float
            A threshold value that tells the algorithm which values are handled as occurences in the occurence data. This is needed if your occurence data has values between 0 and 1 (not just 0s and 1s).
            The algorithm cannot handle uncertancies in the data and hence values over or equal to the threshold are handled as occurence. The default value is 0.8.

        Returns
        -------
        None
        """

        cbf = ContentBasedFiltering()
        cbf.fit(
            occurences = occurences,
            site_data = site_data,
            genus_data = genus_data, 
            normalization = normalization,
            occurence_threshold = occurence_threshold
        )

        self.cbf_scores = cbf.get_recommendations(matrix_form=True)
        self.cbf_scores_table = self.cbf_scores.stack().reset_index().rename(columns={"level_1": "genus", 0: "cbf_similarity"})
    

    def __fit_kNN(
            self, 
            occurences: pd.DataFrame, 
            k, 
            min_k, 
            sim_options
        ):
        """
        Fits the kNN Collaborative filtering on the given data

        Parameters:
        -----------
        occurences: pd.DataFrame
            a Pandas DataFrame containing occurences of genera at each site in a matrix form. The first column must be the site name.

        k: int
            The maximum number of neigbours used by the kNN Collaborative filter algorithm

        min_k: int
            The minimum number of neigbours used by the kNN Collaborative filter algorithm

        sim_options: dict
            The similarity parameters used by the kNN Collaborative filtering. See the Surprise documentation if you want to change this.

        Returns
        -------
        None
        """

        print("Fitting the kNN Collaborative filtering...")
        # Occurences in stacked form (edit sot that site name is taken from first column)
        occurences_non_matrix = occurences.set_index("SITE_NAME").stack().reset_index().rename(columns={"level_1": "genus", 0: "occurance"})

        # Fit the kNN Collaborative Filtering
        reader = Reader(rating_scale=(0, 1))
        data = Dataset.load_from_df(occurences_non_matrix, reader) # Column order must be user, item, rating

        trainset = data.build_full_trainset()

        knn = KNNBasic(k=k, min_k=min_k, sim_options=sim_options)
        knn.fit(trainset)

        testset = trainset.build_testset()

        # Get predictions for all user-item pairs
        predictions = knn.test(testset)

        # Get item scores from the predictions
        item_scores = [(prediction.uid, prediction.iid, prediction.est) for prediction in predictions]
        self.knn_scores = pd.DataFrame(item_scores, columns =['SITE_NAME', 'genus', 'knn_similarity'])
        print("kNN fit complete.")
    

    def __calculate_hybrid_score_mean(self, content_based_weight: float=0.5):
        """
        Calculates the hybrid score using the average method

        Parameters:
        -----------
        content_based_weight: float
            A weight of content-based similarity values in hybrid results. Must be between 0 and 1. This is used if method is either 'average' or 'filter_average'.

        Returns
        -------
        None
        """
        knn_weight = 1 - content_based_weight

        self.scores = pd.merge(self.cbf_scores_table, self.knn_scores, on=["SITE_NAME", "genus"], how="inner")
        self.scores['hybrid_similarity'] = content_based_weight * self.scores['cbf_similarity'] + knn_weight * self.scores['knn_similarity']

        self.hybrid_scores = self.scores.pivot(index="SITE_NAME", columns="genus", values="hybrid_similarity")
    

    def __calculate_hybrid_score_filter(self, filter_threshold):
        """
        Calculates the hybrid score using the filter method

        Parameters:
        -----------
        filter_threshold: float
            A threshold value for kNN Collaborative filtering similarity score. Values below the threshold are assigned to 0 in hybrid similarity if using
            'filter' or 'filter_average' method.

        Returns
        -------
        None
        """

        self.scores = pd.merge(self.cbf_scores_table, self.knn_scores, on=["SITE_NAME", "genus"], how="inner")

        self.scores['hybrid_similarity'] = self.scores.apply(lambda row: row['cbf_similarity'] if row['knn_similarity'] > filter_threshold else 0, axis=1)
        self.hybrid_scores = self.scores.pivot(index="SITE_NAME", columns="genus", values="hybrid_similarity")
    

    def __calculate_hybrid_score_filter_average(self, filter_threshold, content_based_weight):
        """
        Calculates the hybrid score using the filter_average method

        Parameters:
        -----------
        filter_threshold: float
            A threshold value for kNN Collaborative filtering similarity score. Values below the threshold are assigned to 0 in hybrid similarity if using
            'filter' or 'filter_average' method.

        content_based_weight: float
            A weight of content-based similarity values in hybrid results. Must be between 0 and 1. This is used if method is either 'average' or 'filter_average'.
        
        Returns
        -------
        None
        """

        knn_weight = 1 - content_based_weight
    
        self.scores = pd.merge(self.cbf_scores_table, self.knn_scores, on=["SITE_NAME", "genus"], how="inner")

        self.scores['hybrid_similarity'] = self.scores.apply(
            lambda row: 
            content_based_weight * row['cbf_similarity'] + knn_weight * row['knn_similarity'] 
            if row['knn_similarity'] > filter_threshold else 0, 
            axis=1
        )

        self.hybrid_scores = self.scores.pivot(index="SITE_NAME", columns="genus", values="hybrid_similarity")


    def get_recommendations(self, matrix_form:bool=True) -> pd.DataFrame:
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


    def predict(self, test_set: pd.DataFrame) -> pd.DataFrame:
        """
        Gives a DataFrame with true values of the test set and predicted values from the fit

        Parameters:
        -----------
        test_set: pd.DataFrame
            A Pandas DataFrame containing the test set. The DataFrame must contain columns SITE_NAME, genus, occurence.

        Returns:
        --------
        pd.DataFrame
            A Pandas DataFrame containing columns SITE_NAME, genus, occurence and (predicted) similarity.
        """

        df_test = test_set.merge(
            self.scores, on=["SITE_NAME", "genus"], how="left"
        ).rename(
            columns={"hybrid_similarity": "similarity"}
        ).sort_values(
            by=["SITE_NAME", "similarity"], ascending=[True, False]
        ).drop(
            columns=["cbf_similarity", "knn_similarity"]
        )

        return df_test

#%%
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
    hybrid.fit(occurences, site_info, df_genus_info, k=10, min_k=1, normalization='min-max', method="filter_average")

    print(hybrid.get_recommendations())
# %%
