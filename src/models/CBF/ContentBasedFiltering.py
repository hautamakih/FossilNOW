# %%
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

# %%


class ContentBasedFiltering:
    """
    Content Based Filtering algorithm for recommending genera to sites.

    Attibutes
    ---------
    None

    Methods
    -------
    fit(occurences, site_data, genus_data, normalization):
        Fits the algorithm on given data

    get_recommendations(matrix_form=True):
        Gives the similarity scores for all the genus-site pairs

    predict(test_set):
        Gives a DataFrame with true values of the test set and predicted values from the fit
    """

    def __init__(self):
        pass

    def fit(self,
        occurences: pd.DataFrame,
        site_data: pd.DataFrame,
        genus_data: pd.DataFrame,
        normalization: str = "min-max",
        occurence_threshold: float = 0.8
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

        normalization: str
            The type of normalization used to normalize columns before calculating the similarity scores. Possible values: ["min-max", "mean"]. The default value is min-max.
        
        occurence_threshold: float
            A threshold value that tells the algorithm which values are handled as occurences in the occurence data. This is needed if your occurence data has values between 0 and 1 (not just 0s and 1s).
            The algorithm cannot handle uncertancies in the data and hence values over or equal to the threshold are handled as occurence. The default value is 0.8.

        Returns
        -------
        None
        """

        print("Fitting Content-based filtering algorithm...")
        # Changing occurence values above threshold to ones.
        numerical_cols = occurences.select_dtypes(include='number').columns
        occurences.loc[:, numerical_cols] = occurences.loc[:, numerical_cols].apply(lambda x: x.mask(x > occurence_threshold, 1))


        
        self.site_info = site_data.rename(columns={site_data.columns[0]: "SITE_NAME"}) # Saving the site data to a class variable and giving name "SITE_NAME" to the first column
        self.genus_info = genus_data.rename(columns={genus_data.columns[0]: "genus"}) # Saving the genus data to a class variable and giving name "genus" to the first column
        self.site_genus_matrix = occurences.rename(columns={occurences.columns[0]: "SITE_NAME"}).set_index("SITE_NAME") # Saving the occurences to a class variable and giving name "SITE_NAME" to the first column and assigning it to index
        occurences = occurences.rename(columns={occurences.columns[0]: "SITE_NAME"}) # Renaming the first column to site name

        self.__build_genus_related_site_info()
        self.__build_genus_info_with_site_info()
        self.__build_site_info_with_genus_info()
        self.__find_recommendations_for_all_sites(
            occurences, normalization=normalization
        )
        print("Contend-based filtering fit complete.")

    def get_recommendations(self, matrix_form: bool = True) -> pd.DataFrame:
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
            return self.recommendation_matrix
        else:
            return self.recommendations

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
            self.recommendations, on=["SITE_NAME", "genus"], how="left"
        ).sort_values(by=["SITE_NAME", "similarity"], ascending=[True, False])

        return df_test

    def __build_genus_related_site_info(self):
        """
        Calculates genus related information for sites by calculating means of the genera features from genera that are present at the site. Saved as class variable

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """

        genus_info = self.genus_info
        site_genus = self.site_genus_matrix

        site_genus = (
            site_genus.stack()
            .reset_index()
            .rename(columns={"level_1": "genus", 0: "presence"})
        )
        site_genus = site_genus[site_genus["presence"] == 1].drop(
            "presence", axis="columns"
        )

        site_genus = site_genus.merge(genus_info, on="genus", how="left")
        site_genus = site_genus.drop(["genus"], axis=1)

        # Calculate using mean, max, mode, median?
        site_genus = (
            site_genus.groupby("SITE_NAME").mean().reset_index().set_index("SITE_NAME")
        )

        self.genus_related_site_info = site_genus

    def __build_genus_info_with_site_info(self):
        """
        Adds site related information to genus information by calculating means of the site features from sites that the genus is present. Saved as class variable

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """

        site_genus = self.site_genus_matrix
        site_genus = (
            site_genus.stack()
            .reset_index()
            .rename(columns={"level_1": "genus", 0: "presence"})
        )
        site_info = self.site_info

        genus_info = site_genus.merge(site_info, on="SITE_NAME", how="left")
        genus_info = genus_info[genus_info["presence"] == 1]

        genus_info = genus_info.drop(["SITE_NAME", "presence"], axis=1)
        genus_info = genus_info.groupby("genus").mean().reset_index()

        df_genus_data = self.genus_info
        genus_info = (
            genus_info.merge(
                df_genus_data, on="genus", how="left"
            )
            .reset_index(drop=True)
            .set_index("genus")
        )

        self.genus_info_with_site_info = genus_info

    def __build_site_info_with_genus_info(self):
        """
        Adds genus related information for sites by calculating means of the genera features from genera that are present at the site. Saved as class variable

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """

        df_site_info = self.site_info.set_index("SITE_NAME")

        df_site_info_by_genera = self.genus_related_site_info
        df_site_info = df_site_info.merge(
            df_site_info_by_genera, left_index=True, right_index=True, how="left"
        )

        self.site_info_with_genus_info = df_site_info

    @staticmethod
    def __normalize_columns_min_max(column: pd.Series):
        """
        Normalizes the DataFrame columns using min-max method

        Parameters:
        -----------
        df: pd.Series
            A Pandas DataFrame column to be normalized

        Returns:
        --------
        df: pd.Series
            The normalized DataFrame column
        """
        min_val = column.min()
        max_val = column.max()
        if min_val == max_val:
            return column
        else:
            return (column - min_val) / (max_val - min_val)

    @staticmethod
    def __normalize_columns_mean(column: pd.Series):
        """
        Normalizes the DataFrame columns using mean and standard deviation

        Parameters:
        -----------
        df: pd.DataFrame
            A Pandas DataFrame column to be normalized

        Returns:
        --------
        df: pd.DataFrame
            The normalized DataFrame column
        """
        std = column.std()
        mean = column.mean()
        if std == 0:
            return column
        else:
            return (column - mean) / std

    def __get_recommendations_for_site(
        self,
        genus_info: pd.DataFrame,
        site_name: str,
        site_indices: pd.Series,
        genus_site_similarity_matrix: np.array,
    ):
        """
        Calculates similarity scores to a spesified site

        Parameters:
        -----------
        genus_info: pd.DataFrame
            A Pandas DataFrame containing the info about genuses with info calculated from the sites

        site_name: str
            The name of the site the recommendations are wanted to be genrated

        site_indices: pd.Seris
            A Series of indices of the genus_info DataFrame

        genus_site_similarity_matrix: np.array
            A numpy array containing the similarities of genus-site pairs

        Returns:
        --------
        recommended_genus: pd.DataFrame
            A DataFrame containing the recommendated genera and the similarity scores
        """

        idx = site_indices[site_name]

        # Sorted similarity scores
        sim_scores = sorted(
            list(enumerate(genus_site_similarity_matrix[:, idx])),
            key=lambda x: x[1],
            reverse=True,
        )

        # Get the scores of the num_recommend most similar sites
        similar_genus_for_site = sim_scores

        # Get the genus indices
        genus_indices = [i[0] for i in similar_genus_for_site]
        genus_site_similarities = [i[1] for i in similar_genus_for_site]

        recommended_genus = (
            genus_info.iloc[genus_indices]
            .index.to_frame(index=False)
            .assign(similarity=genus_site_similarities)
        )
        recommended_genus.insert(0, "SITE_NAME", site_name)

        return recommended_genus

    def __find_recommendations_for_all_sites(
        self, df: pd.DataFrame, normalization: str
    ):
        """
        Calculates the similarity scores for all the genus-site pairs.

        Parameters:
        -----------
        df: pd.DataFrame
            A Pandas DataFrame containing the site-genus matrix

        normalization: str
            The type of normalization used to normalize columns before calculating the similarity scores. Possible values: ["min-max", "mean"].

        Returns:
        --------
        None
        """

        genus_info = self.genus_info_with_site_info
        site_info = self.site_info_with_genus_info

        # Normalizing columns
        if normalization == "min-max":
            normalization = self.__normalize_columns_min_max
        elif normalization == "mean":
            normalization = self.__normalize_columns_mean
        else:
            raise ValueError("The normalization must be either 'min-max' or 'mean'.")

        genus_info = genus_info.apply(normalization)
        site_info = site_info.apply(normalization)

        # The DataFrames must not contain Nans
        if genus_info.isnull().values.any():
            print("WARNING! Genus info data contains nans. Assigning to zeros")
            genus_info = genus_info.fillna(0)

        if site_info.isnull().values.any():
            print("WARNING! Site info data contains nans. Assigning to zeros")
            site_info = site_info.fillna(0)

        # Finding the indices of the sites so the right row can be found from the numpy arrya
        site_indices = pd.Series(df.index, index=df["SITE_NAME"]).drop_duplicates()
        sim = cosine_similarity(genus_info, site_info)

        # Looping through all the sites and finding the similarity scores
        recommendations = []
        for site, idx in site_indices.items():
            site_recommendations = self.__get_recommendations_for_site(
                genus_info=genus_info,
                site_name=site,
                site_indices=site_indices,
                genus_site_similarity_matrix=sim,
            )

            recommendations.append(site_recommendations)

        self.recommendations = pd.concat(recommendations).reset_index(drop=True)
        self.recommendation_matrix = pd.pivot(
            self.recommendations,
            index="SITE_NAME",
            columns="genus",
            values="similarity",
        ).fillna(0)


# %%
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

    cbf = ContentBasedFiltering()
    cbf.fit(occurences, site_info, df_genus_info, normalization="min-max")

    print(cbf.get_recommendations())
# %%
