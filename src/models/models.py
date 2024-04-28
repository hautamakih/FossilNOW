import pandas as pd
from models.CBF.ContentBasedFiltering import ContentBasedFiltering
from models.Hybrid.CBF_CF_hybrid import CbfCfHybrid
from models.knn import knn
from models.MF import mf
from pandas import DataFrame
from surprise import Dataset, KNNBasic, Reader

from . import evaluation, utils

device = "cpu"
# if torch.backends.mps.is_available():
#     device = "mps"
# elif torch.cuda.is_available():
#     device = "cuda"


def get_recommend_list_mf(
    dataframe: DataFrame,
    output_prob: bool = True,
    num_epochs: int = 100,
    dim_hid: int = 10,
    lr: float = 5e-3,
) -> DataFrame:
    df_train, df_test = utils.split_traintest(
        dataframe, is_packed=True, is_encoded=True
    )

    df_out = mf.trigger_train(
        df_train,
        df_test,
        num_epochs=num_epochs,
        output_prob=output_prob,
        dim_hid=dim_hid,
        device=device,
        lr=lr,
    )
    return df_out


def get_metrics_mf(
    dataframe: pd.DataFrame,
    output_prob: bool = True,
    dim_hid: int = 10,
    include_tnr: bool = False,
) -> dict:
    """Test trained MF algorithm and calculate metrics

    Args:
        dataframe (pd.DataFrame): testing dataset
        output_prob (bool, optional): which model type to test: model outputing probability or not. Defaults to True.
        include_tnr (bool, optional): whether calculating True Negative Rate. Defaults to True.

    Returns:
        dict: evaluating metrics
    """

    if include_tnr is False:
        _, df_test = utils.split_traintest(dataframe, is_packed=True, is_encoded=True)
    else:
        df_test = dataframe

    metrics = mf.trigger_test(
        df_test,
        output_prob=output_prob,
        device=device,
        dim_hid=dim_hid,
        include_tnr=include_tnr,
    )
    return metrics


def get_recommend_list_knn(
    dataframe: pd.DataFrame,
    output_prob: bool = True,
    topK: int = 15,
) -> DataFrame:
    df_train_, df_test_ = utils.split_traintest(
        dataframe, is_packed=True, is_encoded=True
    )
    df_train = utils.conv_dataset_patch2df(df_train_.to_records())
    df_test = utils.conv_dataset_patch2df(df_test_.to_records())

    # Load necessary things
    enc_genera = utils.CategoryDict.from_file(utils.paths.encoding_genera)
    enc_site = utils.CategoryDict.from_file(utils.paths.encoding_sites)

    df_out = knn.core(
        df_train, df_test, is_predict=True, output_prob=output_prob, topK=topK
    )
    df_out.loc[:, "genus"] = enc_genera.ids2names(df_out["genus"])
    df_out.loc[:, "site"] = enc_site.ids2names(df_out["site"])

    df_out = df_out.pivot(index="site", columns="genus", values="pred")

    return df_out


def get_metrics_knn(
    dataframe: pd.DataFrame,
    output_prob: bool = True,
    topK: int = 15,
    include_tnr: bool = False,
) -> dict:
    df_train_, df_test_ = utils.split_traintest(
        dataframe, is_packed=True, is_encoded=True
    )
    df_train = utils.conv_dataset_patch2df(df_train_.to_records())
    df_test = utils.conv_dataset_patch2df(df_test_.to_records())

    df_out = knn.core(
        df_train, df_test, is_predict=False, output_prob=output_prob, topK=topK
    )

    epr = evaluation.calc_expected_percentile_rank(df_out)
    tpr, tnr = -1, -1
    if output_prob is True:
        tpr = evaluation.calc_tpr(df_out)

        if include_tnr is True:
            tnr = evaluation.calc_tnr(df_out)
    return {
        "expected_percentile_rank": epr,
        "true_positive_rate": tpr,
        "true_negative_rate": tnr,
    }


cbf = ContentBasedFiltering()


def get_recommend_list_content_base(
    df: pd.DataFrame, site_df: pd.DataFrame, genera_df: pd.DataFrame, occurence_threshold: float = 0.8, train_size=0.8
) -> pd.DataFrame:
    """
    Fits the Content Based Filtering algorithm on given data and makes predictions.

    Parameters:
    -----------
    df: pd.DataFrame
        a Pandas DataFrame containing occurences of genera at each site in a matrix form. The first column must be the site name.

    site_df: pd.DataFrame
        a Pandas DataFrame containing the site metadata. The first column must be the site name. 
        Categorical variables must be converted to numerical beforehand.

    genera_df: pd.DataFrame
        a Pandas DataFrame containing the information about the genera. Categorical features must be converted using one-hot-encoding beforehand. 
        The first column must be the genus name. Categorical variables must be converted to numberical beforehand.
    
    occurence_threshold: float
        A threshold value that tells the algorithm which values are handled as occurences in the occurence data. This is needed if your occurence data has values between 0 and 1 (not just 0s and 1s).
        The algorithm cannot handle uncertancies in the data and hence values over or equal to the threshold are handled as occurence. The default value is 0.8.
    
    train_size: float
        The size of the data set used for training. The default is 0.8.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the prediction values (between 0 and 1) for each site-genus pair.
    """

    # Divide data into training and testing
    df_train, df_test = utils.split_traintest(df, is_packed=False, is_encoded=False, ratio_traintest=train_size)

    # Converting the training data into matrix form
    train_cols = df_train.columns.to_list()
    df_train = (
        pd.pivot(
            df_train, index=train_cols[0], columns=train_cols[1], values=train_cols[2]
        )
        .fillna(0)
        .reset_index()
    )

    ## Train with df_train
    global cbf
    cbf.fit(df_train, site_df, genera_df, normalization="min-max", occurence_threshold=occurence_threshold)

    ## Predict scores
    df = cbf.get_recommendations()

    return df


def get_metrics_content_base(dataframe: pd.DataFrame, output_prob: bool = True, train_size=0.8) -> dict:
    """
    Calculates the metrics for the Content Based Filtering

    Parameters:
    -----------
    dataframe: pd.DataFrame
        a Pandas DataFrame containing occurences of genera at each site in a matrix form. The first column must be the site name.
    
    output_prob: bool
        Included for consistency but not implemented.
    
    train_size: float
        The size of the data set used for training. The value must be the same as the one that was used for fitting.

    Returns:
    --------
    dictionary
        A dictionary containing the expected percentile rank and true postive rate
    """

    # Get predictions for all user-item pairs
    if not output_prob:
        raise NotImplementedError()

    # Divide data into training and testing
    df_train, df_test = utils.split_traintest(
        dataframe, is_packed=False, is_encoded=False, ratio_traintest=train_size
    )

    df_test = df_test.rename(columns={"site": "SITE_NAME"})

    predictions = cbf.predict(df_test)
    predictions = predictions.fillna(0).rename(columns={"similarity": "pred"})

    return {
        "expected_percentile_rank": evaluation.calc_expected_percentile_rank(
            predictions
        ),
        "true_positive_rate": evaluation.calc_tpr(predictions),
    }


def get_recommend_list_colab(
    dataframe: pd.DataFrame,
    k=5,
    min_k=1,
    sim_options={"name": "MSD", "user_based": True},
    train_size=0.8,
) -> pd.DataFrame:
    df_train, df_test = utils.split_traintest(
        dataframe, is_packed=False, is_encoded=False, ratio_traintest=train_size
    )
    """
    Fits the knn Collaborative Filtering algorithm on given data and makes predictions

    Parameters:
    -----------
    dataframe: pd.DataFrame
        a Pandas DataFrame containing occurences of genera at each site in a matrix form. The first column must be the site name.

    k: int
        The maximum number of neigbours used by the kNN Collaborative filter algorithm

    min_k: int
        The minimum number of neigbours used by the kNN Collaborative filter algorithm

    sim_options: dict
        The similarity parameters used by the kNN Collaborative filtering. See the Surprise documentation if you want to change this.
    
    train_size: float
        The size of the data set used for training. The default is 0.8.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the prediction values (between 0 and 1) for each site-genus pair.

    References:
    -----------
    Surprise documentation for similarity options:
        https://surprise.readthedocs.io/en/stable/similarities.html
    """

    occurences = df_train.rename(columns={df_train.columns[0]: "SITE_NAME"})

    # Fit the kNN Collaborative Filtering
    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(
        occurences, reader
    )  # Column order must be user, item, rating

    trainset = data.build_full_trainset()

    global knn
    knn = KNNBasic(k=k, min_k=min_k, sim_options=sim_options)
    knn.fit(trainset)

    testset = trainset.build_testset()

    # Get predictions for all user-item pairs
    predictions = knn.test(testset)

    # Get item scores from the predictions
    item_scores = [
        (prediction.uid, prediction.iid, prediction.est) for prediction in predictions
    ]

    global knn_scores
    knn_scores = pd.DataFrame(item_scores, columns=["SITE_NAME", "genus", "similarity"])
    knn_scores_pivot = knn_scores.pivot(
        index="SITE_NAME", columns="genus", values="similarity"
    ).fillna(0)

    return knn_scores_pivot


def get_metrics_colab(dataframe: pd.DataFrame, output_prob: bool = True, train_size=0.8) -> dict:
    """
    Calculates the metrics for the knn Collaborative Filtering

    Parameters:
    -----------
    dataframe: pd.DataFrame
        a Pandas DataFrame containing occurences of genera at each site in a matrix form. The first column must be the site name.
    
    output_prob: bool
        Included for consistency but not implemented.
    
    train_size: float
        The size of the data set used for training. The value must be the same as the one that was used for fitting.

    Returns:
    --------
    dictionary
        A dictionary containing the expected percentile rank and true postive rate
    """
    # Get predictions for all user-item pairs
    if not output_prob:
        raise NotImplementedError()

    # Divide data into training and testing
    df_train, df_test = utils.split_traintest(
        dataframe, is_packed=False, is_encoded=False, ratio_traintest=train_size
    )

    df_test = df_test.rename(columns={df_train.columns[0]: "SITE_NAME"})

    predictions = df_test.merge(
        knn_scores, on=["SITE_NAME", "genus"], how="left"
    ).sort_values(by=["SITE_NAME", "similarity"], ascending=[True, False])

    predictions = predictions.fillna(0).rename(columns={"similarity": "pred"})

    return {
        "expected_percentile_rank": evaluation.calc_expected_percentile_rank(
            predictions
        ),
        "true_positive_rate": evaluation.calc_tpr(predictions),
    }


hybrid = CbfCfHybrid()


def get_recommend_list_hybrid(
    df: pd.DataFrame,
    site_df: pd.DataFrame,
    genera_df: pd.DataFrame,
    k: int,
    min_k: int,
    method: str = "average",
    content_based_weight: float = 0.5,
    filter_threshold: float = 0.01,
    normalization: str = "min-max",
    occurence_threshold: float = 0.8,
    sim_options: dict = {"name": "MSD", "user_based": True},
    train_size=0.8,
) -> pd.DataFrame:
    """
    Fits the Hybrid algorithm on given data and makes predictions

    Parameters:
    -----------
    df: pd.DataFrame
        a Pandas DataFrame containing occurences of genera at each site in a matrix form. The first column must be the site name.

    site_df: pd.DataFrame
        a Pandas DataFrame containing the site metadata. The first column must be the site name.
        Categorical variables must be converted to numberival beforehand.

    genera_df: pd.DataFrame
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
    
    occurence_threshold: float
        A threshold value that tells the algorithm which values are handled as occurences in the occurence data. This is needed if your occurence data has values between 0 and 1 (not just 0s and 1s).
        The algorithm cannot handle uncertancies in the data and hence values over or equal to the threshold are handled as occurence. The default value is 0.8.

    sim_options: dict
        The similarity parameters used by the kNN Collaborative filtering. See the Surprise documentation if you want to change this.

    train_size: float
        The size of the data set used for training. The default is 0.8.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the prediction values (between 0 and 1) for each site-genus pair.

    References:
    -----------
    Surprise documentation for similarity options:
        https://surprise.readthedocs.io/en/stable/similarities.html
    """

    # Divide data into training and testing
    df_train, df_test = utils.split_traintest(df, is_packed=False, is_encoded=False, ratio_traintest=train_size)

    # Converting the training data into matrix form
    train_cols = df_train.columns.to_list()
    df_train = (
        pd.pivot(
            df_train, index=train_cols[0], columns=train_cols[1], values=train_cols[2]
        )
        .fillna(0)
        .reset_index()
    )

    # Train with df_train
    global hybrid
    hybrid.fit(
        df_train,
        site_df,
        genera_df,
        k=k,
        min_k=min_k,
        method=method,
        content_based_weight=content_based_weight,
        filter_threshold=filter_threshold,
        normalization=normalization,
        occurence_threshold=occurence_threshold,
        sim_options=sim_options,
    )

    ## Predict scores
    df = hybrid.get_recommendations()

    return df


def get_metrics_hybrid(dataframe: pd.DataFrame, output_prob: bool = True, train_size=0.8) -> dict:
    """
    Calculates the metrics for the Hybrid algorithm

    Parameters:
    -----------
    dataframe: pd.DataFrame
        a Pandas DataFrame containing occurences of genera at each site in a matrix form. The first column must be the site name.
    
    output_prob: bool
        Included for consistency but not implemented.
    
    train_size: float
        The size of the data set used for training. The value must be the same as the one that was used for fitting.

    Returns:
    --------
    dictionary
        A dictionary containing the expected percentile rank and true postive rate
    """
    # Get predictions for all user-item pairs
    if not output_prob:
        raise NotImplementedError()

    # Divide data into training and testing
    df_train, df_test = utils.split_traintest(
        dataframe, is_packed=False, is_encoded=False, ratio_traintest=train_size
    )

    df_test = df_test.rename(columns={"site": "SITE_NAME"})

    predictions = hybrid.predict(df_test)
    predictions = predictions.fillna(0).rename(columns={"similarity": "pred"})

    return {
        "expected_percentile_rank": evaluation.calc_expected_percentile_rank(
            predictions
        ),
        "true_positive_rate": evaluation.calc_tpr(predictions),
    }


# 2 input files:
# - DentalTraits_Genus_PPPA
# - data_occ
