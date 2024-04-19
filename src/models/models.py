import pandas as pd
from models.knn import knn
from models.MF import mf
from models.CBF.ContentBasedFiltering import ContentBasedFiltering
from models.Hybrid.CBF_CF_hybrid import CbfCfHybrid
from pandas import DataFrame
from surprise import KNNBasic, Dataset, Reader

from . import evaluation, utils


def get_recommend_list_mf(
    dataframe: DataFrame,
    output_prob: bool = True,
    num_epochs: int = 100,
    dim_hid: int = 10,
) -> DataFrame:
    df_train, df_test = utils.split_traintest(
        dataframe, is_packed=True, is_encoded=True
    )

    df_out = mf.trigger_train(
        df_train, df_test, num_epochs=num_epochs, output_prob=output_prob, dim_hid=dim_hid
    )
    return df_out


def get_metrics_mf(dataframe: pd.DataFrame, output_prob: bool = True) -> dict:
    _, df_test = utils.split_traintest(dataframe, is_packed=True, is_encoded=True)

    metrics = mf.trigger_test(df_test, output_prob=output_prob)
    return metrics


def get_recommend_list_knn(
    dataframe: pd.DataFrame, output_prob: bool = True, topK: int = 15
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
    dataframe: pd.DataFrame, output_prob: bool = True, topK: int = 15
) -> dict:
    df_train_, df_test_ = utils.split_traintest(
        dataframe, is_packed=True, is_encoded=True
    )
    df_train = utils.conv_dataset_patch2df(df_train_.to_records())
    df_test = utils.conv_dataset_patch2df(df_test_.to_records())

    df_out = knn.core(
        df_train, df_test, is_predict=False, output_prob=output_prob, topK=topK
    )

    return {
        "expected_percentile_rank": evaluation.calc_expected_percentile_rank(df_out),
        "true_positive_rate": evaluation.calc_tpr(df_out),
    }

cbf = ContentBasedFiltering()
def get_recommend_list_content_base(
    df: pd.DataFrame, site_df: pd.DataFrame, genera_df: pd.DataFrame
) -> pd.DataFrame:
    # Divide data into training and testing
    df_train, df_test = utils.split_traintest(df, is_packed=False, is_encoded=False)

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
    cbf.fit(df_train, site_df, genera_df, normalization="min-max")

    ## Predict scores
    df = cbf.get_recommendations()

    return df


def get_metrics_content_base(dataframe: pd.DataFrame, output_prob: bool = True) -> dict:
    # Get predictions for all user-item pairs
    if not output_prob:
        raise NotImplementedError()

    # Divide data into training and testing
    df_train, df_test = utils.split_traintest(
        dataframe, is_packed=False, is_encoded=False
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


def get_recommend_list_colab(dataframe: pd.DataFrame, k=5, min_k=1, sim_options={'name': "MSD",'user_based': True}) -> pd.DataFrame:
    df_train, df_test = utils.split_traintest(dataframe, is_packed=False, is_encoded=False)

    occurences = df_train.rename(columns={df_train.columns[0]: "SITE_NAME"})

    # Fit the kNN Collaborative Filtering
    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(occurences, reader) # Column order must be user, item, rating

    trainset = data.build_full_trainset()

    global knn
    knn = KNNBasic(k=k, min_k=min_k, sim_options=sim_options)
    knn.fit(trainset)

    testset = trainset.build_testset()

    # Get predictions for all user-item pairs
    predictions = knn.test(testset)

    # Get item scores from the predictions
    item_scores = [(prediction.uid, prediction.iid, prediction.est) for prediction in predictions]

    global knn_scores
    knn_scores = pd.DataFrame(item_scores, columns =['SITE_NAME', 'genus', 'similarity'])
    knn_scores_pivot = knn_scores.pivot(index="SITE_NAME", columns="genus", values="similarity").fillna(0)

    return knn_scores_pivot


def get_metrics_colab(dataframe: pd.DataFrame, output_prob: bool = True) -> dict:
    # Get predictions for all user-item pairs
    if not output_prob:
        raise NotImplementedError()

    # Divide data into training and testing
    df_train, df_test = utils.split_traintest(
        dataframe, is_packed=False, is_encoded=False
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
        method: str="average",
        content_based_weight: float=0.5,
        filter_threshold: float=0.01, 
        normalization: str="min-max", 
        sim_options: dict={'name': "MSD",'user_based': True}
    ) -> pd.DataFrame:

    # Divide data into training and testing
    df_train, df_test = utils.split_traintest(df, is_packed=False, is_encoded=False)

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
        sim_options=sim_options)

    ## Predict scores
    df = hybrid.get_recommendations()

    return df

def get_metrics_hybrid(dataframe: pd.DataFrame, output_prob: bool = True) -> dict:
    # Get predictions for all user-item pairs
    if not output_prob:
        raise NotImplementedError()

    # Divide data into training and testing
    df_train, df_test = utils.split_traintest(
        dataframe, is_packed=False, is_encoded=False
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
