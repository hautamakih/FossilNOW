import pandas as pd
from models.knn import knn
from models.MF import mf
from models.CBF.ContentBasedFiltering import ContentBasedFiltering
from pandas import DataFrame

from . import evaluation, utils


def get_recommend_list_mf(
    dataframe: DataFrame,
    output_prob: bool = True,
    num_epochs: int = 100,
) -> DataFrame:
    df_train, df_test = utils.split_traintest(
        dataframe, is_packed=True, is_encoded=True
    )

    df_out = mf.trigger_train(
        df_train, df_test, num_epochs=num_epochs, output_prob=output_prob
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
    df: pd.DataFrame, genera_df: pd.DataFrame, n_site_info_cols: int
) -> pd.DataFrame:
    ## Divide data into training and testing
    df_train, df_test = utils.split_traintest(df, is_packed=False, is_encoded=False)

    # Converting the training data into matrix form
    train_cols = df_train.columns.to_list()
    df_train=pd.pivot(df_train, index=train_cols[0], columns=train_cols[1], values=train_cols[2]).fillna(0).reset_index()

    site_info = pd.merge(df.iloc[:,0],df.iloc[:,-n_site_info_cols:], left_index=True, right_index=True, how="inner")

    # Merging site info to column info, name of site must be the first column
    site_info_cols = site_info.columns.to_list()
    train_cols = df_train.columns.to_list()
    df_train = df_train.merge(site_info, "left", left_on=train_cols[0], right_on=site_info_cols[0]).drop(columns=site_info_cols[0])


    ## Train with df_train
    global cbf
    cbf.fit(df_train, genera_df, n_site_info_cols, normalization="min-max")

    ## Predict scores
    df = cbf.get_recommendations()

    return df


def get_metrics_content_base(dataframe: pd.DataFrame, output_prob: bool=True) -> dict:
    # Get predictions for all user-item pairs
    if not output_prob:
        raise NotImplementedError()
    
    # Divide data into training and testing
    df_train, df_test = utils.split_traintest(dataframe, is_packed=False, is_encoded=False)
    
    df_test = df_test.rename(columns={"site": "SITE_NAME"})

    global cbf
    predictions = cbf.predict(df_test)
    predictions = predictions.fillna(0).rename(columns={"similarity": "pred"})

    return {
        "expected_percentile_rank": evaluation.calc_expected_percentile_rank(predictions),
        "true_positive_rate": evaluation.calc_tpr(predictions),
    }


def get_recommend_list_colab(dataframe: pd.DataFrame, params=None) -> pd.DataFrame:
    pass


def get_metrics_colab() -> dict:
    return {
        "tpr": 0.9,
        "expected_rank": 0.34,
    }


# 2 input files:
# - DentalTraits_Genus_PPPA
# - data_occ
