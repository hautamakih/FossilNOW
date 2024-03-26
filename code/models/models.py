import pandas as pd
from models.knn import knn
from models.MF import mf
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


def get_recommend_list_content_base(
    df: pd.DataFrame, genera_df: pd.DataFrame
) -> pd.DataFrame:
    ## Divide data into training and testing
    df_train, df_test = divide()

    ## Train with df_train
    train(df_train)

    ## Evaluate
    evaluate(df_test)

    ## Predict on entire df
    df = predict()

    return df


def get_metrics_content_base() -> dict:
    return {
        "tpr": 0.9,
        "expected_rank": 0.34,
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
