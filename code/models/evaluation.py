import numpy as np
import pandas as pd
from scipy import stats


def f(df_species: pd.DataFrame):
    if np.sum(df_species["occurence"]) == 0:
        return 0

    preferences = df_species["pred"]
    percentile_rank = 1 - stats.percentileofscore(preferences, preferences) / 100

    expected_rank_species = np.sum(df_species["occurence"] * percentile_rank) / np.sum(
        df_species["occurence"]
    )

    return expected_rank_species


def calc_expected_percentile_rank(df_pred: pd.DataFrame) -> float:
    """Calculate the expected percentile rank as in paper "Collaborative Filtering for Implicit Feedback Datasets"

    Args:
        df_pred (pd.DataFrame): prediction dataframe

    Returns:
        float: expected percentile rank
    """
    expected_ranks = (
        df_pred.sort_values(by="pred")
        .groupby(by="species")
        .apply(f, include_groups=False)
    )
    expected_percentile_rank = expected_ranks[expected_ranks > 0].mean()

    return expected_percentile_rank


def calc_tpr(pred: pd.DataFrame, thres_occ: float = 0.5) -> float:
    """Calculate the True Positive Rate as treating the problem as one-class classification

    Args:
        pred (list | np.ndarray | Series): predicted probabbility
        thres_occ (float, optional): probability threshold of occurence. Defaults to 0.5.

    Returns:
        float: result
    """
    assert "occurence" in pred and "pred" in pred

    pred_all1 = pred[pred["occurence"] == 1]["pred"]

    return (np.sum(pred_all1 >= thres_occ) * 1.0 / len(pred)).item()
