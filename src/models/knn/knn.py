import numpy as np
import pandas as pd
from pandas import DataFrame

from .. import utils


def core(
    df_train: DataFrame,
    df_test: DataFrame,
    is_predict: bool = True,
    output_prob: bool = True,
    topK: int = 15,
):
    if output_prob is True:
        # path_embd_site = utils.paths.embd_sites_prob_true
        path_embd_genera = utils.paths.embd_genera_prob_true
    else:
        # path_embd_site = utils.paths.embd_sites_prob_false
        path_embd_genera = utils.paths.embd_genera_prob_false
    # embd_sites = np.load(path_embd_site).squeeze()
    embd_genera = np.load(path_embd_genera).squeeze()

    # Calculate cosine-sim
    emb_genera_normed = embd_genera / np.clip(
        np.linalg.norm(embd_genera, axis=1)[:, None], a_max=10, a_min=1e-6
    )

    if is_predict is True:
        df_total = pd.concat([df_train, df_test]).reset_index()
    else:
        df_total = df_test

    for genus in df_total["genus"].unique():
        df_total_genus = df_total[df_total["genus"] == genus]

        emb = emb_genera_normed[genus]
        sim = emb_genera_normed @ emb[:, None]

        # Get top N similar genera
        idx_top = np.argpartition(sim.squeeze(), -topK)[-topK:]

        sites_occ_pred = (
            df_train[
                (df_train["genus"].isin(idx_top))
                & (df_train["site"].isin(df_total_genus["site"]))
            ]
            .groupby(by="site")
            .mean()["occurence"]
        )

        # print(len(df_total_genus))
        # print(sites_occ_pred.values)
        # print(len(df_total.loc[df_total_genus.sort_values(by="site").index]))

        df_total.loc[
            df_total_genus.sort_values(by="site").index, "pred"
        ] = sites_occ_pred.values

    return df_total
