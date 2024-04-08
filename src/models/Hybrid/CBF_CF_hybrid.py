from models.CBF.ContentBasedFiltering import ContentBasedFiltering
from surprise import KNNBaseline

import numpy as np
import pandas as pd

class CbfCfHybrid:
    def __init__(self):
        pass

    def fit(self, site_data: pd.DataFrame, genus_data: pd.DataFrame, n_site_columns: int, normalization: str="min-max"):
        # Fit the Content Based Filtering
        cbf = ContentBasedFiltering()
        cbf.fit(
            site_data = site_data,
            genus_data = genus_data, 
            n_site_columns = n_site_columns, 
            normalization = "min-max"
        )

        self.cbf_scores = cbf.get_recommendations(matrix_form=True)

        # Fit the kNN Collaborative Filtering

        # Combine the results

    def predict(self):
        pass

if __name__ == "__main__":
    pass