# FossilNOW

The FossilNOW app allows user to predict missing genera at fossil sites. As fossilization is a rare process, there is a high possiblity that on the sites where fossils are found, not all the genera that have lived there have become fossils. The FossilNOW uses recommender algorithms to try to predict the missing genera for the sites. Also, the app makes visualizations of the real occurences and the predictions. The app uses a csv file that the user gives as an input. The input should include the information about the genera occurences at each site.

We consider this recommendation problem is *postive-only rating* (i.e. we only know the occurence of somes genera in specific sites and it is indicated as greater-than-zero values, the 0s imply that we haven't know yet whether or not genus occurs in that site).

## Recommender Algorithms

FossilNow supports different recommendation algorithms for genera-site recommendation such as:

- Content-based filtering
- Matrix Factorization
  - Estimate the embedding matrix of genera and site; available in both mode `output_probability = True` and `output_probability = False`, the predicted score is obtained by multiplying 2 embedding matrices.
2 modes correspond to two way of considering the values in the training dataframe. If we consider cells' value is the probability of occurence, then it corresponds to the mode `output_probability = True`. On the contrary, if the cells' value only indicate the bare occurence, it corresponds to the mode `output_probability = False`
- KNN
  - Use the learned embeddings from **Matrix Factorization** to filter the similar genera and average the know occurences of similar genera to predict the occurence of the given genus.
- kNN Collaborative Filtering
- Content-based Filtering and kNN Collaborative Filtering hybrid algorithm


## Instructions

1. Install packages with pip by running:

    `pip install -r requirements.txt`

    or with conda by running:

    `conda env create -f environment.yml`

    and activate the environment with:

    `conda activate FossilNOW`


2. Go to the source directory `src` and run locally:

    `python app.py`

    The app starts in a web browser.


3. Upload the files:
- a csv file containing the genera occurences at sites in a matrix form (containing 0s and 1s). This file should include also information about the site (longitude, latitude etc) as last columns. The number of site information columns should be spesified in the app. The name of the site should be the first column
- a csv file containing information about the genera. This should contain columns: 'Genus','LogMass', 'HYP_Mean' and 'LOP_Mean'
- For the Content-Based Filtering and the hybrid algorithm the site and genera metadata should be preprocessed so that the categorical columns are one-hot-encoded.

4. Select the recommender algorithm

5. Check the validation results

6. Visualize the results

7. Evaluation metrics

As we consider the problem as *postive-only rating* recommendation, we employ 3 methods for evaluation:
    1. Expected Percentile Rank: 
        - Range: 0 - 100; lower is better ; EPR = 50 indicates that the algorihm is no better than random algorithm.
        - Meaning: indicates how likely the sites that the genus actually appear are on the top of the recommended list of sites.
        - Availability: With algorithm **MF** and **KNN**, this metric is available in both modes `output_probability = True` and `output_probability = False`
    2. True Positive Rate:
        - Range: 0 - 100; higher is better
        - Meaning: indicates how likely the genus appears in a site.
        - Availability: With algorithm **MF** and **KNN**, this metric is available only in mode `output_probability = True`
    3. True Negative Rate:
        - Range: 0 - 100; lower is better
        - Meaning: indicates how unlikely the genus appears in a site.
        - Availability: With algorithm **MF** and **KNN**, this metric is available only in mode `output_probability = True` and the flag `include_tnr` is set to True. Moreover, this metric is done on separated dataframe, not one used for training.

