# FossilNOW

The FossilNOW app allows user to predict missing genera at fossil sites. As fossilization is a rare process, there is a high possiblity that on the sites where fossils are found, not all the genera that have lived there have become fossils. The FossilNOW uses recommender algorithms to try to predict the missing genera for the sites. Also, the app makes visualizations of the real occurences and the predictions. The app uses a csv file that the user gives as an input. The input should include the information about the genera occurences at each site.

We consider this recommendation problem is *postive-only rating* (i.e. we only know the occurence of somes genera in specific sites and it is indicated as greater-than-zero values, the 0s imply that we haven't know yet whether or not genus occurs in that site).

## Recommender Algorithms

FossilNow supports different recommendation algorithms for genera-site recommendation such as:

- Content-based filtering
  - Uses the site and genera metadata to find the most similar genera to each site.
- Matrix Factorization
  - Estimate the embedding matrix of genera and site; available in both mode `output_probability = True` and `output_probability = False`, the predicted score is obtained by multiplying 2 embedding matrices.
2 modes correspond to two way of considering the values in the training dataframe. If we consider cells' value is the probability of occurence, then it corresponds to the mode `output_probability = True`. On the contrary, if the cells' value only indicate the bare occurence, it corresponds to the mode `output_probability = False`
- KNN
  - Use the learned embeddings from **Matrix Factorization** to filter the similar genera and average the know occurences of similar genera to predict the occurence of the given genus.
- kNN Collaborative Filtering
  - Finds the k most similar sites to each site and uses their occurences to predict the missing genera.
- Content-based Filtering and kNN Collaborative Filtering hybrid algorithm
  - Combines the Content-based Filtering and Collaborative filtering algorithms

## Requirements

This app works with `python3.10` and `python3.11`. `python3.12` will be working as soon as `scikit-surprise` package will support it.

## Instructions

0. Create virtual environment (Optional):

    `python -m venv venv`

    Activate the virtual environment:

    `source venv/bin/activate`

1. Install packages with pip by running:

    `pip install -r requirements.txt`

    or with conda by running:

    `conda env create -f environment.yml`

    and activate the environment with:

    `conda activate FossilNOW`


2. Go to the source directory `src` and run locally:

    `cd src`

    `python app.py`

    The app starts running on http://127.0.0.1:8050


3. Upload the files:
- a csv file containing the genera occurences at sites in a matrix form (containing 0s and 1s or float value between). This file should include also information about the site (longitude, latitude etc) as last columns. The number of site information columns should be specified in the app. The name of the site should be the first column. Name it as "SITE_NAME" (recommended), "NAME" or "loc_name". It is important to extract the site information from this data in the app or otherwise the algorithms will not work as intended. This file must always include longitude, latitude and country for the visualizations to work.
- a csv file containing the dental traits and log masses ('Genus','LogMass', 'HYP_Mean' and 'LOP_Mean'). This is needed for the visualizations
- a csv file containing information about the genera. The first column of the data should be 'Genus'. (optional, used by the content-based filtering and the hybrid)
- a csv file containing the true negatives (optional. Needed if true negative rate is wanted to be calculated)


  **Important note about the data!:**


  The categorical values in csv files containing the occurences (and site metadata) and the genera metadata should be one-hot-encoded as dummy variables. If there are any columns that are not numerical (either integer or float) the content-based filtering and the hybrid algorithms will not work.

4. Select the recommender algorithm

  - The paramters each algorithm uses are described below.

5. Check the validation results

6. Visualize the results

## Parameters of the algorithms

- Common paramters
  - Size of the train data
    - Spesifies the share of the train set size. The occurence data is randomly split into train and test datasets according to this value. If the size of the train data is set to 1, all the data is used for training but evaluation metrics cannot be calculated. Note, that true negatives are not used for training and if they are calculated, the whole data is always used for that.
   - Include true negatives (not implemented for kNN)
     - Tells the app whether the true negative rates are calculated. Please set this to 'No' if the true negative rate csv has not been given as an input.
- Algorithm spesific parameters
  - Matrix Factorization
    - Epochs: the number of training epochs
    - Dim-hid: the dimension of site and genus embedding
    - `output_probability`: set `True` is we want the output is the probability, otherwise the output is the non-negative number.
  - kNN
    - Top k: top `k` sites with highest score
    - `output_probability`: set `True` is we want the output is the probability, otherwise the output is the non-negative number.
  - Content-based filtering
    - Occurence threshold
      - Values above this are counted as occured. Content-based filtering implementation cannot handle uncertain occurences which is why this has to be spesified if the occurence matrix contains other values than 0s and 1s.
  - kNN Collaborative filtering
    - Top k
      - The max number of neighbours (sites) taken into account.
    - Min k
      - The min number of neighbours (sites) taken into account. 
  - Hybrid
    - Occurence threshold
      - Values above this are counted as occured. Content-based filtering implementation cannot handle uncertain occurences which is why this has to be spesified if the occurence matrix contains other values. Used by the content-based filtering part.
    - Method
      - average
        - Calculates the weighted average of the both algorithm predictions.
      - filter
        - Filters the predictions of the content-based filtering predictions by the predictions of the collaborative filtering. If the collaborative filtering prediction is under the spesified threshold, the prediction value is zero, otherwise the prediction of content-based filtering is used.
      - filter_average
        - Combines the above two methods. First calculates the averages and the filters as described above.
    - Weight
      - The weight of content-based similarity values in hybrid results. Must be between 0 and 1. This is used if method is either 'average' or 'filter_average'.
    - Threshold
      - A threshold value for kNN Collaborative filtering similarity score. Values below the threshold are assigned to 0 in hybrid similarity if using 'filter' or 'filter_average' method.
    - Top k
      - The max number of neighbours (sites) taken into account.
    - Min k
      - The min number of neighbours (sites) taken into account. 

## Evaluation metrics

As we consider the problem as *postive-only rating* recommendation, we employ 3 methods for evaluation:

1. Expected Percentile Rank: 
  - Range: 0 - 100; lower is better ; EPR = 50 indicates that the algorihm is no better than random algorithm.
  - Meaning: indicates how likely the sites that the genus actually appear are on the top of the recommended list of sites.
  - Availability: With algorithm **MF** and **KNN**, this metric is available in both modes `output_probability = True` and `output_probability = False`
  
2. True Positive Rate:
  - Range: 0 - 1; higher is better
  - Meaning: indicates how likely the genus appears in a site.
  - Availability: With algorithm **MF** and **KNN**, this metric is available only in mode `output_probability = True`

3. True Negative Rate:
  - Range: 0 - 1; lower is better
  - Meaning: indicates how unlikely the genus appears in a site.
  - Availability: With algorithm **MF** and **KNN**, this metric is available only in mode `output_probability = True` and the flag `include_tnr` is set to True. Moreover, this metric is done on separated dataframe, not one used for training.


## Modifying the histograms

If you wish to plot other columns than the LogMass, HYP_Mean and LOP_Mean, you can do this by modifying the COLUMN1, COLUMN2 and COLUMN3 values in src/callback.py and src/utils/scatter_mapbox.py
