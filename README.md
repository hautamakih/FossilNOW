# FossilNOW

The FossilNOW app allows user to predict missing genera at fossil sites. As fossilization is a rare process, there is a high possiblity that on the sites where fossils are found, not all the genera that have lived there have become fossils. The FossilNOW uses recommender algorithms to try to predict the missing genera for the sites. Also, the app makes visualizations of the real occurences and the predictions. The app uses a csv file that the user gives as an input. The input should include the information about the genera occurences at each site.

## Recommender Algorithms

FossilNow supports different recommendation algorithms for genera-site recommendation such as:

- Content-based filtering
- Matrix Factorization
- KNN
- kNN Collaborative Filtering
- Content-based Filtering and kNN Collaborative Filtering hybrid algorithm


## Instructions

1. Install packages by running:

   `conda install --file requirements.txt` or `pip install -r requirements.txt`

2. Go to the source directory `src` and run locally:

    `python app.py`

    The app starts in a web browser.


3. Upload the files:
- a csv file containing the genera occurences at sites in a matrix form (containing 0s and 1s). This file should include also information about the site (longitude, latitude etc) as last columns. The number of site information columns should be spesified in the app. The name of the site should be the first column
- a csv file containing information about the genera
- For the Content-Based Filtering and the hybrid algorithm the site and genera metadata should be preprocessed so that the categorical columns are one-hot-encoded.

4. Select the recommender algorithm

5. Check the validation results

6. Visualize the results
