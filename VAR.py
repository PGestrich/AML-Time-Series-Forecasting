# Implementation for VAR on entire data set. Unfortunately this VAR method doesnt work for timeseries that
# are constant 0 which is the case for the sales of store 1 and product BABY CARE. Hence this method is not suitable
# to predict the sales for the entire data set.
#imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR, VARMAX
from scipy.stats import wasserstein_distance

#load training data
train = pd.read_csv('datasets/train_extended.csv')
test = pd.read_csv('datasets/test.csv')
train.index = train['date']

num_all_combinations = len(train['store_nbr'].unique()) * len(train['family'].unique())
num_pred_days = 16
result_mat = np.zeros((num_all_combinations,num_pred_days))
stores = list(train['store_nbr'].unique())
families = list(train['family_id'].unique())
ctr = 0
col_names = ['sales', 'oilprice', 'weekday']  # takes those columns to make predictions
start_date_training = '2016-08-15'
data = train[start_date_training:]


# run two for loops to train AutoReg model for each pair of possible compinations (store_nbr,family)
for store in stores:
    for family in families:
        # select store and family
        train_df = data.loc[(data['store_nbr'] == store) & (data['family_id'] == family)][col_names]
        #train the VAR only for sales data that is not equal to 0 for all timesteps
        if not (train_df['sales'] == 0).all():
            var_model = VARMAX(train_df, order=(7, 0), enforce_stationarity=True)
            fitted_model = var_model.fit(disp=False)
            predict = fitted_model.get_prediction(start=len(train_df), end=len(train_df) + num_pred_days - 1)
            predictions = predict.predicted_mean
            predictions = np.round(predictions['sales'], 0)
            result_mat[ctr, :] = np.round(predictions, 0)
        else:
            result_mat[ctr, :] = 0
        ctr = ctr + 1

pred = np.reshape(result_mat.T, (result_mat.size, 1)).flatten()
submission_VAR = pd.DataFrame({'id': test['id'], 'sales': pred})
submission_VAR.to_csv('predicted_data/submission_VAR.csv', index=False)