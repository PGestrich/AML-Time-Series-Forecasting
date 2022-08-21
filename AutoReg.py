# implementation of a autoregression model to predict the future sales for all stores and products

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

train = pd.read_csv('datasets/train.csv',index_col='date',parse_dates=True)
# include family_id into training set
train['family_id'] = np.tile(np.arange(33),int(3000888/33))
test = pd.read_csv('datasets/test.csv')

num_all_combinations = len(train['store_nbr'].unique()) * len(train['family'].unique())
num_pred_days = 16
result_mat = np.zeros((num_all_combinations, num_pred_days))
stores = list(train['store_nbr'].unique())
families = list(train['family_id'].unique())
ctr = 0
# run two for loops to train AutoReg model for each pair of possible compinations (store_nbr,family)
for store in stores:
    for family in families:
        # select store and family
        df = train.loc[(train['store_nbr'] == store) & (train['family_id'] == family)]
        train_values = df['sales'].values
        model = AutoReg(train_values, lags=50, trend='t')
        model_fit = model.fit()
        predictions = model_fit.predict(start=len(train_values), end=len(train_values) + num_pred_days - 1,
                                        dynamic=False)
        result_mat[ctr, :] = np.round(predictions, 0)
        ctr = ctr + 1

pred = np.reshape(result_mat.T, (result_mat.size, 1)).flatten()
submission_AutoReg = pd.DataFrame({'id': test['id'], 'sales': pred})

# store submission
submission_AutoReg.to_csv('predicted_data/submission_AutoReg.csv',index=False)