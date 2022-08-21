# implementation of VAR model for one store and one product family

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

#train a VAR model for one store and product family
store_nbr = 1
family = 'AUTOMOTIVE'
num_pred_days = 15
start_date_training = '2016-08-15'
train.index = train['date']
col_names = ['store_nbr','family','sales','oilprice','weekday'] #takes those columns to make predictions
data = train[start_date_training:]
data = data.loc[(data['store_nbr']==store_nbr)&(data['family']==family)][col_names]
data = data.drop(columns=['store_nbr','family'],axis=1)
train_df, test_df = data[:-15], data[-15:]

model = VAR(train_df)

#check which order to select
sorted_order=model.select_order(maxlags=10)
print(sorted_order.summary())

#the printed data shows that the optimal order to select in the VAR model is 7

#fit model and make predictions
var_model = VARMAX(train_df, order=(7,0),enforce_stationarity=True)
fitted_model = var_model.fit(disp=False)
predict = fitted_model.get_prediction(start=len(train_df),end=len(train_df)+num_pred_days-1)
predictions = predict.predicted_mean
predictions = np.round(predictions['sales'],0)
rmse = np.sqrt(mean_squared_error(test_df['sales'].values,predictions))
wd = wasserstein_distance(test_df['sales'].values,predictions)
print('RMSE-value: %s'%rmse)
print('Wasserstein-dist: %s'%wd)

#plot the results
df_plot = pd.DataFrame({'sales_pred':predictions,'sales_truth':test_df['sales'].values})
df_plot.plot(style=['x--','o--'],title='VAR prediction',ylabel='sales',figsize=(8,6))
plt.show()
