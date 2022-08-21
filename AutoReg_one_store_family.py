# implementation of Autoregression to predict the sales for one store and one product family
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

#load data
train = pd.read_csv('datasets/train.csv',index_col='date',parse_dates=True)

#select store and family and plot data
store_nbr = 3
family = 'LINGERIE'
df = train.loc[(train['store_nbr']==store_nbr) & (train['family']==family)]['sales']
plt.figure()
df.plot(ylabel='sales',title='Store_%s and family=%s'%(store_nbr,family))
plt.show()

#check if data is stationary
fuller_result = adfuller(df)
print('p_value:%s'%fuller_result[1])

#check for autocorrelation
plot_pacf(df, lags=50)
plt.show()

#train AutoReg model
X = df.values
train, test = X[0:len(X)-15], X[len(X)-15:]
model = AutoReg(train, lags = 50, trend='t')
model_fit = model.fit()

# make prediction on the test set
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1,dynamic=False)
predictions = np.round(predictions)
rmse = np.sqrt(mean_squared_error(test,predictions))
wd = wasserstein_distance(test,predictions)
print('RMSE-value: %s'%rmse)
print('Wasserstein-dist: %s'%wd)

#plot prediction
df_plot = pd.DataFrame({'sales_pred':predictions,'sales_truth':test},index = df.index[-15:])
df_plot.plot(style=['x--','o--'],title='AutoReg prediction',ylabel='sales',figsize=(8,6))
plt.show()

# train on whole dataset
test = pd.read_csv('datasets/test.csv',index_col='date',parse_dates=True)
test = test.loc[(test['store_nbr']==store_nbr) & (test['family']==family)]
train = df.values
model = AutoReg(train, lags = 50, trend='t')
model_fit = model.fit()

# predict from 2017-08-16 to 2017-08-31
predictions = model_fit.predict(start=len(train),end=len(train)+15,dynamic=False)
predictions = np.round(predictions,0)
df_pred = pd.DataFrame(predictions,index=test.index,columns=['sales'])

# plot predictions in future
df_pred.plot(title='predictions for future',ylabel='sales',style=['x--'])
plt.show()