import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

#load data
holidays_events = pd.read_csv('datasets/holidays_events.csv')
oil = pd.read_csv('datasets/oil.csv')
sample_submission = pd.read_csv('datasets/sample_submission.csv')
stores = pd.read_csv('datasets/stores.csv')
test = pd.read_csv('datasets/test.csv')
train = pd.read_csv('datasets/train.csv')
transactions = pd.read_csv('datasets/transactions.csv')

#plot timeseries for oil price
oil.plot()
plt.show()
#print head of training and test data
print('Trainingdata:')
print(train.head(50))
print('_________________________________________________')
print('Testdata:')
print(test.head(50))


#plot training data for 1 store and 1 product
train.loc[(train['store_nbr']==1) & (train['family']=='HARDWARE')].plot('date','sales',kind='scatter',figsize=(10,10),rot=90,title='Productfamily HARDWARE at Store_1')
plt.show()