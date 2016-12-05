import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso

#fin_data=pd.read_csv('cleaned.csv')
price = pd.read_csv("data/cleaned_Price.csv")
pe = pd.read_csv("data/cleaned_PE.csv")
ps = pd.read_csv("data/cleaned_PS.csv")
cap = pd.read_csv("data/cleaned_Cap.csv")
pb = pd.read_csv("data/cleaned_PB.csv")
dividend = pd.read_csv("data/cleaned_Dividend.csv")
price_change = pd.read_csv("data/cleaned_PriceChange.csv") #what we want to predict

data_list = [pe, ps, cap, pb, dividend]

nrows = price_change.shape[0] # number of days
ncols = price_change.shape[1] # number of stocks

#separate data into 80% train, 20% test to start
train_limit = int(.8*nrows)

train_list=[]
test_list=[]
for d in data_list:
	
	train_list.append(np.ravel(d.values[0:train_limit, 1:]))
	test_list.append(np.ravel(d.values[train_limit:, 1:]))


all_training_data = reduce((lambda x, y: np.column_stack((x,y))), train_list )
all_testing_Data =  reduce((lambda x, y: np.column_stack((x,y))), test_list )

labels = np.ravel(price_change)
train_labels = labels[0:train_limit]
test_labels = labels[train_limit : ]

#Now can move on to feeding data to model

#first step is to deal with missing or NAN values

clf = Lasso()
clf.fit(all_training_data, train_labels)

print(clf.coef_)