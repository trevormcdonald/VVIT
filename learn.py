from __future__ import print_function
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

#fin_data=pd.read_csv('cleaned.csv')
price = pd.read_csv("data/cleaned_Price.csv")
pe = pd.read_csv("data/cleaned_PE.csv")
ps = pd.read_csv("data/cleaned_PS.csv")
cap = pd.read_csv("data/cleaned_Cap.csv")
pb = pd.read_csv("data/cleaned_PB.csv")
dividend = pd.read_csv("data/cleaned_Dividend.csv")
price_change = pd.read_csv("data/cleaned_PriceChange.csv") #what we want to predict

data_list = [pe, ps, cap, pb, dividend]
names = ["PE", "PS", "Cap", "PB", "Dividend"]

#fill in missing values with the last previous value, then the next value
#could also try interpolation
for i in range(len(data_list)):
	d=data_list[i]
	d=d.fillna(method='pad')
	d=d.fillna(method='bfill')
	d=d.fillna(0)
	data_list[i] = d

#fill in missing price changes with zero
price_change=price_change.fillna(0)

for i, d in enumerate(data_list):
	d.to_csv('data/processed_' + names[i] + '.csv')

price_change.to_csv('data/processed_price_change.csv')

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
all_testing_data =  reduce((lambda x, y: np.column_stack((x,y))), test_list )


all_labels = np.ravel(price_change.values[:, 1:])
train_labels = np.ravel(price_change.values[0:train_limit, 1:])
test_labels = np.ravel(price_change.values[train_limit, 1:])


print(train_labels.shape)
print(all_training_data.shape)


#Now can move on to feeding data to model
alphas = np.logspace(-6, -1, 30)

def make_prediction(model, test_data, test_labels):
	preds=model.predict(test_data)
	error=mean_squared_error(test_labels, preds)
	print("Error for model : %d", error)
	return error

def lasso_with_cv():
	

	lasso_cv = LassoCV(alphas=alphas)

	k_fold = KFold(5)

	for k, (train, test) in enumerate(k_fold.split(all_training_data, train_labels)):
	    lasso_cv.fit(all_training_data[train], train_labels[train])
	    print("[fold {0}] alpha: {1:.5f}, score: {2:.5f}".
	          format(k, lasso_cv.alpha_, lasso_cv.score(all_training_data[test], train_labels[test])))

	print()

	#clf.fit(all_training_data, train_labels)

	#print(clf.coef_)
	#print(clf.intercept_)

def make_lasso_cv():

	lasso = Lasso(random_state=0)

	X=all_training_data
	y=train_labels
	scores = list()
	scores_std = list()

	n_folds = 3

	for alpha in alphas:
		lasso.alpha = alpha
		this_scores = cross_val_score(lasso, X, y, cv=n_folds, n_jobs=1)
		scores.append(np.mean(this_scores))
		scores_std.append(np.std(this_scores))

	scores, scores_std = np.array(scores), np.array(scores_std)

	plt.figure().set_size_inches(8, 6)
	plt.semilogx(alphas, scores)

	# plot error lines showing +/- std. errors of the scores
	std_error = scores_std / np.sqrt(n_folds)

	plt.semilogx(alphas, scores + std_error, 'b--')
	plt.semilogx(alphas, scores - std_error, 'b--')

	# alpha=0.2 controls the translucency of the fill color
	plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

	plt.ylabel('CV score +/- std error')
	plt.xlabel('alpha')
	plt.axhline(np.max(scores), linestyle='--', color='.5')
	plt.xlim([alphas[0], alphas[-1]])

	plt.show()



	
if __name__ == '__main__':
	make_lasso_cv()