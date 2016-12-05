from __future__ import print_function
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV, Ridge, LinearRegression, HuberRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import LinearSVC

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
test_labels = np.ravel(price_change.values[train_limit:, 1:])

bin_train_labels= (np.ravel(price_change.values[0:train_limit, 1:]) > 0).astype(int)
bin_test_labels= (np.ravel(price_change.values[train_limit:, 1:]) > 0).astype(int)


print(train_labels.shape)
print(all_training_data.shape)


#Now can move on to feeding data to model
alphas = np.logspace(-6, -1, 30)

def make_prediction(model, test_data, test_labels):
	preds=model.predict(test_data)
	error=mean_squared_error(test_labels, preds)
	print("R2 score:", r2_score(test_labels, preds))
	print("Mean square error for model:", error)
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

	#X=StandardScaler().fit_transform(all_training_data)
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
		clf = Lasso(random_state=0)
		clf.alpha = alpha
		clf.fit(X,y)
		print(clf.coef_)
		make_prediction(clf, all_training_data, train_labels)

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
	print(scores.argmax(axis=0))
	plt.xlim([alphas[0], alphas[-1]])

	#TRAIN STATISTICS
	#[  5.78095079e-08   0.00000000e+00  -6.66919151e-10   0.00000000e+00
  	#-1.46713844e-04]
	#R2 score: 0.000554848134686
	#Mean square error for model: 0.000379071509183

	#TEST STATISTICS
	#[  8.66518676e-08   0.00000000e+00  -5.82401268e-10   0.00000000e+00
  	#-1.04665831e-04]
			#R2 score: -0.00600409234752
		#Mean square error for model: 0.000360113552396

	plt.show()


def make_ridge():
	#X=StandardScaler().fit_transform(all_training_data)
	X=all_training_data
	y=train_labels
	n_alphas = 200
	alphas = np.logspace(-6, 6, n_alphas)
	ridge = Ridge(fit_intercept=False)
	"""
	coefs = []
	for a in alphas:
		clf.set_params(alpha=a)
		clf.fit(X, y)
		coefs.append(clf.coef_)
		print(clf.coef_)
		make_prediction(clf, all_testing_data, test_labels)"""

	scores = list()
	scores_std = list()

	n_folds = 3

	for i, alpha in enumerate(alphas):
		print(i)
		ridge.alpha = alpha
		this_scores = cross_val_score(ridge, X, y, cv=n_folds, n_jobs=1)
		scores.append(np.mean(this_scores))
		scores_std.append(np.std(this_scores))
		clf = Ridge(fit_intercept=False)
		clf.alpha = alpha
		clf.fit(X,y)
		print(clf.coef_)
		make_prediction(clf, all_testing_data, test_labels)

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
	print(scores.argmax(axis=0))
	plt.xlim([alphas[0], alphas[-1]])

	plt.show()


	#TRAIN STATISTICS
	#[  6.64606137e-08   1.08176398e-04   3.38084653e-10   3.66845035e-06
  	#-7.73313163e-05]
	#R2 score: 1.49640024261e-06
	#Mean square error for model: 0.000379281385509

	#TEST STATISTICS
	#[  7.02185692e-08   1.07375357e-04   3.40018010e-10   3.67298574e-06
  	#-7.64411545e-05]
	#R2 score: -0.0081151195638
	#Mean square error for model: 0.000360869224779

def make_linreg():
	#X=StandardScaler().fit_transform(all_training_data)
	X=all_training_data
	y=train_labels

	clf=LinearRegression()
	clf.fit(X, y)
	print(clf.coef_)
	make_prediction(clf, all_training_data, train_labels)

	#TRAIN STATISTICS
	#[ -1.08077479e-07   2.65085446e-05  -8.72124124e-10   6.50959233e-07
  	#-2.33324985e-04]
	#R2 score: 0.000658485804145
	#Mean square error for model: 0.000379032201285


	#TEST STATISTICS
	#[ -1.08077479e-07   2.65085446e-05  -8.72124124e-10   6.50959233e-07
	  #-2.33324985e-04]
	#R2 score: -0.00666567529883
	#Mean square error for model: 0.000360350375475


def make_huber():
	X=all_training_data
	y=train_labels

	huber = HuberRegressor(fit_intercept=True)
	huber.fit(X,y)
	print("Huber coefficients: ",huber.coef_)
	print("Huber intercept: ", huber.intercept_)

	print("Training Statistics:")
	make_prediction(huber, all_training_data, train_labels)
	print("Testing Statistics:")
	make_prediction(huber, all_testing_data, test_labels)

	#Huber coefficients:  [  7.60688999e-12   5.93235380e-13   9.93338859e-10   7.98625362e-13
	#  -3.18515366e-14]
	#Huber intercept:  1.80872771017e-13
	#Training Statistics:
	#R2 score: -0.000398163043167
	#Mean square error for model: 0.000379432969123
	#Testing Statistics:
	#R2 score: -0.00995293878596
	#Mean square error for model: 0.000361527098453

def make_svm():
	X= all_training_data
	y=bin_train_labels
	clf = LinearSVC()
	clf.fit(X,y)
	print("SVM score: ", clf.score(X, y))

	print("Training Statistics:")
	make_prediction(clf, all_training_data, bin_train_labels)

	print("Testing Statistics:")
	make_prediction(clf, all_testing_data, bin_test_labels)




	
if __name__ == '__main__':
	make_svm()