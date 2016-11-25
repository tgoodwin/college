import random
import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import xgboost as xgb
from sklearn.grid_search import GridSearchCV

from sklearn.preprocessing import StandardScaler

INPUT_SIZE = 2000

#n_estimators
#max_depth
#min_samples_leaf


max_depth = [2, 3, 4]
n_estimators = [50, 100, 150, 200]

cv_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5], 'n_estimators': [50, 100, 150]}

optimized_GBM = GridSearchCV(xgb.XGBClassifier(), 
                            cv_params, 
                             scoring = 'accuracy', cv = 5, n_jobs = -1, verbose=1)

#n_jobs=-1

def main():
	# read in csv text, modify training data and labels
	training = pd.read_csv('reviews_tr.csv')
	dtrain = xgb.DMatrix(training)
	training_text = training['text'][0:INPUT_SIZE]
	training_labels = training['label'][0:INPUT_SIZE].values.reshape((INPUT_SIZE, 1))
	zeros = np.where(training_labels == 0)
	training_labels[zeros] = -1
	del training, zeros

	# read in csv, modify test 
	test = pd.read_csv('reviews_te.csv')
	test_text = test['text'][0:INPUT_SIZE]
	test_labels = test['label'][0:INPUT_SIZE].values.reshape((INPUT_SIZE, 1))
	zeros = np.where(test_labels == 0)
	test_labels[zeros] = -1
	del test, zeros

	unigram_train, unigram_test = build_unigram(training_text, test_text)

	clf = GridSearchCV(xgb.XGBClassifier(), cv_params, scoring = 'accuracy', cv = 5, n_jobs = -1, verbose=1)
	clf.fit(unigram_train, unigram_test)
	print clf.best_score_, clf.best_params_


# takes in raw data
def build_unigram(training_data, test_data):
	vectorizer = CountVectorizer()
	unigram_tr = vectorizer.fit_transform(training_data)
	unigram_te = vectorizer.transform(test_data)
	print "built unigram feature representation"
	return unigram_tr, unigram_te

def select_hyperparams(X, Y, depths, n_estimators):
	best_score_so_far = 0.0
	best_hyperparams = (None, None)

	for d in depths:
			# build the DecisionTree base estimator with these parameters
		print("----\nTree with max_depth=%s\n----" % (str(d)))
		# for each estimator amount, build an AdaBoostClassifier
		for n in n_estimators:
			clf = GradientBoostingClassifier(n_estimators=n, max_depth=d)
			# default is 3-fold CV
			scores = cross_val_score(clf, X, Y, cv=3, n_jobs=-1)
			avg_score = scores.mean()
			print("Evaluating classifier w/ Tree depth=%s and min_samples=%s and AVG SCORE=%s" % (str(d), str(n), str(avg_score)))
			if avg_score > best_score_so_far:
				best_hyperparams = (d, n)
				best_score_so_far = avg_score

	print "BEST SCORE:", best_score_so_far
	return best_hyperparams


# def Kfold_cross_validation(k, X, Y, classifier):
# 	k_folds = KFold(n_splits = k)
# 	n = 0
# 	errorSum = 0.0
# 	for train, test in k_folds.split(X):
# 		n += 1
# 		classifier.fit(X[train], Y[train])
# 		err = classifier.score(X[test], Y[test])
# 		print("Training set score on trial %d: %f" % (n, err))
# 		errorSum += err

# 	avg_error = float(errorSum) / float(k)
# 	return avg_error


if __name__ == "__main__":
	main()