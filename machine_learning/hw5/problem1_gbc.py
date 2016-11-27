import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import xgboost as xgb
from sklearn.grid_search import GridSearchCV

from sklearn.preprocessing import StandardScaler

INPUT_SIZE = 2000

max_depth = [2, 3, 4]

cv_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5], 'n_estimators': [50, 100, 150]}

#n_jobs=-1

def main():
	# read in csv text, modify training data and labels
	training = pd.read_csv('reviews_tr.csv')
	training_text = training['text'][0:INPUT_SIZE]
	training_labels = training['label'][0:INPUT_SIZE].values.reshape((INPUT_SIZE, 1))

	# read in csv, modify test 
	test = pd.read_csv('reviews_te.csv')
	test_text = test['text'][0:INPUT_SIZE]
	test_labels = test['label'][0:INPUT_SIZE].values.reshape((INPUT_SIZE, 1))

	unigram_train, unigram_test = build_unigram(training_text, test_text)
        print unigram_train.shape
        print unigram_test.shape
        training = xgb.DMatrix(unigram_train, training_labels)

        param = {'max_depth':2, 'eta':0.5, 'silent':1, 'objective':'binary:logistic' }
        etas = [0.25, 0.5, 0.75]
        depths = [2, 4, 6]
        best_eta, best_depth = select_best_params(training, training_labels, etas, depths)
        print "best eta, depth = ", best_eta, best_depth

        best_params = {'max_depth':best_depth, 'eta':best_eta, 'silent':1, 'objective':'binary:logistic' }

        best_clf = xgb.train(best_params, training, 100)

        training_preds = np.floor(best_clf.predict(training) + 0.5)
        print "Final training error: ", get_error_rate(training_labels, training_preds)

        tester = xgb.DMatrix(unigram_test, test_labels)
        raw_preds = best_clf.predict(tester)
        binary_preds = np.floor(raw_preds + 0.5)
        print "Final test error: ", get_error_rate(test_labels, binary_preds)

def select_best_params(X, Y, etas, depths):
	lowest_err = 1.0
	best_params = (None, None)
	for e in etas:
		for d in depths:
			avg_err = K_fold_cross_validation(3, X, Y, e, d)
			print "error on depth=%s and eta=%s" % (str(d), str(e), str(avg_err))
			if avg_err < lowest_err:
				lowest_err = avg_err
				best_params = (e, d)

	return best_params

def K_fold_cross_validation(k, X, Y, e, d):
	k_fold = KFold(n_splits = k)
	n = 0
	errorSum = 0
	param = {'max_depth':d, 'eta':e, 'silent':1, 'objective':'binary:logistic' }
	for train, test in k_fold.split(X):
		n += 1
		clf = xgb.train(param, X[train], 100)
		preds = np.floor(clf.predict(X[test]) + 0.5)
		error = get_error_rate(preds, Y[test])
		errorSum += error
	
	avg_err = float(errorSum) / float(n)
	
	return avg_err

# takes in raw data
def build_unigram(training_data, test_data):
	vectorizer = CountVectorizer()
	unigram_tr = vectorizer.fit_transform(training_data)
	unigram_te = vectorizer.transform(test_data)
	print "built unigram feature representation"
	return unigram_tr, unigram_te

def get_error_rate(preds, labels):
        errorCount = 0
        totalCount = 0
        for i in range(len(preds)):
            if (int(preds[i]) != int(labels[i])):
                errorCount += 1
            totalCount += 1
        return float(errorCount) / float(totalCount)

# def select_hyperparams(X, Y, depths, n_estimators):
# 	best_score_so_far = 0.0
# 	best_hyperparams = (None, None)

# 	for d in depths:
# 			# build the DecisionTree base estimator with these parameters
# 		print("----\nTree with max_depth=%s\n----" % (str(d)))
# 		# for each estimator amount, build an AdaBoostClassifier
# 		for n in n_estimators:
# 			clf = GradientBoostingClassifier(n_estimators=n, max_depth=d)
# 			# default is 3-fold CV
# 			scores = cross_val_score(clf, X, Y, cv=3, n_jobs=-1)
# 			avg_score = scores.mean()
# 			print("Evaluating classifier w/ Tree depth=%s and min_samples=%s and AVG SCORE=%s" % (str(d), str(n), str(avg_score)))
# 			if avg_score > best_score_so_far:
# 				best_hyperparams = (d, n)
# 				best_score_so_far = avg_score

# 	print "BEST SCORE:", best_score_so_far
# 	return best_hyperparams


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
