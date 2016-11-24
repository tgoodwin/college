import random
import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold, cross_val_score
INPUT_SIZE = 2000

#n_estimators
#max_depth
#min_samples_leaf


max_depth = [2]
min_samples_leaf = [10, 8, 4, 2, 1]
n_estimators = [50, 100, 150, 200]

#n_jobs=-1

def main():
	# read in csv text, modify training data and labels
	training = pd.read_csv('reviews_tr.csv')
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

	scaler = StandardScaler(with_mean=False)
	unigram_train, unigram_test = build_unigram(training_text, test_text)
	scaler.fit(unigram_train)
	unigram_train = scaler.transform(unigram_train)
	unigram_test = scaler.transform(unigram_test)

	best_d, best_m, best_n = select_hyperparams(unigram_train, training_labels.ravel(), max_depth, min_samples_leaf, n_estimators)
	print("\nselected hyperparamters: max_depth=%s min_samples=%s n_estimators=%s" % (str(best_d), str(best_m), str(best_n)))

	final_tree_estimator = DecisionTreeClassifier(max_depth=best_d, min_samples_leaf=best_m)
	final_ada = AdaBoostClassifier(base_estimator=final_tree_estimator, n_estimators=best_n)

	final_ada.fit(unigram_train, training_labels.ravel())

	print("Final training set score: %f" % final_ada.score(unigram_train, training_labels.ravel()))
	print("Final test set score: %f" % final_ada.score(unigram_test, test_labels.ravel()))


# takes in raw data
def build_unigram(training_data, test_data):
	vectorizer = CountVectorizer()
	unigram_tr = vectorizer.fit_transform(training_data)
	unigram_te = vectorizer.transform(test_data)
	print "built unigram feature representation"
	return unigram_tr, unigram_te

def select_hyperparams(X, Y, depths, min_samples_leaf, n_estimators):
	best_score_so_far = 0.0
	best_hyperparams = (None, None, None)

	for d in depths:
		for m in min_samples_leaf:
			# build the DecisionTree base estimator with these parameters
			estimator = DecisionTreeClassifier(max_depth=d, min_samples_leaf=m)
			print("----\nTree with max_depth=%s and min_samples_leaf=%s\n----" % (str(d), str(m)))
			# for each estimator amount, build an AdaBoostClassifier
			for n in n_estimators:
				clf = AdaBoostClassifier(base_estimator=estimator, n_estimators=n)
				# default is 3-fold CV
				scores = cross_val_score(clf, X, Y, cv=5, n_jobs=-1)
				avg_score = scores.mean()
				print("Evaluating classifier w/ Tree depth=%s and min_samples=%s and n_estimators=%s: AVG SCORE=%s" % (str(d), str(m), str(n), str(avg_score)))
				if avg_score > best_score_so_far:
					best_hyperparams = (d, m, n)
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