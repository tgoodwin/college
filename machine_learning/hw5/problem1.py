import random
import numpy as np
import sklearn as sk
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, cross_val_score
INPUT_SIZE = 200000

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
	print "done"

# takes in raw data
def build_unigram(training_data, test_data):
	vectorizer = CountVectorizer()
	unigram_tr = vectorizer.fit_transform(training_data)
	unigram_te = vectorizer.transform(test_data)
	return unigram_tr, unigram_te

def five_fold_cross_validation(X, Y, estimator):
	five_fold = KFold(n_splits = 5)
	n = 0
	errorSum = 0
	for train, test in five_fold.split(X):
		n += 1
		if estimator == "perceptron":
			weights, bias = train_average_perceptron(X[train], Y[train])
			error = test_average_perceptron(X[test], Y[test], weights, bias)
			errorSum += error
			print "%s error on trial #%d: %s" % (estimator, n, str(error))
		if estimator == "bayes":
			priors, conditionals = train_naive_bayes(X[train], Y[train], classNum=2)
			test_preds = test_input(X[test], priors, conditionals)
			error = get_error_rate(test_preds, Y[test])
			errorSum += error
			print "%s error on trial #%d: %s" % (estimator, n, str(error))
	avg_error = float(errorSum) / float(5)
	return avg_error

def get_error_rate(preds, labels):
	errorCount = 0
	totalCount = 0
	for i in range(len(preds)):
		if (int(preds[i]) != int(labels[i])):
			errorCount += 1
		totalCount += 1
	return float(errorCount) / float(totalCount)

if __name__ == "__main__":
	main()