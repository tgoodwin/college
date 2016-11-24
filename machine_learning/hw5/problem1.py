import random
import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
INPUT_SIZE = 200000

hyperparams = {
	'hidden_layer_sizes' : [(5, 2), (5, 5), (10,), (2, 5)],
	'alphas': [1e-5, 0.001, .01, 10.0, 100.0]
}

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

	scaler = StandardScaler(with_mean=False)
	unigram_train, unigram_test = build_unigram(training_text, test_text)
	scaler.fit(unigram_train)
	unigram_train = scaler.transform(unigram_train)
	unigram_test = scaler.transform(unigram_test)

	best_alpha, best_dims = select_hyperparams(unigram_train, training_labels.ravel(), hyperparams['alphas'], hyperparams['hidden_layer_sizes'])

	classifier = MLPClassifier(solver='lbgfs', alpha=best_alpha, hidden_layer_sizes=best_dims, random_state=1)
	classifier.fit(unigram_train, training_labels)

	print("Final training set score: %f" % classifier.score(unigram_train, training_labels.ravel()))
	print("Final test set score: %f" % classifier.score(unigram_test, test_labels.ravel()))


# takes in raw data
def build_unigram(training_data, test_data):
	vectorizer = CountVectorizer()
	unigram_tr = vectorizer.fit_transform(training_data)
	unigram_te = vectorizer.transform(test_data)
	print "built unigram feature representation"
	return unigram_tr, unigram_te

def select_hyperparams(X, Y, alphas, dims):
	best_tr_accuracy_so_far = 0.0
	best_hyperparams = (alphas[0], dims[0])
	for a in alphas:
		for dim in dims:
			nnet = construct_classifier_with_hyperparameters(a, dim)
			scores = cross_val_score(nnet, X, Y, cv=5, n_jobs=-1)
			k = 5
			avg_accuracy = scores.mean()
			print("Evaluating classifier with alpha=%f and dims=%s: SCORE=%s" % (a, str(dim), str(avg_accuracy)))
			if avg_accuracy > best_tr_accuracy_so_far:
				best_tr_accuracy_so_far = avg_accuracy
				best_hyperparams = (a, dim)

	print("selected hyperparamters: alpha=%f dims=%s. BEST SCORE: %s" % (best_hyperparams[0], str(best_hyperparams[1]), str(best_tr_accuracy_so_far)))
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

def construct_classifier_with_hyperparameters(alpha_val, nnet_dims):
	return MLPClassifier(solver='lbgfs', alpha=alpha_val, hidden_layer_sizes=nnet_dims, random_state=1)

if __name__ == "__main__":
	main()
