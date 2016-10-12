import random
import numpy as np
import sklearn as sk
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
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

	min_error = float('inf')
	best_classifier = 0

	print "1. RUNNING NAIVE BAYES ON UNIGRAM:"
	bayes_unigram_train, bayes_unigram_test = build_unigram(training_text, test_text)
	# modify data for naive bayes (only count each word once)
	nonzero_train_idx = np.nonzero(bayes_unigram_train)
	nonzero_test_idx = np.nonzero(bayes_unigram_test)
	bayes_unigram_train[nonzero_train_idx] = 1
	bayes_unigram_test[nonzero_test_idx] = 1
	del nonzero_test_idx, nonzero_train_idx
	average_bayes_error = five_fold_cross_validation(bayes_unigram_train, training_labels, "bayes")
	print "Avg error for NAIVE BAYES with 5-fold cross validation: [%s]" % str(average_bayes_error)
	if average_bayes_error < min_error:
		min_error = average_bayes_error
		best_classifier = 1

	print "2. RUNNING PERCEPTRON ON UNIGRAM:"
	unigram_train, unigram_test = build_unigram(training_text, test_text)
	avg_unigram_error = five_fold_cross_validation(unigram_train, training_labels, "perceptron")
	print "Avg error for UNIGRAM Perceptron with 5-fold cross validation: [%s]" % str(avg_unigram_error)
	if avg_unigram_error < min_error:
		min_error = avg_unigram_error
		best_classifier = 2

	print "3. RUNNING PERCEPTRON ON BIGRAM:"
	bigram_train, bigram_test = build_bigram(training_text, test_text)
	avg_bigram_error = five_fold_cross_validation(bigram_train, training_labels, "perceptron")
	print "Avg error for BIGRAM Perceptron with 5-fold cross validation: [%s]" % str(avg_bigram_error)
	if avg_bigram_error < min_error:
		min_error = avg_bigram_error
		best_classifier = 3

	print "4. RUN PERCEPTRON ON TF-IDF:"
	idf_train, idf_test = build_tf_idf(unigram_train, unigram_test)
	avg_idf_error = five_fold_cross_validation(idf_train, training_labels, "perceptron")
	print "Avg error for TF-IDF Perceptron with 5-fold cross validation: [%s]" % str(avg_idf_error)
	if avg_idf_error < min_error:
		min_error = avg_idf_error
		best_classifier = 4

	print "5. RUN PERCEPTRON ON SUBLINEAR:"
	sublinear_train, sublinear_test = build_tf_sublinear(unigram_train, unigram_test)
	avg_sublinear_error = five_fold_cross_validation(sublinear_train, training_labels, "perceptron")
	print "Avg error for SUBLINEAR Perceptron with 5-fold cross validation: [%s]" % str(avg_sublinear_error)
	if avg_sublinear_error < min_error:
		min_error = avg_sublinear_error
		best_classifier = 5

	# now train best classifier on training data and evaluate on test data.

	if best_classifier == 1:
		print "Best: Naive Bayes on Unigram"
		priors, conditionals = train_naive_bayes(bayes_unigram_train, training_labels, classNum=2)
		training_preds = test_input(bayes_unigram_train, priors, conditionals)
		training_error = get_error_rate(training_preds, training_labels)
		test_preds = get_input(bayes_unigram_test, priors, conditionals)
		test_error = get_error_rate(test_preds, test_labels)
	elif best_classifier == 2:
		print "Best: Perceptron on Unigram"
		training_err, test_err = evaluate_perceptron(unigram_train, training_labels, unigram_test, test_labels)
	elif best_classifier == 3:
		print "Best: Perceptron on Bigram"
		training_err, test_err = evaluate_perceptron(bigram_train, training_labels, bigram_test, test_labels)
	elif best_classifier == 4:
		print "Best: Perceptron on TF-IDF"
		training_err, test_err = evaluate_perceptron(idf_train, training_labels, idf_test, test_labels)
	elif best_classifier == 5:
		print "Best: Perceptron on Sublinear"
		training_err, test_err = evaluate_perceptron(sublinear_train, training_labels, sublinear_test, test_labels)
	print "with training error: [%s] and test error [%s]" % (training_err, test_err)

def evaluate_perceptron(tr_data, tr_labels, te_data, te_labels):
	weights, bias = train_average_perceptron(tr_data, tr_labels)
	training_error = test_average_perceptron(tr_data, tr_labels, weights, bias)
	w2, b2 = train_average_perceptron(te_data, te_labels)
	test_error = test_average_perceptron(te_data, te_labels, w2, b2)
	return training_error, test_error

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

# ------------- DATA REPRESENTATIONS -------------- #

# takes in raw data
def build_unigram(training_data, test_data):
	vectorizer = CountVectorizer()
	unigram_tr = vectorizer.fit_transform(training_data)
	unigram_te = vectorizer.transform(test_data)
	return unigram_tr, unigram_te

# takes in raw data
def build_bigram(training_data, test_data):
	bigram_vectorizer = CountVectorizer(ngram_range=(1,2))
	bigram_tr = bigram_vectorizer.fit_transform(training_data)
	bigram_te = bigram_vectorizer.transform(test_data)
	return bigram_tr, bigram_te

# takes in sparce matrices
def build_tf_idf(training_data, test_data):
	transformer = TfidfTransformer(use_idf=True, smooth_idf=False)
	idf_tr = transformer.fit_transform(training_data)
	idf_te = transformer.transform(test_data)
	return idf_tr, idf_te

# takes in sparce matrices
def build_tf_sublinear(training_data, test_data):
	transformer = TfidfTransformer(use_idf=True, smooth_idf=False, sublinear_tf=True)
	sublinear_tr = transformer.fit_transform(training_data)
	sublinear_te = transformer.transform(test_data)
	return sublinear_tr, sublinear_te

# ------------- LEARNING METHODS ------------------- #
# // Perceptron //
def train_average_perceptron(X, Y):
	# Y.shape = (1000000, 1)
	# shuffle the input data randomly
	rand = np.random.permutation(np.shape(X)[0]-1)
	X = X[rand]
	Y = Y[rand]
	w = np.zeros((1, np.shape(X)[1]))
	u = w 	#cached
	b = 0 	#bias
	B = 0 	#cached bias
	c = 1 	#counter

	for i in range(np.shape(X)[0]):
		a = X[i] * w.T + b
		if (a * Y[i][0] <= 0):
			w = w + Y[i][0] * X[i]
			b = b + Y[i][0]
			u = u + Y[i] * c * X[i]
			B = B + Y[i] * c
		c = c + 1
	return w - (1 / c) * u, b - (1 / c) * B

def test_average_perceptron(X, Y, w, b):
	res = X * w.T + b
	signs = np.sign(res)
	errorCount = 0
	totalCount = 0
	for i in range(np.shape(signs)[0]):
		if int(signs[i]) != int(Y[i]):
			errorCount += 1
		totalCount += 1
	return float(errorCount) / float(totalCount) #as error rate

#  // Naive Bayes //
def get_priors(X, Y, classNum):
	sparse_priors = np.zeros((classNum, len(Y)))
	count = 0
	for i in range(len(Y)):
		if Y[i] != 0:
			count += 1

	for i in range(classNum):
		indices = (Y.T[0] == i + 1)
		sparse_priors[i][indices] = 1
	class_totals = np.sum(sparse_priors, axis=1)
	class_priors = class_totals.astype(np.float32) / float(count) # percentage of a given label in label set
	return class_priors

def get_class_conditionals(X, Y, classNum):
	sparse_conditionals = np.zeros((classNum, len(Y)))
	#sparse matrix identifying every label (column) as 'positive' or 'negative' via a 1 in a respective class (row)
	for i in range(classNum):
		indices = (Y.T[0] == i + 1)
		sparse_conditionals[i][indices] = 1

	numerator = ((sparse_conditionals * X) + 1).T
	denominator = np.array(2 + np.sum(sparse_conditionals, axis=1)).reshape(classNum, 1).T
	class_conditionals = np.divide(numerator.astype(np.float32), denominator.astype(np.float32))
	return class_conditionals

def test_input(X, priors, conditionals):
	# log(pi) + Sum_d(log(1 - u)) + Sum_d(log(u) - log(1 - u)) * xj of the form alpha_0 + x_j*a_j
	x_complement = np.ones(X.shape, dtype=np.float32) - X #(1 - Xj) matrix of 0's and 1's
	u_complement = np.ones(conditionals.shape, dtype=np.float32) - conditionals #(1 - u_yj)
	log_u = np.log(conditionals)
	log_u_complement = np.log(u_complement)
	normal_term = X * log_u 
	complement_term = x_complement * log_u_complement
	#end summation, add log priors
	final = (normal_term + complement_term) + np.log(priors)
	preds = np.argmax(final, axis=1) + 1 # add 1 since python is 0 indexed
	return preds

def train_naive_bayes(X, Y, classNum):
	class_priors = get_priors(X, Y, classNum)
	class_conditionals = get_class_conditionals(X, Y, classNum)
	return class_priors, class_conditionals

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
