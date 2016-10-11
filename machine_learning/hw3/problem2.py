import random
import numpy as np
import sklearn as sk
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import KFold, cross_val_score

INPUT_SIZE = 200

def main():
	# read in csv text, modify training data and labels
	df = pd.read_csv('reviews_tr.csv')
	review_docs = df['text'][0:INPUT_SIZE]
	review_labels = df['label'][0:INPUT_SIZE].values.reshape((INPUT_SIZE, 1))
	zeros = np.where(review_labels == 0)
	review_labels[zeros] = -1

	# read in csv, modify test 
	test = pd.read_csv('reviews_te.csv')
	test_text = test['text'][0:INPUT_SIZE]
	test_labels = test['label'][0:INPUT_SIZE].values.reshape((INPUT_SIZE, 1))
	zeros = np.where(test_labels == 0)
	test_labels[zeros] = -1

	train_matrix, test_matrix = unigram(review_docs, test_text)
	print train_matrix.shape, test_matrix.shape
	# MODIFY DATA FOR BAYES
	nonzero_train = np.nonzero(train_matrix)
	nonzero_test = np.nonzero(test_matrix)
	train_matrix[nonzero_train] = 1
	test_matrix[nonzero_test] = 1

	average_bayes_error = five_fold_cross_validation(DATA, LABELS, "bayes")

	# RUN PERCEPTRON ON REPRESENTATIONS
	# build unigram representation
	train_matrix, test_matrix = unigram(review_docs, test_text)
	print train_matrix.shape
	w, b = train_average_perceptron(train_matrix, review_labels)
	error_rate = test_average_perceptron(test_matrix, test_labels, w, b)
	print "error rate:", str(error_rate)

	# use count matrix to build IDF representation
	#idf_matrix = build_tf_idf(train_matrix)
	#print "idf", idf_matrix.shape
	#bigram_matrix = build_bigram(train_matrix)
	#print "build_bigram", bigram_matrix.shape
	
def five_fold_cross_validation(X, Y, estimator):
	five_fold = KFold(n_splits = 5)
	n = 0
	errorSum = 0
	for train, test in five_fold.split(X):
		n += 1
		print "Train: %s | test: %s" % (train, test)
		if estimator == "perceptron":
			weights, bias = train_average_perceptron(X[train], Y[train])
			error = test_average_perceptron(X[test], Y[test], weights, bias)
			errorSum += error
			print "%s error, trial %d: %s" % (estimator, n, str(error))
		if estimator == "bayes":
			class_priors = get_priors(X[train], Y[train], classNum=2)
			class_conditionals = get_class_conditionals(X[train], Y[train], classNum=2)
			test_preds = test_input(X[test], class_priors, class_conditionals)
			error = get_error_rate(test_preds, Y[test])
			errorSum += error
			print "%s error, trial %d: %s" % (estimator, n, str(error))

	avg_error = float(errorSum) / float(5)
	print "avg error for 5-fold cross validation on %s: %s" % (estimator, str(avg_error))
	return avg_error

# ------------- DATA REPRESENTATIONS -------------- #

def unigram(training_data, test_data):
	print "building unigram"
	vectorizer = CountVectorizer()
	unigram_tr = vectorizer.fit_transform(training_data)
	unigram_te = vectorizer.transform(test_data)
	return unigram_tr, unigram_te

def build_bigram(training_data, test_data):
	bigram_vectorizer = CountVectorizer(ngram_range=(1,2))
	bigram_tr = bigram_vectorizer.fit_transform(training_data)
	bigram_te = bigram_vectorizer.transform(test_data)
	return bigram_tr, bigram_te

def build_tf_sublinear(training_data, test_data):
	print "building unigram"
	transformer = TfidfTransformer(use_idf=True, smooth_idf=False, sublinear_tf=True)
	sublinear_tr = transformer.fit_transform(training_data)
	sublinear_te = transformer.transform(test_data)
	return sublinear_tr, sublinear_te

def build_tf_idf(training_data, test_data):
	print "building idf"
	transformer = TfidfTransformer(use_idf=True, smooth_idf=False)
	idf_tr = transformer.fit_transform(training_data)
	idf_te = transformer.transform(test_data)
	return idf_tr, idf_te

# ------------- LEARNING METHODS ------------------- #

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
	print "y", Y.shape
	print "w", w.shape

	for i in range(np.shape(X)[0]):
		a = X[i] * w.T + b
		print a * Y[i][0]
		print "----"
		if (a * Y[i][0] <= 0):
			w = w + Y[i][0] * X[i]
			b = b + Y[i][0]
			u = u + Y[i] * c * X[i]
			B = B + Y[i] * c
		# end if
		c = c + 1
	# end for
	return w - (1 / c) * u, b - (1 / c) * B

def test_average_perceptron(X, Y, w, b):
	print w.shape
	print X.shape
	res = X * w.T + b
	signs = np.sign(res)
	temp = np.multiply(Y, signs)
	num_correct = len(np.where(temp > 0))
	return float(num_correct) / float(len(Y)) #as error rate

#  // Naive Bayes //
def get_priors(X, Y, classNum):
	print "getting priors"
	sparse_priors = np.zeros((classNum, len(Y)))
	count = 0
	# brittle solution for skipping over labels we don't care about.
	# in this homework's data, '0' is never a label we'll care about, by my design.
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
	print "getting conditionals"
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
	print "testing input"
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

def get_error_rate(preds, labels):
	errorCount = 0
	totalCount = 0
	for i in range(len(preds)):
		if (int(preds[i]) != int(labels[i])):
			errorCount += 1
		totalCount += 1
	return str(float(errorCount) / float(totalCount))

if __name__ == "__main__":
	main()
