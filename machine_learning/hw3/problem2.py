import random
import numpy as np
import sklearn as sk
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def main():
	# read in csv text
	df = pd.read_csv('reviews_tr.csv')
	review_docs = df['text'][0:2000] #(1000000,)
	review_labels = df['label'][0:2000].values.reshape((2000, 1))
	zeros = np.where(review_labels == 0)
	review_labels[zeros] = -1

	test = pd.read_csv('reviews_te.csv')
	test_text = test['text'][0:2000]
	test_labels = test['label'][0:2000].values.reshape((2000, 1))
	zeros = np.where(test_labels == 0)
	test_labels[zeros] = -1

	# build unigram representation
	count_matrix, test_matrix = unigram(review_docs, test_text)
	print count_matrix.shape
	w, b = train_average_perceptron(count_matrix, review_labels)
	error_rate = test_average_perceptron(test_matrix, test_labels, w, b)
	print "error rate:", str(error_rate)

	#print "count_matrix", count_matrix.shape
	# use count matrix to build IDF representation
	#idf_matrix = build_tf_idf(count_matrix)
	#print "idf", idf_matrix.shape
	#bigram_matrix = bigram(count_matrix)
	#print "bigram", bigram_matrix.shape
	

def unigram(training_data, test_data):
	print "building unigram"
	vectorizer = CountVectorizer()
	X_tr = vectorizer.fit_transform(training_data)
	X_te = vectorizer.transform(test_data)
	return X_tr, X_te

def bigram(training_data, test_data):
	bigram_vectorizer = CountVectorizer(ngram_range=(1,2))
	X_2 = bigram_vectorizer.fit_transform(training_data)
	X2_te = bigram_vectorizer.transform(test_data)
	return X_2, X2_te

def build_tf_sublinear(training_data, test_data):
	print "building unigram"
	transformer = TfidfTransformer(use_idf=True, smooth_idf=False, sublinear_tf = True)
	X_tr = transformer.fit_transform(training_data)
	X_te = transformer.transform(test_data)
	return X_tr, X_te

def build_tf_idf(training_data, test_data):
	print "building idf"
	transformer = TfidfTransformer(use_idf=True, smooth_idf=False)
	idf_training = transformer.fit_transform(training_data)
	idf_test = transformer.transform(test_data)
	return idf_training, idf_test

def train_average_perceptron(X, Y):
	# Y.shape = (1000000, 1)
	# shuffle the input data randomly
	rand = np.random.permutation(np.shape(X)[0]-1)
	X = X[rand]
	Y = Y[rand]
	w = np.zeros((1, np.shape(X)[1]))
	u = w #cached
	b = 0 #bias
	B = 0 #cached bias
	c = 1 #counter
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
	mult = np.multiply(Y, signs)
	match = len(np.where(mult > 0))
	return float(match) / float(len(Y)) #as error rate

if __name__ == "__main__":
	main()
