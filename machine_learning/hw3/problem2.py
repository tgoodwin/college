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
	review_docs = df['text'] #(1000000,)
	review_labels = df['label'][0:].values.reshape((1000000, 1))
	# build unigram representation
	count_matrix = unigram(review_docs)
	print count_matrix.shape
	print len(count_matrix)
	#print "count_matrix", count_matrix.shape
	# use count matrix to build IDF representation
	#idf_matrix = build_tf_idf(count_matrix)
	#print "idf", idf_matrix.shape
	bigram_matrix = bigram(count_matrix)
	print "bigram", bigram_matrix.shape
	

def unigram(training_data):
	print "building unigram"
	vectorizer = CountVectorizer()
	X = vectorizer.fit_transform(training_data)
	return X

def bigram(training_data):
	bigram_vectorizer = CountVectorizer(ngram_range=(1,2))
	X_2 = bigram_vectorizer.fit_transform(training_data)
	return X_2

def build_tf_sublinear(training_data):
	print "building unigram"
	transformer = TfidfTransformer(use_idf=True, smooth_idf=False, sublinear_tf = True)
	tf_set = transformer.fit_transform(training_data)
	print tf_set
	print "shape", tf_set.shape
	return tf_set

def build_tf_idf(training_data):
	print "building idf"
	transformer = TfidfTransformer(use_idf=True, smooth_idf=False)
	tf_idf_set = transformer.fit_transform(training_data)
	return tf_idf_set

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

	for i in range(np.shape(X)[0]):
		if (Y[i][0] * (w.T * X[i] + b) <= 0):
			w = w + Y[i][0] * X[i]
			b = b + Y[i][0]
			u = u + Y[i] * c * X[i]
			B = B + Y[i] * c
		# end if
		c = c + 1
	# end for
	return w - (1 / c)*u , b - (1/c) * B





if __name__ == "__main__":
	main()
