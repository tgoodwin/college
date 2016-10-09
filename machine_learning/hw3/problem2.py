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
	vectorizer = CountVectorizer()
	count_matrix = vectorizer.fit_transform(review_docs) # build sparse matrix (1,000,000 * 2000000)
	print count_matrix

	build_tf_unigram(count_matrix)


def build_tf_unigram(training_data):
	print "building unigram"
	transformer = TfidfTransformer(use_idf=False)
	tf_set = transformer.transform(training_data, copy=True)
	print tf_set
	print "shape", tf_set.shape
	return tf_set

def build_tf_idf(training_data):
	print "building idf"
	transformer = TfidfTransformer(use_idf=True, smooth_idf=False)
	tf_idf_set = TfidTransformer.transform(training_data, copy=True)
	return tf_idf_set

def build_bigram(training_data):
	return 1

if __name__ == "__main__":
	main()