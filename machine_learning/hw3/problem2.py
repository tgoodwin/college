import random
import numpy as np
import sklearn as sk
import pandas as pd
import sklearn

df = pd.read_csv('reviews_tr.csv')
review_docs = df['text']

from sklearn.feature_extraction.text import CountVectorizer


def build_unigram(training_data):
	return representation

