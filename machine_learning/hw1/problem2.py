# problem2 for homework 1
# Tim Goodwin, tlg2132@columbia.edu
# 9-21-16

import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
ocr = loadmat('ocr.mat')

def get_random_set(collection, labels, n, maxval):
	sel = random.sample(xrange(maxval), n)	#selection of n random numbers
	sample = np.array(collection[sel]) 	# select n vectors at random from the set
	sample_labels = labels[sel]				# parallel array of labels
	return (sample, sample_labels)

# Euclidean distance speedhack: break down SUM(|X_i - Y_j|^2) into SUM(-2X_i*Y_j) + SUM(X_i)^2 + SUM(Y_j)^2
# a lot of transposing here done only to allow for valid matrix arithmetic
def fast_find_neighbor(X, Y, test):
	test = test.T.astype(np.float32)
	X = X.astype(np.float32)
	mid = np.dot(X, test * -2)		# -2X_i * Y_j
	X2 = np.square(X)				# X_i^2
	X2sum = np.sum(X2, axis=1, dtype=np.float32)
	test2 = np.square(test, dtype=np.float32)
	test2sum = np.sum(test2, axis=0, dtype=np.float32)
	temp = mid.T + X2sum
	final = temp.T + test2sum		# matrix of distances btwn each test feature and prototype feature
	answer = np.argmin(final, axis=0)
	preds = Y[answer]				# collapse the array into a vector of the min values
	return preds					# return the nearest neighbors as an array parallel to ocr['testlabels']

def run_test_data(sample, labels, testdata, testlabels):
	errorcount = 0
	totalcount = 0

	NN_labels = fast_find_neighbor(sample, labels, testdata)
	for i in range(len(NN_labels)):
		if NN_labels[i] != testlabels[i]:
			errorcount += 1
		totalcount += 1

	error_rate = float(errorcount) / float(totalcount)
	# print "error rate for sample size %d: %s" % (len(labels), str(error_rate))
	return error_rate

def select_prototypes():
	prototypes = {} # maps sample size to best of 10 (sample, labels)
	training_data = np.array(ocr['data'])
	training_labels = np.array(ocr['labels'])
	valSel = random.sample(xrange(60000), 16000) #pull 10,000 for validation set
	valSet = training_data[valSel]
	valLabels = training_labels[valSel]
	remainder_data = np.delete(training_data, valSel, axis=0)
	remainder_labels = np.delete(training_labels, valSel, axis=0)
	print valSet.shape, valLabels.shape
	print remainder_data.shape, remainder_labels.shape
	for i in range(4):
		sample_size = (2 ** i) * 1000
		error_dict = {}
		min_err = float('inf')
		for j in range(10):
			sample, labels = get_random_set(remainder_data, remainder_labels, sample_size, len(remainder_data))
			error_rate = run_test_data(sample, labels, valSet, valLabels)
			error_dict[error_rate] = (sample, labels)
			# print "evaluated prototype m=%d no. %d of 10 @ error rate %s" % (sample_size, j + 1, str(error_rate))
			if error_rate < min_err:
				min_err = error_rate
				print "new min=%s" % str(min_err)
		best_sample = error_dict[min_err]	#(sample, labels)
		print "training error for m=%d prototype: [%s]" % (sample_size, str(min_err))
		prototypes[sample_size] = best_sample 	#(sample, labels)
	return prototypes

def main():
	results = {}
	for i in range(10):
		print "starting process %d of 10" % (i + 1)
		prototypes = select_prototypes() # returns dictionary
		for sample_size in prototypes:
			best_sample, best_labels = prototypes[sample_size]
			error_rate = run_test_data(best_sample, best_labels, ocr['testdata'], ocr['testlabels'])
			print "prototype m=%s ran with test error: [%s]" % (str(sample_size), str(error_rate))
			if sample_size not in results:
				results[sample_size] = [error_rate]
			else:
				results[sample_size].append(error_rate)

	x = [1000, 2000, 4000, 8000]
	avg = []
	err = []
	for r in results:
		mean = np.sum(results[r]) / len(results[r])
		avg.append(mean)
		std = np.std(results[r])
		err.append(std * 2)
	print "avg", avg
	print "results", results
	print "err = 2 std devs", err
	plt.plot(x, avg)
	plt.errorbar(np.array(x), np.array(avg), np.array(err), linestyle='None', marker='*')
	plt.xlim([0, 10000])
	plt.ylim([0, 0.2])
	plt.xlabel('n')
	plt.ylabel('Error rate')
	plt.show()

if __name__ == "__main__":
    main()