import numpy as np
import math
from scipy.io import loadmat
hw4data = loadmat('hw4data.mat')

def main():
    print "hey"
    data = hw4data['data']
    lifted_data = np.insert(data, 0, 1, axis=1)
    labels = hw4data['labels']
    #compute_objective(B, lifted_data, labels)
    #descend_gradient(B, lifted_data, labels, .0035)
    gd_handler(lifted_data, labels, 0.65064)

def compute_objective(B, data, labels):
    val = 0
    # b is a vector of length 4
    b_vec = B.reshape(1, 4)
    dot = np.dot(b_vec, data.T)
    # data.T is (4, 4096)
    #print dot.shape
    log_arg = 1 + np.exp(dot)
    log_term = np.log(log_arg)
    second = labels.reshape(1, len(labels)) * dot
    temp = log_term - second
    thesum = np.sum(temp, axis=1)
    #print thesum[0] / len(labels)
    return thesum[0] / len(labels)

def descend_gradient(B, data, labels, stepsize):
    errs = np.zeros(B.shape)
    b_vec = B
    exp = np.exp(np.dot(b_vec, data.T))
    #print np.dot(b_vec, data.T)
    #print exp.shape
    #print (data * data).T.shape
    numerator = (data * data) * exp.T
    #print numerator.shape #(4096, 4)
    denominator = 1 + exp
    #print denominator.shape
    frac = numerator.T / denominator
    #print frac.shape
    subt = labels.reshape((1, len(labels))) * data.T
    #print subt.shape
    res = frac - subt
    #print res.shape
    sums = np.sum(res, axis=1)
    gradientVal = sums / len(B)
    #print gradientVal.shape

    # update rule
    b_vec = b_vec - (gradientVal * stepsize)
    #print b_vec
    return b_vec

def gd_handler(data, labels, target):
    B = np.zeros((1, 4))
    print "ok"
    print B.shape
    steps = 0
    while (steps < 1000):
        steps += 1
        stepsize = 0.0035 / math.sqrt(steps)
        B = descend_gradient(B, data, labels, stepsize)
        obj = compute_objective(B, data, labels)
        print"OBJECTIVE VALUE", obj


if __name__ == "__main__":
	main()

# write function to compute the value of the objective function
# write function using the above function to do the gradient descent with the line-search approach
# write function to count step size while the value of the objective is not less than the assignments requested threshold.
# try starting with step size of 0.002


# while obj > threshold
# 	steps += 1
# 	Bo, B = gradientDescent(Bo, B, data, labels, 0.0035/sqrt(steps))
#	obj = computeObjective(Bo, B, data, labels);


# data is 4096x3, add a column of 1s in the front to represent B_o, and now make sure your B vector is of length 4 to match the augmented matrix.
   
