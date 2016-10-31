# Timothy Goodwin
# tlg2132
# 10-31-16
# Homework 4, problem 2d source code

import numpy as np
import math
from scipy.io import loadmat
hw4data = loadmat('hw4data.mat')

def main():
    data = hw4data['data']
    labels = hw4data['labels']
    PARTITION = int(math.floor(0.8 * data.shape[0]))
    THRESHOLD = 0.65064
    data = linear_transform(data)
    lifted_data = np.insert(data, 0, 1, axis=1)
    training_data = lifted_data[0:PARTITION]
    training_labels = labels[0:PARTITION]
    holdout_data = lifted_data[PARTITION:]
    holdout_labels = labels[PARTITION:]
    gd_handler(training_data, training_labels, holdout_data, holdout_labels)

def inspect_data(data):
    for i in range(data.shape[0]):
        print data[i]

# scales the middle term of the hw4data set
def linear_transform(data):
    A = np.array([[1, 0, 0],[0,12,0],[0,0,1]])
    return np.dot(data, A)

def compute_objective(B, data, labels):
    b_vec = B.reshape(1, 4)
    dot = np.dot(b_vec, data.T)
    log_arg = 1 + np.exp(dot)
    log_term = np.log(log_arg)
    second = labels.reshape(1, len(labels)) * dot
    temp = log_term - second
    thesum = np.sum(temp, axis=1)
    return np.divide(thesum[0], len(labels))

def descend_gradient(B, data, labels):
    b_vec = B
    exp = np.exp(np.dot(b_vec, data.T))
    numerator = data * exp.T
    denominator = 1 + exp
    frac = np.divide(numerator.T, denominator)
    subt = labels.reshape((1, len(labels))) * data.T
    res = frac - subt
    sums = np.sum(res, axis=1)
    gradient_t = np.divide(sums, data.shape[0])
    return gradient_t

# determine next stepsize via backtracking line search
def line_search(data, labels, gradient_b, B):
    stepsize = 1
    lambda_norm = np.dot(gradient_b, gradient_b.T)  # ||lambda||^2 term
    B_diff = B - (stepsize * gradient_b)
    obj_orig = compute_objective(B, data, labels)
    obj_diff = compute_objective(B_diff, data, labels)

    while (obj_diff > obj_orig - (0.5 * stepsize * lambda_norm)):
        stepsize = 0.5 * stepsize
        B_diff = B - (stepsize * gradient_b)
        obj_diff = compute_objective(B_diff, data, labels)
    return stepsize

def is_power_of_two(n):
    return n != 0 and not ( n & (n - 1))

def record_holdout_rate(B, holdout_data, holdout_labels):
    signs = np.dot(B, holdout_data.T)
    preds = np.sign(signs).T
    errs = 0
    total = 0
    for i in range(len(preds)):
        if int(preds[i][0]) == -1:
            preds[i][0] = 0
        if int(preds[i][0]) != int(holdout_labels[i][0]):
            errs += 1
        total += 1
    return float(errs) / float(total)

def gd_handler(data, labels, h_data, h_labels):
    B = np.zeros((1, 4)) # start at zero as specified
    steps = 0
    stepsize = 1
    best_holdout_rate = float('inf')
    obj = compute_objective(B, data, labels) #objective value
    while (1):
        steps += 1
        if is_power_of_two(steps):
            holdout_rate = record_holdout_rate(B, h_data, h_labels)
            print steps, "objective: ", obj, "holdout error rate: ", holdout_rate
            if (holdout_rate > 0.99 * best_holdout_rate and steps >= 32):
                break
            if holdout_rate < best_holdout_rate:
                best_holdout_rate = holdout_rate
        gradient_b = descend_gradient(B, data, labels)
        stepsize = line_search(data, labels, gradient_b, B)
        B = B - (stepsize * gradient_b)
        # update objective afer stepping down gradient
        obj = compute_objective(B, data, labels)

if __name__ == "__main__":
	main()

