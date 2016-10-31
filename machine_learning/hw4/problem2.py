import numpy as np
import math
from scipy.io import loadmat
hw4data = loadmat('hw4data.mat')

def main():
    data = hw4data['data']
    trans_data = linear_transform(data)
    print trans_data.shape
    lifted_data = np.insert(trans_data, 0, 1, axis=1)
    labels = hw4data['labels']
    B = np.zeros((1, 4))
    #plot_data(lifted_data)
    gd_handler(lifted_data, labels, 0.65064)

def plot_data(data):
    for i in range(data.shape[0]):
        print data[i]

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

def line_search(data, labels, gradient_b, B):
    stepsize = 1
    lambda_norm = np.dot(gradient_b, gradient_b.T)
    B_diff = B - (stepsize * gradient_b)
    obj_orig = compute_objective(B, data, labels)
    obj_diff = compute_objective(B_diff, data, labels)

    while (obj_diff > obj_orig - (0.5 * stepsize * lambda_norm)):
        stepsize = 0.5 * stepsize
        B_diff = B - (stepsize * gradient_b)
        obj_diff = compute_objective(B_diff, data, labels)
    return stepsize

def gd_handler(data, labels, target):
    B = np.zeros((1, 4))
    steps = 0
    stepsize = 1
    obj = 100
    while (obj > target):
        steps += 1
        gradient_b = descend_gradient(B, data, labels)
        stepsize = line_search(data, labels, gradient_b, B)
        B = B - (stepsize * gradient_b)
        obj = compute_objective(B, data, labels)
        print steps, "OBJECTIVE VALUE", obj, "stepsize", stepsize

if __name__ == "__main__":
	main()

