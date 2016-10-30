import numpy as np
# import sklearn as sk
from scipy.io import loadmat
hw4data = loadmat('hw4data.mat')

def main():
    print "hey"
    data = hw4data['data']
    lifted_data = np.insert(data, 0, 1, axis=1)
    labels = hw4data['labels']
    B = np.array([1,3.5,4.3,2.3])
    compute_objective(2, B, lifted_data, labels)

def compute_objective(Bo, B, data, labels):
    val = 0
    # b is a vector of length 4
    b_vec = B.reshape(1, 4)
    dot = np.dot(b_vec, data.T)
    # data.T is (4, 4096)
    print dot.shape
    log_arg = 1 + np.exp(dot)
    log_term = np.log(log_arg)
    second = labels.reshape(1, len(labels)) * dot
    temp = log_term - second
    thesum = np.sum(temp, axis=1)
    print thesum[0] / len(labels)
    return thesum[0] / len(labels)

def descend_gradient(Bo, B, data, labels, stepsize):
    return 1

def gd_handler(data, labels, target):
    obj = float('inf')
    B = np.zeros(np.shape(([1, 4])))
    steps = 0
    while (steps < 1000):
        steps += 1
        Beta = descend_gradient(B, data, labels, 10)


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
   
