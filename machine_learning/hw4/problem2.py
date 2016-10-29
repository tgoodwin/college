import numpy as np
# import sklearn as sk
from scipi.io import loadmat
data = loadmat('hw4data.mat')

def main():
	return 1

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
   